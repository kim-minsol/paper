import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import *


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransforemr):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self,
                 channels,
                 emb_channels,
                 dropout,
                 out_channels,
                 dims=2,
                 g=8,
                 time_emb_dim=None,
                 use_checkpoint=False,
                 use_scale_shift_norm=False):
        super().__init__()

        self.use_scale_shift_norm = use_scale_shift_norm

        self.layers = nn.Sequential(
              nn.GroupNorm(32, channels),
              nn.SiLU(),
              conv_nd(dims, channels, out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
              nn.SiLU(),
              nn.Linear(emb_channels, 2 * out_channels if use_scale_shift_norm else out_channels)
        )

        self.out_layers = nn.Sequential(
              nn.GroupNorm(32, out_channels),
              nn.SiLU(),
              nn.Dropout(p=dropout),
              zero_module(
                  conv_nd(dims, channels, out_channels, 3, padding=1)
              ),
        )

        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = conv_nd(dims, channels, out_channels, 1)

    def forward(self, x, emb):
        h = self.layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[...,None] # 필요한 함수인지 확인할 것 !
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        
        return self.skip_connection(x) + h




# time embedding module
class LearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered


class QKVAttention(nn.Module):
    """
    QKV attention which splits heads before qkv
    """
    def __init__(self, heads):
        super().__init__()
        self.n_heads = heads

    def forward(self, qkv):
        batch_size, w, length = qkv.shape # (B x (H * 3 * C) x T)
        assert w % (3 * self.n_heads) == 0 # split heads 가능한지 확인!
        ch = w // (3 * self.n.heads)
        q, k, v = qkv.reshape(batch_size * self.n_heads, ch * 3, length).split(ch, dim=1) # (B * n_heads, ch, length)
        scale = 1. / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        attn = torch.einsum("bts,bcs->bct", weight, v)
        return attn.reshape(batch_size, -1, length)


class AttentionBlock(nn.Module):
    """
    """
    def __init__(self,
                 channels,
                 num_heads=1,
                 num_head_ch=-1,
                 use_checkpoint=False
                 ):
        super().__init__()

        # setting hyperparameters
        self.ch = channels
        if num_heads != 1:
            assert channels % num_head_ch == 0
            self.num_heads = channels // num_head_ch
        else:
            self.n_heads = num_heads

        # normalization
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.n_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

        def forward(self, x):
            b, c, *spatial = x.shape
            x = x.reshape(b, c, -1)
            qkv = self.qkv(self.norm(x))
            h = self.attention(qkv)
            h = self.proj_out(h)
            return (x + h).reshape(b, c, *spatial)
        ### TODO: make utils dir, conv_nd class, zero_module class


class Unet(nn.Module):
    def __init__(self,
                 input_channels,
                 model_channels,
                 out_channels,
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0,
                 channel_mults=(1, 2, 4, 8),
                 dims=2,
                 resnet_groups=8,
                 attn_heads=-1,
                 attn_dim_head=-1,
                 num_classes=None,
                 use_checkpoint=False,
                 use_scale_shift_norm=False,
                 ):
        '''
        input_channels: dim of inputs.
        model_channels: dim of model in initial.
        channel_mults:
        dims: signal is 1D or 2D or 3D.
        resnet_groups: groups of normalization.
        attn_heads / attn_dim_head: num of heads, head dimension in AttentionBlock.
        num_classes: if exists, this model will be class-conditional.
        use_checkpoint: use gradient checkpointing.
        use_scale_shift_norm: use a FiLM-like conditioning mechanism.


        '''
        super().__init__()
        # Attention: num_heads or num_head_channels 지정하기
        assert attn_heads and attn_dim_head == -1

        # position embedding
        #pos_embbder = LearnedSinusoidalPosEmb(dim)
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            #pos_embbder,
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        if exists(num_classes):
            self.label_emb = nn.Embedding(num_classes, time_dim)


        self.input_blocks = nn.ModuleList(
            TimestepEmbedSequential(
                conv_nd(dims, input_channels, model_channels, 3, padding=1)
            )
        )

        self.feature_size = model_channels
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mults):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(
                        ch,
                        time_dim,
                        dropout,
                        out_channels = mult * ch,
                        dims = dims,
                        use_checkpoint = use_checkpoint,
                        use_scale_shift_norm = use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attn_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads = attn_heads,
                            num_head_ch = attn_dim_head,
                            use_checkpoint = use_checkpoint
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
            if level != len(channel_mults) -1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ### TODO !!! Downsample class 정의하기, AttentionBlock heads, dim 선언 후 넣어주기




