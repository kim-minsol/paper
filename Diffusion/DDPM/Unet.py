import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import *


class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.dims = dims
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv_nd(dims, channels, self.out_channels, 3, padding=padding)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shapep[2], x.shape[3] * 2, x.shape[4] * 2), mode = "nearest"
            )
        else:
            x = F.interpolate(x, scale_factor = 2, mode = "nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.dims = dims
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2 if dims != 3 else (1, 2, 2)

        if use_conv:
            self.layer = conv_nd(dims, channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            assert self.channels == self.out_channels
            self.layer = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.layer(x)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shapep[2], x.shape[3] * 2, x.shape[4] * 2), mode = "nearest"
            )
        else:
            x = F.interpolate(x, scale_factor = 2, mode = "nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class TimestepBlock(nn.Module): # 꼭 필요한 class인지...?
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class SpatialTransforemr(nn.Module): # use for image-to-image task
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    def forward(self, x, context):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


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
                 use_scale_shift_norm=False,
                 up=False,
                 down=False
                 ):
        super().__init__()

        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
              nn.GroupNorm(32, channels),
              nn.SiLU(),
              conv_nd(dims, channels, out_channels, 3, padding=1),
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

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
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)

        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[...,None] # make shape same

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h



# timestep embedding
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    sinusoidal timestep embeddings
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half,dtype=torch.float32)).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


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
            self.n_heads = channels // num_head_ch
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
                 resblock_updown=False,
                 conv_resample=True,
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

        self.model_channels = model_channels
        self.dtype = torch.float16
        self.use_conv_upsample = conv_resample
        self.use_conv_downsample = conv_resample

        # Attention: num_heads or num_head_channels 지정하기
        assert attn_heads and attn_dim_head == -1

        # position embedding
        time_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
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

        input_block_channels = [model_channels]
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
                    if attn_heads == -1:
                        attn_dim_head = ch // attn_heads
                    else:
                        attn_heads = ch // attn_dim_head
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads = attn_heads,
                            num_head_ch = attn_dim_head,
                            use_checkpoint = use_checkpoint
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(ch)
            
            if level != len(channel_mults) -1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResnetBlock(
                            ch,
                            time_dim,
                            dropout,
                            out_channels = mult * ch,
                            dims = dims,
                            use_checkpoint = use_checkpoint,
                            use_scale_shift_norm = use_scale_shift_norm,
                            down=True,
                        ) if resblock_updown
                        else Downsample(
                            ch, self.use_conv_downsample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_channels.append(ch)
                ds *= 2
        
        if attn_heads == -1:
            attn_dim_head = ch // attn_heads
        else:
            attn_heads = ch // attn_dim_head
        
        self.middle_block = TimestepEmbedSequential(
            ResnetBlock(
                ch,
                time_dim,
                dropout,
                out_channels=ch,
                dims = dims,
                use_checkpoint = use_checkpoint,
                use_scale_shift_norm = use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads = attn_heads,
                num_head_ch = attn_dim_head,
                use_checkpoint = use_checkpoint
            ),
            ResnetBlock(
                ch,
                time_dim,
                dropout,
                out_channels=ch,
                dims = dims,
                use_checkpoint = use_checkpoint,
                use_scale_shift_norm = use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mults))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_channels.pop()
                layers = [
                    ResnetBlock(
                        ch + ich,
                        time_dim,
                        dropout,
                        out_channels = mult * model_channels,
                        dims = dims,
                        use_checkpoint = use_checkpoint,
                        use_scale_shift_norm = use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attn_resolutions:
                    if attn_heads == -1:
                        attn_dim_head = ch // attn_heads
                    else:
                        attn_heads = ch // attn_dim_head
                    layers.append(
                        AttentionBlock(
                            ch,
                            num_heads = attn_heads,
                            num_head_ch = attn_dim_head,
                            use_checkpoint = use_checkpoint
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResnetBlock(
                            ch,
                            time_dim,
                            dropout,
                            out_channels = out_ch,
                            dims = dims,
                            use_checkpoint = use_checkpoint,
                            use_scale_shift_norm = use_scale_shift_norm,
                            up = True,
                        ) if resblock_updown
                        else Upsample(
                            ch, self.use_conv_upsample, dims=dims, out_channels=out_ch
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
        
        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            zero_module(
                  conv_nd(dims, model_channels, out_channels, 3, padding=1)
              ),
        )
    
    def forward(self, x, timesteps, x_self_cond=None):
        t_emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_mlp(t_emb)

        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)

        return self.out(h)

        ### TODO 1.  Training iteration 돌려보기 !!
        ### TODO 2.  Training 시 log 작성하기 !!
        ### TODO 3.  x_self_cond 구현하기 !!
