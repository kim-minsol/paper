import torch
import torch.nn as nn
import torch.nn.functional as F
#from einops.layer.torch import Rearrange
from einops import rearrange, reduce, repeat, einsum
from functools import partial
from collections import namedtuple
import math


# x 존재하는지 확인
def exists(x):
    return x is not None


# x or y 중 존재하는 변수 반환
def default(x, y):
    if exists(x):
        return x
    return y() if callable(y) else y


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim, dim_out), 3, padding=1)
    )


# def Downsample(dim, dim_out=None):
#     return nn.Sequential(
#         Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
#         nn.Conv2d(dim*4, dim_out, 1)
#     )


class Block(nn.Module):
    def __init__(self, dim, dim_out, g):
        '''
        dim / dim_out: dimension of input/output
        g: groups of normalization
        '''
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, 1)
        self.norm = nn.GroupNorm(g, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, g=8, time_emb_dim=None):
        super().__init__()
        if exists(time_emb_dim):
            self.mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, dim_out*2)
            )

        self.block1 = Block(dim, dim_out, g=g)
        self.block2 = Block(dim_out, dim_out, g=g)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity() # x 임베딩 + 차원 맞추기 for connection

    def forward(self, x, time_emb=None):
        scale_shift = None

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift)
        h = self.block2(x)  # scale_shift 사용 x

        return h + self.res_conv(x)


class RMSNorm(nn.Module):
    def __init__(self, d):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        """
        super(RMSNorm, self).__init__()
        self.d = d
        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

    def forward(self, x):
        norm_x = F.normalize(x, dim=1)
        rms_x = norm_x * self.d ** (-1. / 2)

        return rms_x


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


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=8, drop_=0., num_mem_kv = 4):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = heads * dim_head

        self.norm = RMSNorm(dim)
        self.attn_dropout = nn.Dropout(drop_)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim*3, 1)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)

        # out = self.attend(q, k, v)
        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        # attention
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(self, dim, input_channels=3, dim_mults=(1, 2, 4, 8), resnet_groups=8, attn_heads=4, attn_dim_head=32):
        super().__init__()
        self.init = nn.Conv2d(input_channels, dim, 7, padding=3)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]

        # position embedding
        pos_embbder = LearnedSinusoidalPosEmb(dim)
        fourier_dim = dim
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            pos_embbder,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        block_klass = partial(ResnetBlock, g=resnet_groups)
        
        attn_klass = partial(Attention)
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        in_out = zip(dims[:-1], dims[-1:])
        
        for idx, ((dim_in, dim_out), layer_full_attn) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            