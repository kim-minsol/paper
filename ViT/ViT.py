import torch
from torch import nn
from einops import rearrange, repeat


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_heads=64, dropout=0.):
        super().__init__()
        self.heads = heads
        inner_dim = heads * dim_heads

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)
        self.scale = (dim / heads) ** -0.5

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax()

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x) # B X N X D
        qkv = self.to_qkv(x).chunk(3, dim=-1) # 3 X [B X N X (heads X Dim_heads)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)  # B X heads X N X Dim_heads
        dots = q @ k.transpose(-1, -2) * self.scale # B X heads X N X N
        att = self.softmax(dots)
        att = self.dropout(att)
        SA = att @ v # B X heads X N X Dim_heads

        out = rearrange(SA, 'b h n d -> b n (h d)')

        return self.to_out(out) # B X N X D


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_heads, mlp_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList(
                Attention(dim, heads, dim_heads, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, dim, depth, mlp_dim, pool, image_size, patch_size, num_classes, channels=3, num_heads=8, dim_heads=64, dropout=0.):
        super().__init__()
        patch_dim = patch_size ** 2 * channels
        self.to_patch_embedding = nn.Seqential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        num_patches = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(dim, depth, heads, dim_heads, mlp_dim, dropout)

        self.pool = pool

        self.to_latent = nn.Identity()

        self.mlp = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = rearrange(x, 'b c (h p) (w p) -> b (h w) (p p c)', p=patch_size)
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_token = repeat(self.cls_token, '1 1 d -> b n d', b=b)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embedding
        
        x = self.transformer(x)

        if self.pool == 'cls':
            x = x[:, 0]
        elif self.pool == 'avg':
            x = x.mean(dim=1)
        
        x = self.to_latent(x)

        return self.mlp(x)


