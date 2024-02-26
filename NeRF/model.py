import torch
import torch.nn as nn
import numpy as np


class NeRF(nn.Module):
    def __init__(self, embedding_dim_pos=10, embedding_dim_dir=4, hidden_dim=256):
        super(NeRF, self).__init__()

        self.block1 = nn.Sequential(nn.Linear(embedding_dim_pos * 6, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        ,)

        self.block2 = nn.Sequential(nn.Linear(embedding_dim_pos * 6 + hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim + 1)
        ,)

        self.block3 = nn.Sequential(nn.Linear(embedding_dim_dir * 6 + hidden_dim, hidden_dim // 2), nn.ReLU(),)

        self.layer = nn.Sequential(nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(),)

        self.relu = nn.ReLU()

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_dir = embedding_dim_dir

    def positional_encoding(self, x, L):
        out = []
        for i in range(L):
            out.append(torch.sin(2 ** i * x))
            out.append(torch.cos(2 ** i * x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        emb_pts = self.positional_encoding(o, self.embedding_dim_pos)
        emb_dir = self.positional_encoding(d, self.embedding_dim_dir)

        h = self.block1(emb_pts)
        out = self.block2(torch.cat([h, emb_pts], dim=-1))
        h, sigma = out[:, :-1], self.relu(out[:, -1])
        h = self.block3(torch.cat([h, emb_dir], dim=-1))
        rgb = self.layer(h)

        return rgb, sigma