import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layer.torch import Rearrange
from einops import rearrange, reduce, repeat, einsum
from model import *


##### test Attention module
# att = Attention(128)
# print(att)

# tensor = torch.rand(1, 128, 2, 2)
# print(tensor.size())
# print(att(tensor).size())



##### test Upsample
# u = Upsample(128)
# tensor = torch.rand(1, 128, 2, 2)
# print(tensor.size())
# print(u(tensor).size())


###### dims
dim = 128
dim_mults = (1, 2, 4, 8)
dims = [dim, *map(lambda m: dim * m, dim_mults)]

full_attn = (*((False,) * (len(dim_mults) - 1)), True)
print(full_attn)