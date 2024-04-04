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

# utils
def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    else:
        raise ValueError("error with dimensions")

# utils
def zero_module(module):
    for params in module.parameters():
        params.detach().zero_()
    return module