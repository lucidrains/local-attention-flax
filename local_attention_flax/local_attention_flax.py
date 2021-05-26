import flax.linen as nn
from jax import numpy as np
from einops import rearrange

class LocalAttention(nn.Module):
    dim: int
    window_size: int
    heads: int = 8
    dim_head: int = 64

    @nn.compact
    def __call__(self, x):
        return x
