import flax.linen as nn
from jax import numpy as np
from einops import rearrange

ATTN_MASK_VALUE = -1e10

class LocalAttention(nn.Module):
    dim: int
    window_size: int
    heads: int = 8
    dim_head: int = 64

    @nn.compact
    def __call__(self, x):
        n, h, dim_head, wsz = x.shape[0], self.heads, self.dim_head, self.window_size
        assert (n % wsz) == 0, 'sequence length must be divisible by the window size'
        scale = dim_head ** -0.5
        window = n // wsz

        qkv = nn.Dense(features = 3 * h * dim_head, use_bias = False)(x)
        q, k, v = np.split(qkv, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, '(w n) (h d) -> h w n d', w = window, h = h), (q, k, v))

        k, v = map(lambda t: np.pad(t, ((0, 0), (1, 0), (0, 0), (0, 0)), constant_values = 0.), (k ,v))
        k, v = map(lambda t: np.concatenate((t[:, :-1], t[:, 1:]), axis = 2), (k, v))

        sim = np.einsum('h w i d, h w j d -> h w i j', q, k) * scale

        mask = np.tril(np.ones((wsz, wsz * 2)), wsz)
        sim = np.where(mask, sim, ATTN_MASK_VALUE)

        attn = nn.softmax(sim, axis = -1)
        out = np.einsum('h w i j, h w j d -> h w i d', attn, v)
        out = rearrange(out, 'h w n d -> (w n) (h d)')
        out =  nn.Dense(features = self.dim)(out)
        return out
