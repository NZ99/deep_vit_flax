from functools import partial
from typing import Callable, Any, Optional

import jax
from jax import numpy as jnp
from jax.nn import initializers

from flax import linen as nn
from flax import struct

from einops import rearrange

ModuleDef = Any


@struct.dataclass
class DeepViTConfig:
    num_classes: int = 1000
    depth: int = 32
    mlp_dim: int = 408
    token_dim: int = 64
    emb_dim: int = 768
    num_heads: int = 12
    dim_head: int = 64
    shared_theta: bool = True
    activation_fn: ModuleDef = nn.gelu
    dtype: jnp.dtype = jnp.float32
    precision: Any = jax.lax.Precision.DEFAULT
    kernel_init: Callable = initializers.xavier_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)
    posemb_init: Callable = initializers.normal(stddev=0.02)


class AddPositionEmbs(nn.Module):
    config: DeepViTConfig

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config
        assert inputs.ndim == 3

        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = self.param('patch_pos_embedding', cfg.posemb_init, pos_emb_shape)
        return inputs + pe


class ThetaTransform(nn.Module):
    config: DeepViTConfig

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config
        assert inputs.ndim == 4

        theta_shape = (cfg.num_heads, cfg.num_heads)
        theta = self.param('theta', cfg.kernel_init, theta_shape)

        out = jnp.einsum('h i, b h q k -> b i q k',
                         theta,
                         inputs,
                         precision=cfg.precision)
        return out


class MlpBlock(nn.Module):
    config: DeepViTConfig

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config
        out_dim = inputs.shape[-1]
        dense = partial(nn.Dense,
                        use_bias=True,
                        dtype=cfg.dtype,
                        precision=cfg.precision,
                        kernel_init=cfg.kernel_init,
                        bias_init=cfg.bias_init)

        y = dense(features=cfg.mlp_dim)(inputs)
        y = cfg.activation_fn(y)
        output = dense(features=out_dim)(y)
        return output


class ReAttention(nn.Module):
    config: DeepViTConfig
    theta_transform: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config
        assert inputs.ndim == 3

        dense = partial(nn.DenseGeneral,
                        axis=-1,
                        features=(cfg.num_heads, cfg.dim_head),
                        use_bias=False,
                        kernel_init=cfg.kernel_init,
                        precision=cfg.precision)

        query, key, value = (dense(dtype=cfg.dtype)(inputs),
                             dense(dtype=cfg.dtype)(inputs),
                             dense(dtype=cfg.dtype)(inputs))

        query = query / jnp.sqrt(cfg.dim_head).astype(cfg.dtype)

        attn_weights = jnp.einsum('b q h d, b k h d -> b h q k',
                                  query,
                                  key,
                                  precision=cfg.precision)
        attn_weights = nn.softmax(attn_weights).astype(cfg.dtype)

        if cfg.shared_theta:
            attn_weights = self.theta_transform(attn_weights)
        else:
            attn_weights = ThetaTransform(config=cfg)(attn_weights)

        attn_weights = nn.LayerNorm()(attn_weights)

        out = jnp.einsum('b h q k, b q h d -> b k h d',
                         attn_weights,
                         value,
                         precision=cfg.precision)

        if (cfg.num_heads * cfg.dim_head) != cfg.emb_dim:
            out = nn.DenseGeneral(features=cfg.emb_dim,
                                  axis=(-2, -1),
                                  dtype=cfg.dtype,
                                  precision=cfg.precision,
                                  kernel_init=cfg.kernel_init,
                                  bias_init=cfg.bias_init)(out)
        else:
            out = rearrange(out, 'b k h d -> b k (h d)')

        return out


class DeepViTBlock(nn.Module):
    config: DeepViTConfig
    theta_transform: Optional[nn.Module] = None

    @nn.compact
    def __call__(
        self,
        inputs,
    ):
        cfg = self.config

        residual = inputs
        y = nn.LayerNorm()(inputs)
        y = ReAttention(config=cfg, theta_transform=self.theta_transform)(y)
        y = y + residual
        residual = y
        y = nn.LayerNorm()(y)
        y = MlpBlock(config=cfg)(y)
        out = y + residual
        return out


class DeepViTEncoder(nn.Module):
    config: DeepViTConfig

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config
        assert inputs.ndim == 3

        y = AddPositionEmbs(config=cfg)(inputs)

        if cfg.shared_theta:
            theta_transform = ThetaTransform(config=cfg)
        else:
            theta_transform = None

        for lyr in range(cfg.depth):
            y = DeepViTBlock(config=cfg, theta_transform=theta_transform)(y)

        return y


class DeepViT(nn.Module):
    config: DeepViTConfig

    @nn.compact
    def __call__(self, inputs):
        cfg = self.config

        conv = partial(nn.Conv,
                       dtype=cfg.dtype,
                       precision=cfg.precision,
                       kernel_init=cfg.kernel_init,
                       bias_init=cfg.bias_init)

        patch_embeddings = conv(features=cfg.token_dim,
                                kernel_size=(7, 7),
                                strides=(4, 4),
                                padding=[(2, 2), (2, 2)])(inputs)
        patch_embeddings = conv(features=cfg.token_dim,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                padding=[(1, 1), (1, 1)])(patch_embeddings)
        patch_embeddings = conv(features=cfg.emb_dim,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                padding=[(1, 1), (1, 1)])(patch_embeddings)
        patch_embeddings = rearrange(patch_embeddings, 'b h w d -> b (h w) d')

        b, l, d = patch_embeddings.shape
        cls = self.param('cls', initializers.zeros, (1, 1, d))
        cls = jnp.tile(cls, [b, 1, 1])
        patch_embeddings = jnp.concatenate([cls, patch_embeddings], axis=1)

        x = DeepViTEncoder(config=cfg)(patch_embeddings)
        x = x[:, 0]

        out = nn.Dense(features=cfg.num_classes,
                       dtype=cfg.dtype,
                       precision=cfg.precision,
                       kernel_init=initializers.zeros,
                       bias_init=cfg.bias_init)(x)
        return out
