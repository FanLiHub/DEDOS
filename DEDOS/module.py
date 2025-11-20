import logging
import os
import warnings
from typing import Callable, Optional, Union
from torch import Tensor
from torch import nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self,
                 dim: int,
                 cross_dim: Optional[int] = None,
                 num_heads: int = 8,
                 dim_head: int = 64,
                 qk_scale=None,
                 qkv_bias: bool = False,
                 proj_bias: bool = True,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 cross_attention_norm: bool = False,
                 sr_ratio=1):
        super().__init__()
        self.dim = dim

        self.cross_dim = cross_dim if cross_dim is not None else dim
        self.cross_attention_norm = cross_attention_norm
        if cross_attention_norm:
            self.norm_cross = nn.LayerNorm(self.cross_dim)

        self.num_heads = num_heads
        self.head_dim = dim_head
        self.scale = qk_scale or self.head_dim**-0.5
        self.inner_dim = dim_head * num_heads

        self.q = nn.Linear(dim, self.inner_dim, bias=qkv_bias)
        self.kv = nn.Linear(self.cross_dim, self.inner_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop_value = attn_drop
        self.proj = nn.Linear(self.inner_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                self.cross_dim, self.cross_dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(self.cross_dim)

    def forward(self, x, y, y_H=None, y_W=None):
        dtype=x.dtype
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        if y is None:
            y = x
        elif self.cross_attention_norm:
            y = self.norm_cross(y)

        if self.sr_ratio > 1:
            y_C=y.shape[-1]
            x_ = y.permute(0, 2, 1).contiguous().reshape(B, y_C, y_H, y_W)
            x_ = self.sr(x_).reshape(B, y_C, -1).permute(0, 2, 1).contiguous()        ## h=H/sr_ratio
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(y).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        # compute attn
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, self.inner_dim)
        # ## torch2.0+
        # x = F.scaled_dot_product_attention(
        #     q, k, v, attn_mask=None, dropout_p=self.attn_drop_value, is_causal=False
        # )
        # x = x.transpose(1, 2).reshape(B, -1, C).to(dtype)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x