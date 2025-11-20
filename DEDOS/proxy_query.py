from functools import reduce, partial
from operator import mul
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from timm.models.layers import DropPath


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):

        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 mask_ratio: float = 0.3
                 ):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divided by num_heads {num_heads}.'

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.mask_ratio = mask_ratio

    def forward(self, tokens: Tensor, feats: Tensor, H: int, W: int):
        B, Nq, C = tokens.shape
        Bf, Nf, Cf = feats.shape
        assert B == Bf and C == Cf, \
            f"tokens({tokens.shape}) and feats({feats.shape}) must match in batch and dim"

        q = self.q(tokens).reshape(B, Nq, self.num_heads,
                                   C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        if self.sr_ratio > 1:
            x_ = feats.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1).contiguous()
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                2, 0, 3, 1, 4).contiguous()
        else:
            kv = self.kv(feats).reshape(B, -1, 2, self.num_heads,
                                        C // self.num_heads).permute(
                2, 0, 3, 1, 4).contiguous()

        k, v = kv[0], kv[1]
        Nk = k.size(2)

        if self.mask_ratio > 0.0:
            keep_prob = 1.0 - self.mask_ratio
            M = torch.bernoulli(
                torch.full((B, 1, Nk, 1), keep_prob, device=k.device, dtype=k.dtype)
            )
            k = k * M
            v = v * M

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, tokens: Tensor, feats: Tensor, H: int, W: int):
        tokens = tokens + self.drop_path(
            self.attn(self.norm1(tokens), feats, H, W)
        )

        B, Nq, C = tokens.shape
        tokens = tokens + self.drop_path(
            self.mlp(self.norm2(tokens), H=1, W=Nq)
        )
        return tokens



class object_query(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embed_dims: int,
        patch_size: int,
        query_dims: int = 256,
        token_length: int = 100,
        use_softmax: bool = True,
        link_token_to_query: bool = True,
        scale_init: float = 0.001,
        zero_mlp_delta_f: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.query_dims = query_dims
        self.token_length = token_length
        self.link_token_to_query = link_token_to_query
        self.scale_init = scale_init
        self.use_softmax = use_softmax
        self.zero_mlp_delta_f = zero_mlp_delta_f

        self.create_model()

    def create_model(self):

        self.learnable_tokens1 = nn.Parameter(
            torch.empty([self.num_layers, self.token_length, self.embed_dims])
        )

        self.scale = nn.Parameter(torch.tensor(self.scale_init))
        self.mlp_token2feat = nn.Linear(self.embed_dims, self.embed_dims)
        self.mlp_delta_f = nn.Linear(self.embed_dims, self.embed_dims)

        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (self.patch_size, self.patch_size), 1)
                + self.embed_dims
            )
        )

        nn.init.uniform_(self.learnable_tokens1.data, -val, val)
        nn.init.kaiming_uniform_(self.mlp_delta_f.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.mlp_token2feat.weight, a=math.sqrt(5))

        self.transform = nn.Linear(self.embed_dims, self.query_dims)
        self.merge = nn.Linear(self.query_dims * 3, self.query_dims)

        if self.zero_mlp_delta_f:
            del self.scale
            self.scale = 1.0
            nn.init.zeros_(self.mlp_delta_f.weight)
            nn.init.zeros_(self.mlp_delta_f.bias)

        self.block = nn.ModuleList([
            Block(
                dim=self.embed_dims,
                num_heads=8,
                mlp_ratio=4,
                qkv_bias=True,
                qk_scale=None,
                drop=0,
                attn_drop=0,
                drop_path=0.1,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                sr_ratio=1)
            for _ in range(self.num_layers)
        ])

    def get_tokens(self, layer: int) -> Tensor:

        if layer == -1:
            return self.learnable_tokens1
        else:
            return self.learnable_tokens1[layer]

    def return_auto(self):
        tokens = self.transform(self.get_tokens(-1))
        tokens = tokens.permute(1, 2, 0)

        max_pooled = F.max_pool1d(tokens, kernel_size=self.num_layers)
        avg_pooled = F.avg_pool1d(tokens, kernel_size=self.num_layers)
        last = tokens[:, :, -1].unsqueeze(-1)

        merged = torch.cat(
            [max_pooled, avg_pooled, last],
            dim=-1,
        )

        querys = self.merge(merged.flatten(-2, -1))
        return querys



    def forward(
        self,
        feats: Tensor,
        layer: int,
        batch_first: bool = True,
        has_cls_token: bool = False,
    ) -> Tensor:

        if not batch_first:
            feats = feats.permute(1, 0, 2)

        if has_cls_token:
            cls_token, feats = torch.tensor_split(feats, [1], dim=1)

        B, Nf, C = feats.shape
        HW = int(math.sqrt(Nf))
        H = W = HW

        tokens = self.get_tokens(layer)
        tokens = tokens.unsqueeze(0).expand(B, -1, -1).contiguous()


        self.learnable_tokens[layer] = self.block[layer](tokens, feats, H, W)
