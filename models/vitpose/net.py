"""ViTPose as a plain torch module, built from a config dict.

The ONNX export spells the transformer out op by op (2257 nodes for ViTPose-H: every
LayerNorm is ReduceMean/Sub/Pow/Sqrt/Div, every GELU is Div/Erf/Add/Mul). This module runs
the same weights through addmm, F.layer_norm, F.gelu and scaled_dot_product_attention.

The architecture - depth, width, heads, attention scale, LayerNorm epsilon, the patch
embedding and the deconvolution head - is read out of the export once, offline, by
scripts/convert_models.py, and stored as this module's config next to its weights.

config: input_size [h, w], patch_embed (Conv config, see blocks.Conv), num_tokens, dim,
depth, heads, scale (attention scale), mlp_hidden, eps (LayerNorm), gelu ("none" for the
erf GELU, "tanh" for the approximation), head (list of layer configs, each with a "type":
"deconv", "conv", "bn" or "relu").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import Conv, parameter


class Linear(nn.Module):
    """x @ W + b with W kept as the ONNX MatMul stores it, [in, out]; no transposed copy on load."""

    def __init__(self, cin, cout):
        super().__init__()
        self.weight = parameter((cin, cout))
        self.bias = parameter((cout,))

    def forward(self, x):
        return torch.addmm(self.bias, x.reshape(-1, x.shape[-1]), self.weight).reshape(*x.shape[:-1], -1)


class Block(nn.Module):
    def __init__(self, dim, hidden, heads, scale, eps, gelu):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.gelu = gelu
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.qkv = Linear(dim, 3 * dim)
        self.proj = Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.fc1 = Linear(dim, hidden)
        self.fc2 = Linear(hidden, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=self.scale)
        x = x + self.proj(a.transpose(1, 2).reshape(B, N, C))
        return x + self.fc2(F.gelu(self.fc1(self.norm2(x)), approximate=self.gelu))


def _head_layer(cfg):
    kind = cfg["type"]
    if kind == "deconv":
        return nn.ConvTranspose2d(cfg["cin"], cfg["cout"], cfg["kernel"], stride=cfg["stride"],
                                  padding=cfg["padding"], output_padding=cfg["output_padding"],
                                  dilation=cfg["dilation"], bias=cfg["bias"])
    if kind == "conv":
        return Conv(cfg)
    if kind == "bn":
        return nn.BatchNorm2d(cfg["channels"], eps=cfg["eps"])
    if kind == "relu":
        return nn.ReLU()
    raise ValueError(f"ViTPose head layer type must be deconv, conv, bn or relu, found {kind!r}")


class ViTPoseNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        dim = cfg["dim"]
        if dim % cfg["heads"]:
            raise ValueError(f"ViTPose: width {dim} does not split into {cfg['heads']} heads")
        self.patch_embed = Conv(cfg["patch_embed"])
        self.pos_embed = parameter((1, cfg["num_tokens"], dim))
        self.blocks = nn.ModuleList(Block(dim, cfg["mlp_hidden"], cfg["heads"], cfg["scale"], cfg["eps"], cfg["gelu"])
                                    for _ in range(cfg["depth"]))
        self.last_norm = nn.LayerNorm(dim, eps=cfg["eps"])
        self.head = nn.Sequential(*(_head_layer(c) for c in cfg["head"]))
        for module in self.modules():
            for p in module.parameters(recurse=False):
                p.requires_grad_(False)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.last_norm(x).transpose(1, 2).reshape(B, C, H, W)
        return self.head(x)
