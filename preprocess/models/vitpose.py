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
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Conv, parameter


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


# --- test-time flip ------------------------------------------------------------------------

# COCO-WholeBody's mirror: FLIP_INDEX[k] is the keypoint that k is in the mirrored image (left
# and right swapped; nose, the face centre line and the chin stay). Derived from the `swap`
# names of mmpose's configs/_base_/datasets/coco_wholebody.py (open-mmlab/mmpose @ 2a0a2d2),
# the dataset meta ViTPose's wholebody configs read: body 0-16, feet 17-22 (left big toe,
# small toe, heel <-> right), face 23-90 (68 points, mirrored across the centre line), left
# hand 91-111 <-> right hand 112-132.
FLIP_INDEX = (
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19, 39, 38, 37, 36,
    35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 50, 51,
    52, 53, 58, 57, 56, 55, 54, 68, 67, 66, 65, 70, 69, 62, 61, 60, 59, 64, 63, 77, 76, 75, 74, 73, 72,
    71, 82, 81, 80, 79, 78, 87, 86, 85, 84, 83, 90, 89, 88, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111)


def flip_back(heatmaps):
    """The heatmaps [N, 133, h, w] of a mirrored crop, back in the crop's own frame: each
    keypoint takes its mirror partner's map, flipped left-right, then shifted one column right.

    This is ViTPose's own test-time flip (ViTAE-Transformer/ViTPose @ c050ed2,
    mmpose/core/post_processing/post_transforms.py flip_back and
    topdown_heatmap_simple_head.py inference_model), with shift_heatmap=True as its
    ViTPose_huge_wholebody_256x192 config sets it. The shift belongs to the non-UDP decode
    that config and decode_heatmaps use: mirroring a heatmap maps column x to w - 1 - x while
    the image mirror maps it to about w - x (less a quarter column at stride 4), so the
    flipped-back map sits about one column to the left."""
    out = np.ascontiguousarray(heatmaps[:, list(FLIP_INDEX), :, ::-1])
    out[..., 1:] = out[..., :-1].copy()
    return out
