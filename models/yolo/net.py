"""YOLOv10 as a plain torch module, built from a config dict.

The ONNX export (601 nodes for YOLOv10x) is Ultralytics' YOLOv10 with its NMS-free
one-to-one head and the top-k post-processing baked in: it returns [N, 300, 6] rows of
x1, y1, x2, y2 in input pixels, score and class id, best first. This module computes the
same rows from the same weights.

Backbone and neck are the same family as RTMW's, and use the same blocks: SiLU
convolutions, residual convolution chains (the bottleneck, the CIB and SCDown are all
`ConvChain`), an SPPF pyramid and the PAFPN neck, here without the reduce convolutions.
What is YOLO's own is the C2f merge layer, the PSA attention block at the end of the
backbone and the detection head.

Arithmetic as the export spells it: SiLU is `x * sigmoid(x)`, not F.silu.

The architecture is read out of the export once, offline, by scripts/convert_models.py,
and stored as this module's config next to its weights.

config: input_size [h, w], backbone (list of layer configs, each with a "type": "conv",
"chain", "c2f", "spp" or "psa"), features (indices of the backbone layers the neck reads,
finest first), neck (PAFPN config with "c2f" / "conv" / "chain" layers), head (Detect config).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import PAFPN, Conv, ConvChain, SPPBottleneck, conv_chains, parameter


class _SplitInput(nn.Module):
    """The input stage C2f and PSA share: cv1, and its output split in two along the channels.

    config: cv1 (Conv config), split [a, b]."""

    def __init__(self, cfg):
        super().__init__()
        self.cv1 = Conv(cfg["cv1"])
        self.split = list(cfg["split"])

    def halves(self, x):
        """cv1 of `x`, split along the channels into the two halves."""
        return self.cv1(x).split(self.split, dim=1)


class C2f(_SplitInput):
    """One convolution split in two; the second half runs through the blocks one after the
    other, and every intermediate is kept and concatenated for the closing convolution.

    config: cv1, cv2 (Conv configs), split [a, b], blocks [ConvChain configs]."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.blocks = conv_chains(cfg["blocks"])
        self.cv2 = Conv(cfg["cv2"])

    def forward(self, x):
        y = list(self.halves(x))
        for block in self.blocks:
            y.append(block(y[-1]))
        return self.cv2(torch.cat(y, dim=1))


class Attention(nn.Module):
    """Multi-head self-attention over the pixels of a map, with a depthwise convolution of
    the values added back as a position encoding. q, k and v come out of one 1x1 convolution
    laid out per head as [key_dim q | key_dim k | head_dim v].

    config: qkv, pe, proj (Conv configs), heads, key_dim, head_dim, scale."""

    def __init__(self, cfg):
        super().__init__()
        self.qkv = Conv(cfg["qkv"])
        self.pe = Conv(cfg["pe"])
        self.proj = Conv(cfg["proj"])
        self.heads = cfg["heads"]
        self.sizes = [cfg["key_dim"], cfg["key_dim"], cfg["head_dim"]]
        self.scale = cfg["scale"]

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(x).reshape(B, self.heads, sum(self.sizes), H * W).split(self.sizes, dim=2)
        attn = torch.softmax((q.transpose(-2, -1) @ k) * self.scale, dim=-1)
        x = (v @ attn.transpose(-2, -1)).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        return self.proj(x)


class PSA(_SplitInput):
    """Half the channels pass, half go through attention and a two-layer feed-forward, each
    with a residual, and the halves are merged by a 1x1 convolution.

    config: cv1, cv2 (Conv configs), split [a, b], attn (Attention config), ffn (ConvChain
    config, no residual: the residual is added here)."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.attn = Attention(cfg["attn"])
        self.ffn = ConvChain(cfg["ffn"])
        self.cv2 = Conv(cfg["cv2"])

    def forward(self, x):
        a, b = self.halves(x)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat([a, b], dim=1))


_LAYERS = {"conv": Conv, "chain": ConvChain, "c2f": C2f, "spp": SPPBottleneck, "psa": PSA}


def _layer(cfg):
    if cfg["type"] not in _LAYERS:
        raise ValueError(f"YOLO layer type must be one of {sorted(_LAYERS)}, found {cfg['type']!r}")
    return _LAYERS[cfg["type"]](cfg)


class Detect(nn.Module):
    """YOLOv10's one-to-one head and the post-processing the export bakes in.

    Per level a box branch predicts 4 x reg_max distance logits and a class branch
    num_classes logits. The distances are decoded by the distribution focal loss (a softmax
    over reg_max bins, then the expected bin through a 1x1 convolution with weights
    0..reg_max-1), turned into x1y1x2y2 around each anchor point and scaled by the level's
    stride. Then the export's top-k: the max_det anchors with the best class score, and of
    their max_det x num_classes scores the best max_det, as (box, score, class) rows.

    config: box, cls ([ConvChain config] per level), reg_max, num_classes, strides [per
    level], max_det."""

    def __init__(self, cfg):
        super().__init__()
        self.box = conv_chains(cfg["box"])
        self.cls = conv_chains(cfg["cls"])
        self.reg_max = cfg["reg_max"]
        self.num_classes = cfg["num_classes"]
        self.strides = list(cfg["strides"])
        self.max_det = cfg["max_det"]
        self.dfl = parameter((1, self.reg_max, 1, 1))

    def anchors(self, feats):
        """Anchor points (pixel centres in grid units) and their strides, level by level,
        as the export's constants lay them out: [2, A] and [1, A]."""
        points, strides = [], []
        for x, stride in zip(feats, self.strides):
            h, w = x.shape[2:]
            sx = torch.arange(w, device=x.device, dtype=x.dtype) + 0.5
            sy = torch.arange(h, device=x.device, dtype=x.dtype) + 0.5
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            points.append(torch.stack([gx.reshape(-1), gy.reshape(-1)]))
            strides.append(torch.full((1, h * w), float(stride), device=x.device, dtype=x.dtype))
        return torch.cat(points, dim=1), torch.cat(strides, dim=1)

    def forward(self, feats):
        B = feats[0].shape[0]
        channels = 4 * self.reg_max + self.num_classes
        out = torch.cat([torch.cat([box(x), cls(x)], dim=1).reshape(B, channels, -1)
                         for x, box, cls in zip(feats, self.box, self.cls)], dim=2)
        dist, logits = out.split([4 * self.reg_max, self.num_classes], dim=1)
        A = dist.shape[2]
        dist = torch.softmax(dist.reshape(B, 4, self.reg_max, A).transpose(2, 1), dim=1)
        dist = F.conv2d(dist, self.dfl).reshape(B, 4, A)
        lt, rb = dist.split([2, 2], dim=1)
        points, strides = self.anchors(feats)
        boxes = torch.cat([points - lt, points + rb], dim=1) * strides
        pred = torch.cat([boxes, torch.sigmoid(logits)], dim=1).transpose(1, 2)

        boxes, scores = pred.split([4, self.num_classes], dim=-1)
        _, index = torch.topk(scores.amax(dim=-1), self.max_det, dim=-1)
        index = index.unsqueeze(-1)
        boxes = torch.gather(boxes, 1, index.repeat(1, 1, 4))
        scores = torch.gather(scores, 1, index.repeat(1, 1, self.num_classes))
        scores, index = torch.topk(scores.flatten(1), self.max_det, dim=-1)
        labels = index % self.num_classes
        index = torch.div(index, self.num_classes, rounding_mode="trunc")
        boxes = torch.gather(boxes, 1, index.unsqueeze(-1).repeat(1, 1, 4))
        return torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1).to(boxes.dtype)], dim=-1)


class YOLOv10Net(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.backbone = nn.ModuleList(_layer(c) for c in cfg["backbone"])
        self.features = list(cfg["features"])
        self.neck = PAFPN(cfg["neck"], _layer)
        self.head = Detect(cfg["head"])

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            if i in self.features:
                feats.append(x)
        return self.head(self.neck(*feats))
