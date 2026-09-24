"""The convolutional blocks of YOLOv10, built from a config dict.

A SiLU convolution everywhere, chains of convolutions with an optional residual, a spatial
pyramid of max pools, and a PAFPN neck that upsamples from the coarsest level down and
downsamples back up. The layer that merges a level (YOLO's C2f) is the model's own, so the
neck takes that layer's constructor from the model that owns it.

Every class here takes its config dict and allocates its own parameters empty, so a model
can be built on the meta device and filled from a checkpoint without a second copy of the
weights. The config is plain JSON (ints, floats, strings, lists, dicts), which is what the
checkpoint stores next to the weights.

The arithmetic is written the way the ONNX exports spell it, not with the fused kernels
that look equivalent: SiLU is `x * sigmoid(x)`, not F.silu (`x / (1 + exp(-x))`). The two
differ in the last bit, and over a hundred convolutions that difference compounds.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = (None, "silu", "relu")


def parameter(shape):
    """An empty frozen parameter; the checkpoint fills it."""
    return nn.Parameter(torch.empty(*shape), requires_grad=False)


# SiLU as the exports spell it, never F.silu (see the module docstring)
def silu(x):
    return x * torch.sigmoid(x)


class Conv(nn.Module):
    """One ONNX Conv and the activation the export writes after it, as a single module.

    config: cin, cout, kernel [kh, kw], stride, padding, dilation (pairs), groups, bias
    (bool), act (None, "silu" or "relu")."""

    def __init__(self, cfg):
        super().__init__()
        if cfg["act"] not in ACTIVATIONS:
            raise ValueError(f"Conv activation must be one of {ACTIVATIONS}, found {cfg['act']!r}")
        if cfg["cin"] % cfg["groups"]:
            raise ValueError(f"Conv: {cfg['cin']} input channels do not split into {cfg['groups']} groups")
        self.weight = parameter((cfg["cout"], cfg["cin"] // cfg["groups"], *cfg["kernel"]))
        self.bias = parameter((cfg["cout"],)) if cfg["bias"] else None
        self.stride = tuple(cfg["stride"])
        self.padding = tuple(cfg["padding"])
        self.dilation = tuple(cfg["dilation"])
        self.groups = cfg["groups"]
        self.act = cfg["act"]

    def forward(self, x):
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.act == "silu":
            return silu(x)
        if self.act == "relu":
            return F.relu(x)
        return x


class ConvChain(nn.Module):
    """Convolutions run one after the other, with the input added back at the end when the
    export has the residual. YOLO's bottleneck (two 3x3), its CIB (depthwise / pointwise, five deep), its SCDown and every
    branch of its detection head are all this.

    config: convs [Conv config, ...], residual (bool)."""

    def __init__(self, cfg):
        super().__init__()
        if not cfg["convs"]:
            raise ValueError("ConvChain needs at least one convolution")
        self.convs = nn.ModuleList(Conv(c) for c in cfg["convs"])
        self.residual = cfg["residual"]

    def forward(self, x):
        out = x
        for conv in self.convs:
            out = conv(out)
        return out + x if self.residual else out


def conv_chains(cfgs):
    """One ConvChain per config of `cfgs`, as a ModuleList."""
    return nn.ModuleList(ConvChain(c) for c in cfgs)


class SPPBottleneck(nn.Module):
    """A convolution, max pools over its output concatenated with it, and a convolution.

    Without `cascade` every pool reads the same map (SPP); with it each pool reads the last
    pool's output again with the same kernel (SPPF, what YOLO exports). The model file
    stores which one it is.

    config: conv1, conv2 (Conv configs), kernels [k, ...], cascade (bool)."""

    def __init__(self, cfg):
        super().__init__()
        self.conv1 = Conv(cfg["conv1"])
        self.conv2 = Conv(cfg["conv2"])
        self.kernels = list(cfg["kernels"])
        self.cascade = cfg["cascade"]

    def forward(self, x):
        x = self.conv1(x)
        pooled = [x]
        for k in self.kernels:
            src = pooled[-1] if self.cascade else x
            pooled.append(F.max_pool2d(src, k, stride=1, padding=k // 2))
        return self.conv2(torch.cat(pooled, dim=1))


def upsample(x, scale):
    """The export's nearest Resize by an integer factor with asymmetric coordinates and
    floor rounding, which is exactly F.interpolate's nearest mode."""
    return F.interpolate(x, scale_factor=scale, mode="nearest", recompute_scale_factor=False)


class PAFPN(nn.Module):
    """Reduce, upsample and merge from the coarsest level down, then downsample and merge
    back up, over three levels c0 (finest) .. c2 (coarsest).

    A reduce step is a convolution before the upsample, or None for none (YOLOv10 has none). `layer` builds the model's own merge layer and downsample from their configs.

    config: reduce [c2 reduce, p1 reduce] (Conv config or None), top_down [2 layer configs],
    down [2 layer configs], bottom_up [2 layer configs], scale (the upsample factor)."""

    def __init__(self, cfg, layer):
        super().__init__()
        self.reduce = nn.ModuleList(nn.Identity() if c is None else Conv(c) for c in cfg["reduce"])
        self.top_down = nn.ModuleList(layer(c) for c in cfg["top_down"])
        self.down = nn.ModuleList(layer(c) for c in cfg["down"])
        self.bottom_up = nn.ModuleList(layer(c) for c in cfg["bottom_up"])
        self.scale = float(cfg["scale"])

    def forward(self, c0, c1, c2):
        r2 = self.reduce[0](c2)
        p1 = self.top_down[0](torch.cat([upsample(r2, self.scale), c1], dim=1))
        r1 = self.reduce[1](p1)
        p0 = self.top_down[1](torch.cat([upsample(r1, self.scale), c0], dim=1))
        n1 = self.bottom_up[0](torch.cat([self.down[0](p0), r1], dim=1))
        n2 = self.bottom_up[1](torch.cat([self.down[1](n1), r2], dim=1))
        return p0, n1, n2
