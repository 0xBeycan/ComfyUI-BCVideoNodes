"""RTMW as a plain torch module, built from a config dict.

The export is 456 nodes, far fewer than ViTPose's, but they are all small ops on a 384x288
crop, so the run is bound by python and not by the GPU. Measured on one card: the generic
ONNX executor took 5.3 ms a frame, the kernels it issues take 2.8 ms of that when replayed
from a captured CUDA graph, and this module - 188 modules instead of 456 node dispatches -
takes 4.1 ms. The time is in python frames, not kernels, which is why the weights are held
here and handed to F.conv2d directly instead of being wrapped in an nn.Conv2d inside an
nn.Sequential: the graph has 122 convolutions and every extra module in the chain is three
more python frames per frame of video.

For the same reason the arithmetic stays exactly as the export spells it. SiLU is
`x * sigmoid(x)` and the attention gate is `clamp(alpha * x + beta, 0, 1)`, not F.silu and
F.hardsigmoid: those evaluate `x / (1 + exp(-x))` and `relu6(x + 3) / 6`, which differ in
the last bit, and measured over real crops that difference compounds through 122
convolutions far enough to move a SimCC argmax to the other end of the axis on a keypoint
whose distribution is flat. The fused kernels are not worth a keypoint that jumps.

The structure - a CSPNeXt backbone (stem, four stages of a CSP layer, an SPP bottleneck in
the last), a CSPNeXt-PAFPN neck over three levels, and a SimCC head that ends in a gated
attention unit - is read out of the export once, offline, by scripts/convert_models.py,
with every channel count, kernel, block count, epsilon and folded constant, and stored as
this module's config next to its weights.

config: input_size [h, w], stem [3 Conv configs], stages [4 x {downsample: Conv config,
spp: SPPBottleneck config or None, csp: CSPLayer config}], neck (PAFPN config whose layers
are {"type": "csp", ...} or {"type": "conv", ...}), head (SimCCHead config).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.blocks import PAFPN, Conv, SPPBottleneck, conv_chains, parameter, silu


class Projection(nn.Module):
    """x @ W with W kept as the ONNX MatMul stores it, [in, out]; no transposed copy on load
    and no bias, because the head's projections are exported as bare MatMuls.

    config: [in, out]."""

    def __init__(self, shape):
        super().__init__()
        self.weight = parameter(shape)

    def forward(self, x):
        return x @ self.weight


class ChannelAttention(nn.Module):
    """A per-channel gate from the spatial mean, through the export's own HardSigmoid.

    config: fc (Conv config), alpha, beta."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = Conv(cfg["fc"])
        self.alpha = cfg["alpha"]
        self.beta = cfg["beta"]

    def forward(self, x):
        return x * torch.clamp(self.fc(x.mean((2, 3), keepdim=True)) * self.alpha + self.beta, 0, 1)


class CSPLayer(nn.Module):
    """Half the channels go through the CSPNeXt blocks, half go straight to the concat. The
    blocks are a 3x3 conv and a depthwise-separable 5x5; the residual is not always there:
    the export has it in the first three backbone stages and drops it in the fourth and
    throughout the neck.

    config: short_conv, main_conv, final_conv (Conv configs), blocks [ConvChain configs],
    attention (ChannelAttention config or None)."""

    def __init__(self, cfg):
        super().__init__()
        self.short_conv = Conv(cfg["short_conv"])
        self.main_conv = Conv(cfg["main_conv"])
        self.blocks = conv_chains(cfg["blocks"])
        self.attention = ChannelAttention(cfg["attention"]) if cfg["attention"] is not None else None
        self.final_conv = Conv(cfg["final_conv"])

    def forward(self, x):
        short = self.short_conv(x)
        main = self.main_conv(x)
        for block in self.blocks:
            main = block(main)
        out = torch.cat([main, short], dim=1)
        if self.attention is not None:
            out = self.attention(out)
        return self.final_conv(out)


class Stage(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.downsample = Conv(cfg["downsample"])
        self.spp = SPPBottleneck(cfg["spp"]) if cfg["spp"] is not None else None
        self.csp = CSPLayer(cfg["csp"])

    def forward(self, x):
        x = self.downsample(x)
        if self.spp is not None:
            x = self.spp(x)
        return self.csp(x)


def _neck_layer(cfg):
    if cfg["type"] == "csp":
        return CSPLayer(cfg)
    if cfg["type"] == "conv":
        return Conv(cfg)
    raise ValueError(f"RTMW neck layer type must be csp or conv, found {cfg['type']!r}")


class ScaleNorm(nn.Module):
    """mmpose's ScaleNorm: divide by the L2 norm over the channel axis, times a learned
    scalar. Written the way the export spells it - sum of squares, then sqrt - so the
    arithmetic is the one the graph performs.

    config: scale, gain, eps."""

    def __init__(self, cfg):
        super().__init__()
        self.scale = cfg["scale"]
        self.gain = cfg["gain"]
        self.eps = cfg["eps"]

    def forward(self, x):
        norm = (x * x).sum(-1, keepdim=True).sqrt() * self.scale
        return x / norm.clamp(min=self.eps) * self.gain


class GAU(nn.Module):
    """The gated attention unit the SimCC head ends in. One projection produces the gate,
    the values and a shared 128-wide base; the base is offset two ways into a query and a
    key, and the attention over the 133 keypoint tokens is relu(qk / sqrt(s)) squared, with
    no softmax. The residual is scaled per channel.

    config: norm (ScaleNorm config), uv [in, out], sizes [u, v, base], gamma, beta,
    res_scale (the shapes of those three constants), out [in, out], sqrt_s."""

    def __init__(self, cfg):
        super().__init__()
        self.norm = ScaleNorm(cfg["norm"])
        self.uv = Projection(cfg["uv"])
        self.sizes = list(cfg["sizes"])
        self.gamma = parameter(cfg["gamma"])
        self.beta = parameter(cfg["beta"])
        self.out = Projection(cfg["out"])
        self.res_scale = parameter(cfg["res_scale"])
        self.sqrt_s = cfg["sqrt_s"]

    def forward(self, x):
        uv = self.uv(self.norm(x))
        u, v, base = silu(uv).split(self.sizes, dim=-1)
        qk = base.unsqueeze(-2) * self.gamma + self.beta
        q, k = qk[..., 0, :], qk[..., 1, :]
        attn = torch.relu(q @ k.transpose(-1, -2) / self.sqrt_s)
        return x * self.res_scale + self.out(u * ((attn * attn) @ v))


class SimCCHead(nn.Module):
    """Two 7x7 convolutions down to 133 keypoint maps, one on the coarsest neck level and
    one on that level pixel-shuffled up and concatenated with the next; each map is flattened
    and projected to 128, the two halves are joined, and the gated attention unit produces
    the per-keypoint vector the two SimCC axes are read off.

    config: final_layer, mid_layer, final_layer2 (Conv configs), mlp_norm, mlp2_norm
    (ScaleNorm configs), mlp, mlp2, cls_x, cls_y ([in, out]), gau (GAU config), blocksize."""

    def __init__(self, cfg):
        super().__init__()
        self.final_layer = Conv(cfg["final_layer"])
        self.mlp_norm = ScaleNorm(cfg["mlp_norm"])
        self.mlp = Projection(cfg["mlp"])
        self.mid_layer = Conv(cfg["mid_layer"])
        self.final_layer2 = Conv(cfg["final_layer2"])
        self.mlp2_norm = ScaleNorm(cfg["mlp2_norm"])
        self.mlp2 = Projection(cfg["mlp2"])
        self.gau = GAU(cfg["gau"])
        self.cls_x = Projection(cfg["cls_x"])
        self.cls_y = Projection(cfg["cls_y"])
        self.blocksize = cfg["blocksize"]

    def forward(self, p1, p2):
        a = self.mlp(self.mlp_norm(self.final_layer(p2).flatten(2)))
        b = torch.cat([self.mid_layer(F.pixel_shuffle(p2, self.blocksize)), p1], dim=1)
        b = self.mlp2(self.mlp2_norm(self.final_layer2(b).flatten(2)))
        x = self.gau(torch.cat([a, b], dim=-1))
        return self.cls_x(x), self.cls_y(x)


class RTMWNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.stem = nn.Sequential(*(Conv(c) for c in cfg["stem"]))
        self.stages = nn.ModuleList(Stage(c) for c in cfg["stages"])
        self.neck = PAFPN(cfg["neck"], _neck_layer)
        self.head = SimCCHead(cfg["head"])

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        # the first stage is too fine for the neck; RTMW's PAFPN starts at the second, and
        # its finest output only feeds the bottom-up path, so the head reads the other two
        _, n1, n2 = self.neck(*feats[1:])
        return self.head(n1, n2)
