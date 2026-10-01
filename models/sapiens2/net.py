"""Sapiens2 pose as a plain torch module built from a config dict: the ViT backbone and the heatmap head.

Written from facebookresearch/sapiens2 (commit 7e5bae8: sapiens/backbones/standalone/sapiens2.py,
sapiens/pose/src/models/heads/pose_heatmap_head.py) and transformers' Sapiens2ForPoseEstimation
(models/sapiens2/modeling_sapiens2.py, the implementation the HF checkpoints are published for and the
one this module is checked against). Every layer is built from `operations` (a comfy.ops class set:
disable_weight_init / manual_cast for a plain file, MixedPrecisionOps for a .comfy_quant file) the way
ComfyUI core builds its models (comfy/image_encoders/dino3.py), so quantized Linear weights bind.

Backbone, per the config:
- patch embedding (Conv2d, patch 16), then 1 class token and `num_register_tokens` storage tokens in
  front of the patch tokens;
- 2D RoPE on the patch tokens only (prefix tokens are left as they are), computed in float32 and cast to
  the compute dtype. The official `pos_embed_rope_rescale_coords` (HF `pos_embed_rescale`) is a training
  augmentation only: inference uses the plain patch-centre coordinates;
- `num_layers` pre-norm blocks: RMSNorm -> attention -> layer scale (`gamma`) -> residual, RMSNorm ->
  SwiGLU (`w12` = gate | up, silu(gate) * up, `w3`) -> residual. The attention has q_norm / k_norm
  (RMSNorm per head) and grouped-query attention with `kv_heads[i]` key/value heads per layer: "full
  attention" in the source means all heads carry their own key/value (the first and last 8 layers);
  the others share one key/value head between two query heads (num_heads // 2). Nothing is windowed:
  every layer attends over all tokens;
- a final RMSNorm.
Head: the patch tokens as a feature map [B, C, H/16, W/16] -> per upsample stage ConvTranspose2d
(kernel 4, stride 2, padding 1, no bias) -> InstanceNorm2d (no affine) -> SiLU; per conv stage Conv2d
(kernel 1) -> InstanceNorm2d -> SiLU; a 1x1 predictor Conv2d to `num_keypoints` heatmaps at 1/4 of the
input size.

config: input_size [h, w], patch_size, num_channels, hidden_size, num_layers, num_heads, kv_heads (list,
one per layer), intermediate_size, num_register_tokens, rope_theta, rms_norm_eps, qkv_bias [q, k, v],
proj_bias, mlp_bias, num_keypoints, head {upsample_channels, upsample_kernels, conv_channels,
conv_kernels}.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cast(param, x):
    import comfy.ops
    return comfy.ops.cast_to_input(param, x)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _rope(x, cos, sin):
    """RoPE on the patch tokens (the last cos.shape[0] tokens) of x [B, heads, N, head_dim]."""
    prefix = x.shape[-2] - cos.shape[-2]
    head, patches = x[..., :prefix, :], x[..., prefix:, :]
    return torch.cat((head, patches * cos + _rotate_half(patches) * sin), dim=-2)


def rope_tables(num_patches_h, num_patches_w, head_dim, theta, device, dtype):
    """(cos, sin) [h * w, head_dim]: the patch centres in [-1, 1] (y, x), 2 pi * coord * inv_freq, each
    of y and x over head_dim / 4 frequencies, tiled twice (HF Sapiens2RopePositionEmbedding.forward at
    inference)."""
    inv_freq = 1 / theta ** torch.arange(0, 1, 4 / head_dim, dtype=torch.float32, device=device)
    coords_h = torch.arange(0.5, num_patches_h, dtype=torch.float32, device=device) / num_patches_h
    coords_w = torch.arange(0.5, num_patches_w, dtype=torch.float32, device=device) / num_patches_w
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1).flatten(0, 1)
    coords = 2.0 * coords - 1.0
    angles = (2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]).flatten(1, 2).tile(2)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads, kv_heads, eps, qkv_bias, proj_bias, device, dtype, operations):
        super().__init__()
        if dim % heads or heads % kv_heads:
            raise ValueError(f"Sapiens2: width {dim} / {heads} heads / {kv_heads} key-value heads do not divide")
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, dim // heads
        self.wq = operations.Linear(dim, dim, bias=qkv_bias[0], device=device, dtype=dtype)
        self.wk = operations.Linear(dim, kv_heads * self.head_dim, bias=qkv_bias[1], device=device, dtype=dtype)
        self.wv = operations.Linear(dim, kv_heads * self.head_dim, bias=qkv_bias[2], device=device, dtype=dtype)
        self.proj = operations.Linear(dim, dim, bias=proj_bias, device=device, dtype=dtype)
        self.q_norm = operations.RMSNorm(self.head_dim, eps=eps, device=device, dtype=dtype)
        self.k_norm = operations.RMSNorm(self.head_dim, eps=eps, device=device, dtype=dtype)
        self.gamma = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))

    def forward(self, x, cos, sin):
        from comfy.ldm.modules.attention import optimized_attention_for_device
        B, N, C = x.shape
        q = self.q_norm(self.wq(x).view(B, N, self.heads, self.head_dim).transpose(1, 2))
        k = self.k_norm(self.wk(x).view(B, N, self.kv_heads, self.head_dim).transpose(1, 2))
        v = self.wv(x).view(B, N, self.kv_heads, self.head_dim).transpose(1, 2)
        if self.kv_heads != self.heads:
            # query head h reads key/value head h // group (official repeat_interleave, HF repeat_kv)
            group = self.heads // self.kv_heads
            k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        attention = optimized_attention_for_device(q.device, mask=False)
        out = attention(q, k, v, self.heads, None, skip_reshape=True, skip_output_reshape=True,
                        low_precision_attention=False)
        out = self.proj(out.transpose(1, 2).reshape(B, N, C))
        return out * _cast(self.gamma, out)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, bias, device, dtype, operations):
        super().__init__()
        self.w12 = operations.Linear(dim, 2 * hidden, bias=bias, device=device, dtype=dtype)
        self.w3 = operations.Linear(hidden, dim, bias=bias, device=device, dtype=dtype)

    def forward(self, x):
        gate, up = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(gate) * up)


class Block(nn.Module):
    def __init__(self, cfg, kv_heads, device, dtype, operations):
        super().__init__()
        dim, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.ln1 = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.attn = Attention(dim, cfg["num_heads"], kv_heads, eps, cfg["qkv_bias"], cfg["proj_bias"],
                              device, dtype, operations)
        self.ln2 = operations.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
        self.ffn = SwiGLU(dim, cfg["intermediate_size"], cfg["mlp_bias"], device, dtype, operations)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.ffn(self.ln2(x))


class Head(nn.Module):
    def __init__(self, cfg, device, dtype, operations):
        super().__init__()
        head = cfg["head"]
        channels = cfg["hidden_size"]
        self.upsample = nn.ModuleList()
        for out, kernel in zip(head["upsample_channels"], head["upsample_kernels"]):
            self.upsample.append(operations.ConvTranspose2d(channels, out, kernel, stride=2, padding=1, bias=False,
                                                            device=device, dtype=dtype))
            channels = out
        self.conv = nn.ModuleList()
        for out, kernel in zip(head["conv_channels"], head["conv_kernels"]):
            self.conv.append(operations.Conv2d(channels, out, kernel, padding=0, bias=True, device=device, dtype=dtype))
            channels = out
        self.predictor = operations.Conv2d(channels, cfg["num_keypoints"], 1, device=device, dtype=dtype)

    def forward(self, x):
        for layer in (*self.upsample, *self.conv):
            x = F.silu(F.instance_norm(layer(x), eps=1e-5))
        return self.predictor(x)


class Sapiens2PoseNet(nn.Module):
    def __init__(self, cfg, device=None, dtype=None, operations=None):
        super().__init__()
        if operations is None:
            import comfy.ops
            operations = comfy.ops.disable_weight_init
        if len(cfg["kv_heads"]) != cfg["num_layers"]:
            raise ValueError(f"Sapiens2: {len(cfg['kv_heads'])} kv_heads entries for {cfg['num_layers']} layers")
        self.config = cfg
        dim = cfg["hidden_size"]
        self.patch_size = cfg["patch_size"]
        self.head_dim = dim // cfg["num_heads"]
        self.patch_embed = operations.Conv2d(cfg["num_channels"], dim, cfg["patch_size"], stride=cfg["patch_size"],
                                             device=device, dtype=dtype)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim, device=device, dtype=dtype))
        self.storage_tokens = nn.Parameter(torch.empty(1, cfg["num_register_tokens"], dim, device=device, dtype=dtype))
        self.blocks = nn.ModuleList(Block(cfg, kv, device, dtype, operations) for kv in cfg["kv_heads"])
        self.ln_final = operations.RMSNorm(dim, eps=cfg["rms_norm_eps"], device=device, dtype=dtype)
        self.head = Head(cfg, device, dtype, operations)

    def forward(self, x):
        """x [B, 3, H, W] normalised crops -> heatmaps [B, num_keypoints, H / 4, W / 4]."""
        B, _, H, W = x.shape
        h, w = H // self.patch_size, W // self.patch_size
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        prefix = torch.cat((_cast(self.cls_token, tokens).expand(B, -1, -1),
                            _cast(self.storage_tokens, tokens).expand(B, -1, -1)), dim=1)
        tokens = torch.cat((prefix, tokens), dim=1)
        cos, sin = rope_tables(h, w, self.head_dim, self.config["rope_theta"], x.device, tokens.dtype)
        for block in self.blocks:
            tokens = block(tokens, cos, sin)
        tokens = self.ln_final(tokens)[:, prefix.shape[1]:]
        return self.head(tokens.transpose(1, 2).reshape(B, -1, h, w))
