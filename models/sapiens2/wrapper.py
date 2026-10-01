"""Sapiens2 pose as the pipeline calls it: the native module loaded from its model file, managed by
ComfyUI like any other model, run on a batch of person crops and decoded to COCO-WholeBody keypoints.

The model file: one safetensors file holding the module's state dict and the pack's model-file
metadata (format_version, architecture, config, dtype; models/common/checkpoint.py). Two kinds:
- plain (bf16): every tensor is a module tensor;
- quantized (int8_convrot, made from the plain file by convert_to_quant, which keeps the metadata): the
  quantized Linear layers carry an int8 `.weight`, a `.weight_scale` and a `.comfy_quant` JSON tensor
  ({"format": "int8_tensorwise", "convrot": true, "orig_dtype": ...}); every other tensor is plain.
checkpoint.load builds plain `nn` layers and wants one floating precision, so this wrapper builds the
module itself (load_net): the layers from comfy.ops the way core loads a detection model of either kind
(comfy_extras/nodes_sam3d_body.py SAM3DBody_Loader): a quantized file with the operations core picks for
a quantized diffusion model (comfy.ops.pick_operations with the file's quant config ->
mixed_precision_ops), computing in the file's orig_dtype; a plain file with the operations core picks
for the weight dtype (unet_dtype / unet_manual_cast / pick_operations).
"""
import json

import numpy as np
import torch

from ..common import checkpoint
from ..common.wrapper import NativeModel
from .decode import udp_decode
from .keypoints import SAPIENS2_70_NAMES, to_coco133
from .net import Sapiens2PoseNet

ARCHITECTURE = "sapiens2_pose"
# the script that writes the plain file, named by a refused file's error
CONVERTER = "scripts/convert_sapiens2.py"
# Sapiens2 runs in bf16 (its training precision) or fp32; fp16 is not offered
SUPPORTED_DTYPES = [torch.bfloat16, torch.float32]


def quant_compute_dtype(state):
    """The compute dtype of a quantized state dict (the first .comfy_quant's orig_dtype), None when the
    state dict carries no .comfy_quant tensor."""
    for key, value in state.items():
        if key.endswith(".comfy_quant"):
            conf = json.loads(value.numpy().tobytes())
            name = str(conf.get("orig_dtype", "bfloat16")).replace("torch.", "")
            dtype = getattr(torch, name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"{key}: orig_dtype {conf.get('orig_dtype')!r} is not a torch dtype")
            return dtype
    return None


def operations_for(state, load_device, dtype=None):
    """(operations, build dtype, compute dtype, quantized) for a state dict, as core picks them.
    `dtype` forces the precision of a plain file (float32 for a parity check)."""
    import types

    import comfy.model_management as mm
    import comfy.ops
    import comfy.utils

    quant_config = comfy.utils.detect_layer_quantization(state, "")
    if quant_config is not None:
        if dtype is not None:
            raise ValueError("a quantized model file computes in its own orig_dtype; do not force a dtype")
        compute = quant_compute_dtype(state)
        ops = comfy.ops.pick_operations(None, compute, load_device=load_device,
                                        model_config=types.SimpleNamespace(quant_config=quant_config))
        return ops, compute, compute, True
    if dtype is not None:
        return comfy.ops.pick_operations(dtype, None, load_device=load_device, disable_fast_fp8=True), dtype, dtype, False
    weight_dtype = comfy.utils.weight_dtype(state)
    storage = mm.unet_dtype(device=load_device, model_params=-1, supported_dtypes=SUPPORTED_DTYPES,
                            weight_dtype=weight_dtype)
    manual_cast = mm.unet_manual_cast(storage, load_device, supported_dtypes=SUPPORTED_DTYPES)
    ops = comfy.ops.pick_operations(storage, manual_cast, load_device=load_device, disable_fast_fp8=True)
    return ops, storage, manual_cast or storage, False


def build(state, config, load_device, dtype=None, source="state dict"):
    """(net, compute dtype, quantized): the module built from `config` with the operations for `state`,
    `state` loaded into it. Missing or unexpected tensors are refused, every one listed."""
    ops, build_dtype, compute, quantized = operations_for(state, load_device, dtype)
    net = Sapiens2PoseNet(config, dtype=build_dtype, operations=ops).eval()
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(f"{source} does not match the Sapiens2 pose module its config describes: "
                         f"missing {sorted(missing)[:8]}{' ...' if len(missing) > 8 else ''}, "
                         f"unexpected {sorted(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}")
    return net, compute, quantized


class Sapiens2Pose(NativeModel):
    """The model of one file: `self(crops, centers, scales)` -> [N, 133, 3] keypoints."""

    architecture = ARCHITECTURE

    def __init__(self, path, dtype=None):
        # forces a plain file's precision (float32 for a parity check); None is core's choice
        self.dtype = dtype
        super().__init__(path)

    def load_net(self, path):
        """(the module in `path`, its compute dtype): the input is cast to the compute dtype, since a
        quantized module computes in its orig_dtype, not in its int8 weights' dtype."""
        import comfy.model_management as mm
        import comfy.utils

        state, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        config = checkpoint.model_config(metadata or {}, path, self.architecture, CONVERTER)
        net, compute, self.quantized = build(state, config, mm.get_torch_device(), self.dtype, source=path)
        return net, compute

    @property
    def precision(self):
        return f"{'int8_convrot weights, ' if self.quantized else ''}{str(self.input_dtype).replace('torch.', '')} compute"

    def forward(self, crops, centers, scales):
        """`crops` [N, 3, 1024, 768] float32 (decode.crop_input), `centers` and `scales` [N, 2] of the
        crops -> [N, 133, 3] COCO-WholeBody x and y in frame pixels and the confidence (the raw heatmap
        peak); the face rows 23-90 are 0 (keypoints.to_coco133). Only the heatmaps of keypoints 0-69, the
        ones the table reads, are decoded: the decode treats every keypoint on its own."""
        heatmaps = self.run(crops)[:, :len(SAPIENS2_70_NAMES)]
        return to_coco133(np.stack([udp_decode(h, c, s) for h, c, s in zip(heatmaps, centers, scales)]))
