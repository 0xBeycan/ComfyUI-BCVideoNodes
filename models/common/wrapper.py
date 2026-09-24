# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""The detector and pose models as the preprocess calls them: each wraps a native torch
module loaded from its safetensors file, managed by ComfyUI like any other model, and
decodes the raw output into boxes or keypoints in frame coordinates. The wrappers are
models/{vitpose,yolo}/wrapper.py; this module holds what they share."""
import numpy as np
import torch

from . import checkpoint


def load_models(*models):
    """Bring the models to the compute device together, freeing VRAM held by other models
    if needed; ComfyUI moves them back out when another model needs the room."""
    from comfy import model_management as mm
    mm.load_models_gpu([m.patcher for m in models], force_full_load=True)


class NativeModel:
    """A native module loaded from a model file and managed by ComfyUI like any other model."""

    # which module the file has to hold: its name in the registry's "architecture" family
    architecture = None

    def __init__(self, path):
        from comfy import model_management as mm
        from comfy.model_patcher import ModelPatcher
        self.path = path
        self.net = checkpoint.load(path, self.architecture)
        self.config = self.net.config
        # (height, width) of the model input, and the [N, C, H, W] shape the preprocess
        # reads the crop resolution from
        self.input_size = tuple(self.config["input_size"])
        self.input_shape = [1, 3, *self.input_size]
        # the module runs in its weights' precision; float frames are cast to it
        self.input_dtype = next(self.net.parameters()).dtype
        self.patcher = ModelPatcher(self.net, load_device=mm.get_torch_device(), offload_device=mm.unet_offload_device())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def run(self, x):
        x = torch.from_numpy(np.ascontiguousarray(x)).to(self.patcher.load_device, self.input_dtype)
        with torch.inference_mode():
            out = self.net(x)
        return out.float().cpu().numpy()
