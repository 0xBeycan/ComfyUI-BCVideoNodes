"""Running a ComfyUI core node outside the graph: node_class, with_schema_defaults, call_node; and
clip_vision_encode_official, core's CLIP vision model on the official Wan preprocessing."""

import inspect

import torch


def node_class(node_id):
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if cls is None:
        raise RuntimeError("Core node '{}' is not registered. Update ComfyUI.".format(node_id))
    return cls


def with_schema_defaults(cls, kwargs):
    # The graph executor fills widget defaults before calling a node; calling
    # the class directly we must do the same, or a core update that adds a
    # widget turns into a TypeError here.
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, definition in (spec.get(section) or {}).items():
            if name in kwargs or not isinstance(definition, (list, tuple)) or len(definition) < 2:
                continue
            options = definition[1]
            if isinstance(options, dict) and "default" in options:
                kwargs[name] = options["default"]
    return kwargs


def call_node(node_id, **kwargs):
    """Run a core node outside the graph and return its outputs as a tuple.

    V3 nodes (io.ComfyNode) expose FUNCTION as a classmethod and return a
    NodeOutput whose values live in .args; V1 nodes name an instance method
    that returns a tuple, or a dict with a "result" key.
    """
    cls = node_class(node_id)
    fn = getattr(cls, cls.FUNCTION)
    if not inspect.ismethod(fn):
        fn = getattr(cls(), cls.FUNCTION)
    result = fn(**with_schema_defaults(cls, dict(kwargs)))
    if hasattr(result, "args"):
        return tuple(result.args)
    if isinstance(result, dict):
        return tuple(result["result"])
    return tuple(result)


def clip_vision_encode_official(clip_vision, image):
    """``image`` [B, H, W, C] in [0, 1] through core's CLIP vision model as the Wan team's
    CLIPModel.visual preprocesses it (zai-org/SCAIL-2 wan/modules/clip.py, wan-scail2 branch; the
    same in Wan2.2 wan/modules/animate/clip.py and Wan-Animate-2 wanxiang/eval_i2v.py):
    stretched to CLIP's square with bicubic interpolation, align_corners False and no antialias,
    neither clamped nor rounded to 8 bit, then normalized with the model's mean and std. Core's
    CLIPVisionEncode antialiases, clamps and rounds to 8 bit (comfy/clip_model.py clip_preprocess).
    Returns the outputs CLIPVisionEncode gives (comfy/clip_vision.py encode_image); their
    penultimate_hidden_states, which core's Wan models read as clip_fea, are the output of all but
    the last transformer block, the official use_31_block of the 32-block ViT-H."""
    import comfy.clip_vision
    import comfy.model_management

    if getattr(clip_vision, "model_type", None) != "clip_vision_model":
        raise ValueError("clip_vision is a {} model; this sampler needs the CLIP ViT-H vision model of Wan "
                         "(clip_vision_h.safetensors).".format(getattr(clip_vision, "model_type", type(clip_vision).__name__)))
    size = clip_vision.image_size
    pixels = image[..., :3].movedim(-1, 1).to(clip_vision.load_device, torch.float32)
    pixels = torch.nn.functional.interpolate(pixels, size=(size, size), mode="bicubic", align_corners=False)
    mean = torch.tensor(clip_vision.image_mean, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    std = torch.tensor(clip_vision.image_std, device=pixels.device, dtype=pixels.dtype).view(1, 3, 1, 1)
    pixels = (pixels - mean) / std

    comfy.model_management.load_model_gpu(clip_vision.patcher)
    out = clip_vision.model(pixel_values=pixels, intermediate_output=-2)
    device = comfy.model_management.intermediate_device()
    outputs = comfy.clip_vision.Output()
    outputs["last_hidden_state"] = out[0].to(device)
    outputs["image_embeds"] = out[2].to(device)
    outputs["image_sizes"] = [pixels.shape[1:]] * pixels.shape[0]
    outputs["penultimate_hidden_states"] = out[1].to(device)
    outputs["mm_projected"] = out[3]
    return outputs
