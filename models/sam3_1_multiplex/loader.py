"""The SAM 3.1 Multiplex checkpoint as a ComfyUI model and the CLIP that encodes the prompt, loaded
once and kept, and downloaded into models/checkpoints on first use."""
import os

from ...libs import log
from ..common.download import download
from .adapter import DEFAULT_SAM3_1_MULTIPLEX


DEFAULT_SAM3_1_MULTIPLEX_URL = "https://huggingface.co/Comfy-Org/sam3.1/resolve/main/checkpoints/sam3.1_multiplex_fp16.safetensors"


_loaded = {"name": None, "model": None, "clip": None}


def load_sam3_1_multiplex(name=DEFAULT_SAM3_1_MULTIPLEX):
    """(model, clip): the SAM3 checkpoint `name` as a ComfyUI model and the CLIP that encodes
    the prompt (None when the checkpoint carries no text encoder), loaded once and kept; the
    default checkpoint is downloaded into models/checkpoints when it is missing."""
    import comfy.sd
    import folder_paths
    if _loaded["name"] == name:
        return _loaded["model"], _loaded["clip"]
    path = folder_paths.get_full_path("checkpoints", name)
    if path is None and name == DEFAULT_SAM3_1_MULTIPLEX:
        path = os.path.join(folder_paths.get_folder_paths("checkpoints")[0], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        download(DEFAULT_SAM3_1_MULTIPLEX_URL, path)
    if path is None:
        raise FileNotFoundError(f"SAM3 checkpoint {name} is not in models/checkpoints")
    _loaded["name"], _loaded["model"], _loaded["clip"] = None, None, None
    with log.step(f"loading {name}"):
        model, clip = comfy.sd.load_checkpoint_guess_config(path, output_vae=False, output_clip=True)[:2]
    _loaded["name"], _loaded["model"], _loaded["clip"] = name, model, clip
    return model, clip
