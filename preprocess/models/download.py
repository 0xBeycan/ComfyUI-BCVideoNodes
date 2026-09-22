"""Model download used by the pose loader and the SAM3 segmentation on first use."""
import os
import urllib.request

import folder_paths
from comfy.utils import ProgressBar
from tqdm import tqdm

from .. import log

# Every detection and pose model the pack runs is fetched from here, as the safetensors
# file scripts/convert_models.py writes.
MODEL_REPO = "beycanai/BCVideoNodes-models"
MODEL_REPO_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/main/"

# The models live in ComfyUI's own models/detection folder.
DETECTION_FOLDER = "detection"
_detection_path = os.path.join(folder_paths.models_dir, DETECTION_FOLDER)
folder_paths.add_model_folder_path(DETECTION_FOLDER, _detection_path)


def download(url, path):
    """Stream `url` to `path` through a .part file, so an interrupted download never leaves
    a truncated model behind."""
    name = os.path.basename(path)
    log.info(f"downloading {name} from {url}")
    part = path + ".part"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-BCVideoNodes"})
        with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            comfy_pbar = ProgressBar(total) if total else None
            with tqdm(total=total or None, unit="B", unit_scale=True, desc=name) as pbar:
                for chunk in iter(lambda: response.read(1 << 20), b""):
                    out.write(chunk)
                    pbar.update(len(chunk))
                    if comfy_pbar is not None:
                        comfy_pbar.update_absolute(pbar.n)
    except Exception as e:
        if os.path.exists(part):
            os.remove(part)
        raise RuntimeError(f"Could not download {name} from {url} ({e}). Download it by hand to {path}") from e
    os.replace(part, path)


def detection_model_path(filename):
    """Full path of `filename` in models/detection, downloaded from the model repository
    first when it is not there."""
    found = folder_paths.get_full_path(DETECTION_FOLDER, filename)
    if found:
        return found
    os.makedirs(_detection_path, exist_ok=True)
    path = os.path.join(_detection_path, filename)
    download(MODEL_REPO_URL + filename, path)
    return path
