"""Model download used by the pose loader and the SAM3 segmentation on first use."""
import os
import time
import urllib.error
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


# A cut connection is resumed from the bytes already on disk this many times before giving up
# (the Mac's connection to the HF CDN dropped the 1.27 GB ViTPose every 100-200 MB).
ATTEMPTS = 5
RETRY_WAIT = 2.0  # seconds between attempts


def _fetch(url, part, name):
    """One attempt: continue `part` from its size with a Range request, or start it over when
    the server answers with the whole file. Raises when the connection closes early."""
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "ComfyUI-BCVideoNodes"}
    if have:
        headers["Range"] = f"bytes={have}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        length = int(response.headers.get("Content-Length") or 0)
        if have and response.status == 206:
            # "bytes <from>-<to>/<size>"; the size is the whole file's
            size = response.headers.get("Content-Range", "").rpartition("/")[2]
            total = int(size) if size.isdigit() else (have + length if length else 0)
        else:
            # the server ignored the range (or there was nothing to resume): start over
            have, total = 0, length
        comfy_pbar = ProgressBar(total) if total else None
        with open(part, "ab" if have else "wb") as out, \
                tqdm(total=total or None, initial=have, unit="B", unit_scale=True, desc=name) as pbar:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
                pbar.update(len(chunk))
                if comfy_pbar is not None:
                    comfy_pbar.update_absolute(pbar.n)
            # read() returns b"" when the server closes the connection early, so the loop
            # above ends normally on a cut download; only the length tells
            if total and pbar.n != total:
                raise IOError(f"the connection closed after {pbar.n} of {total} bytes")


def download(url, path):
    """Stream `url` to `path` through a .part file, so an interrupted download never leaves
    a truncated model behind. A cut connection is resumed where it stopped, ATTEMPTS times;
    after that the error says where to put the file by hand, and the .part stays for the
    next run to continue from."""
    name = os.path.basename(path)
    part = path + ".part"
    log.info(f"downloading {name} from {url}" + (" (resuming)" if os.path.exists(part) else ""))
    error = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            _fetch(url, part, name)
            error = None
            break
        except urllib.error.HTTPError as e:
            error = e
            if e.code == 416 and os.path.exists(part):
                # the range starts past the end: the .part is not a prefix of this file
                os.remove(part)
            elif 400 <= e.code < 500 and e.code not in (408, 416, 429):
                break  # a wrong URL or a refused request does not get better by retrying
        except Exception as e:
            error = e
        if attempt < ATTEMPTS:
            log.info(f"{name}: attempt {attempt} of {ATTEMPTS} stopped ({error}); retrying from where it stopped")
            time.sleep(RETRY_WAIT)
    if error is not None:
        kept = os.path.getsize(part) if os.path.exists(part) else 0
        raise RuntimeError(
            f"Could not download {name} ({error}). Download it by hand from {url} and put it in "
            f"{os.path.dirname(path)} as {name}."
            + (f" The {kept} bytes fetched so far are kept in {part}; running again continues from there." if kept else "")
        ) from error
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
