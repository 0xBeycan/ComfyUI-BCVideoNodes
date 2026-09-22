"""Which file each model is, and the loader the pose code calls."""
from .. import log
from .download import detection_model_path
from .wrappers import RTMW, ViTPose, Yolo

DETECTOR_FILE = "yolov10x_fp32.safetensors"
# the pose models the Pose Detection node offers: name -> (wrapper, file)
POSE_MODELS = {
    "ViTPose-H": (ViTPose, "vitpose_h_wholebody_fp16.safetensors"),
    "RTMW-l": (RTMW, "rtmw_l_wholebody_384x288_fp32.safetensors"),
}

# built once per process and kept; ComfyUI's model management moves the weights between
# devices, so holding the wrappers costs no VRAM while another model runs
_loaded = {}


def _load(cls, filename):
    if filename not in _loaded:
        with log.step(f"building {filename}"):
            _loaded[filename] = cls(detection_model_path(filename))
    return _loaded[filename]


def load_pose_models(pose_model, detector=True):
    """(detector, pose model) for `pose_model`, one of POSE_MODELS; the files are downloaded
    into models/detection on first use. With `detector` False the detector is neither
    downloaded nor built and None is returned in its place (the person boxes are supplied).
    Call wrappers.load_models(detector, pose) before running them."""
    if pose_model not in POSE_MODELS:
        raise ValueError(f"unknown pose model {pose_model!r}; expected one of {', '.join(POSE_MODELS)}")
    cls, filename = POSE_MODELS[pose_model]
    return (_load(Yolo, DETECTOR_FILE) if detector else None), _load(cls, filename)
