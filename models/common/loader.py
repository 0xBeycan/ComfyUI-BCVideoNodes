"""The loader the pose code calls: the person detector and the pose models, built once per process
from the files their model packages register."""
from ...libs import log
from . import registry
from .download import detection_model_path

# the registry name of the person detector: models/yolo registers it with its wrapper and file
DETECTOR = "YOLOv10x"

# built once per process and kept; ComfyUI's model management moves the weights between
# devices, so holding the wrappers costs no VRAM while another model runs
_loaded = {}


def _load(cls, filename):
    if filename not in _loaded:
        with log.step(f"building {filename}"):
            _loaded[filename] = cls(detection_model_path(filename))
    return _loaded[filename]


def pose_model_names():
    """The pose models the Pose Detection node offers, in registration order (its combo list),
    as a new list."""
    return registry.names("pose_estimator")


def load_pose_models(pose_model, detector=True):
    """(detector, pose model) for `pose_model`, one of pose_model_names(); the files are downloaded
    into models/detection on first use. With `detector` False the detector is neither
    downloaded nor built and None is returned in its place (the person boxes are supplied).
    Call wrapper.load_models(detector, pose) before running them."""
    names = pose_model_names()
    if pose_model not in names:
        raise ValueError(f"unknown pose model {pose_model!r}; expected one of {', '.join(names)}")
    cls, filename = registry.get("pose_estimator", pose_model)
    person_detector = registry.get("person_detector", DETECTOR)
    return (_load(person_detector.implementation, person_detector.file) if detector else None), _load(cls, filename)
