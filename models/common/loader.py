"""The loader the pose code calls: the person detector and the pose model, built once per process
from the files their model packages register."""
from ...libs import log
from . import registry
from .download import detection_model_path

# the registry names of the person detector and the pose estimator: models/yolo and
# models/vitpose register them with their wrappers and files
DETECTOR = "YOLOv10x"
POSE_ESTIMATOR = "ViTPose-H"

# built once per process and kept; ComfyUI's model management moves the weights between
# devices, so holding the wrappers costs no VRAM while another model runs
_loaded = {}


def _load(cls, filename, repo=None):
    if filename not in _loaded:
        with log.step(f"building {filename}"):
            _loaded[filename] = cls(detection_model_path(filename, repo))
    return _loaded[filename]


def load_pose_models(detector=True):
    """(detector, pose model); the files are downloaded into models/detection on first use. With
    `detector` False the detector is neither downloaded nor built and None is returned in its
    place (the person boxes are supplied). Call wrapper.load_models(detector, pose) before
    running them."""
    pose_estimator = registry.get("pose_estimator", POSE_ESTIMATOR)
    person_detector = registry.get("person_detector", DETECTOR)
    return ((_load(person_detector.implementation, person_detector.file) if detector else None),
            _load(pose_estimator.implementation, pose_estimator.file))


def load_pose_estimator(name):
    """The pose estimator registered as `name` (a Sapiens2 pose_model: "Sapiens2 <size> <precision>"),
    its file downloaded into models/detection on first use. Call wrapper.load_models on it before
    running it."""
    if name not in registry.names("pose_estimator"):
        raise ValueError(f"unknown pose model {name!r}; expected one of {', '.join(registry.names('pose_estimator'))}")
    entry = registry.get("pose_estimator", name)
    return _load(entry.implementation, entry.file, entry.repo)
