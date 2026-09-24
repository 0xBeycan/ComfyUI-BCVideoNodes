"""The contracts of the person_detector and pose_estimator families: what the pose pipeline reads
from and calls on the wrapper objects the loader builds. Documentation of the duck-typed
contract only: nothing checks against these classes and no code path reads them.

Registering a new person detector or pose estimator means a wrapper with these attributes and
this call, registered in its package's __init__.py (see common/registry.py).
"""
from typing import Optional, Protocol

import numpy as np


class PersonDetector(Protocol):
    patcher: "ModelPatcher"          # comfy.model_patcher.ModelPatcher, read by wrapper.load_models
    threshold_conf: float            # the score threshold, overridden for the run by the pose pipeline

    def __call__(self, img: np.ndarray, shape_raw: np.ndarray) -> list[list[dict]]:
        """`img` float32 [1, 3, h, w], RGB 0..1, already resized by the pose pipeline to its
        DETECTOR_INPUT_SIZE (the Yolo wrapper refuses any other size); `shape_raw` int [[H, W]] of
        the frame. Returns [[{"bbox": float64 (5,) x1, y1, x2, y2, score, "track_id": int,
        "person_count": int (absent when nobody was found)}]] in frame coordinates."""
        ...


class PoseEstimator(Protocol):
    patcher: "ModelPatcher"          # comfy.model_patcher.ModelPatcher, read by wrapper.load_models
    input_shape: list[int]           # [1, 3, h, w]: the resolution the crop is sampled at
    conf_scale: Optional[float]      # the divisor of a raw score; None: the confidences are not one

    def __call__(self, img: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """`img` float32 [1, 3, h, w] (the normalised crop), `center` [1, 2], `scale` [1, 2] of the
        crop. Returns [1, 133, 3]: x and y in frame pixels and the confidence, per keypoint."""
        ...
