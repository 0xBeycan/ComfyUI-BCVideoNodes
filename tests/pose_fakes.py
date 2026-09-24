"""Pose fakes shared by the pose and node tests, and the `pose`, `loader` and `wrappers` Names
the test bodies read pack names through (tests/names.py).

FakePose, FakeDetector and frames stand in for the models and the clip. No ComfyUI and no real
model.

ScriptedDetector and RecordingPose script a clip's person boxes and keypoints
(tests/pipelines/test_pose_data.py).
"""
import sys
import types

import numpy as np
import pytest

from names import Names, Ref, Value, refs, seams

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

try:
    import comfy.utils  # noqa: F401
except ImportError:
    # only ProgressBar is used; the real one needs a running server to report to
    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = type("ProgressBar", (), {"__init__": lambda self, total: None,
                                                       "update_absolute": lambda self, value: None})
    comfy.utils = comfy_utils
    sys.modules.setdefault("comfy", comfy)
    sys.modules.setdefault("comfy.utils", comfy_utils)

pose = Names("pose", {
    **refs("pipelines.pose", "KEY_FRAME_BODY_POINTS", "PoseConfig", "asdict", "detect", "draw",
           "key_frame_body_points", "pose_detection"),
    **seams("pipelines.pose", "_to_device"),
})
loader = Names("loader", {
    # the pose model's and the detector's file literals: the node test compares with these, not
    # with the registry the loader reads them from
    "POSE_FILE": Value(lambda: "vitpose_h_wholebody_fp16.safetensors"),
    "DETECTOR_FILE": Value(lambda: "yolov10x_fp32.safetensors"),
    "Yolo": Ref("models.yolo.wrapper", "Yolo"),
    **seams("models.common.loader", "_load", "load_pose_models"),
    **seams("models.common.loader", "_loaded", "detection_model_path"),
})
wrappers = Names("wrappers", {
    **refs("models.vitpose.wrapper", "ViTPose"),
    **refs("models.yolo.wrapper", "Yolo"),
    **refs("models.common.wrapper", "load_models"),
})

B, H, W = 12, 160, 120


class FakePose:
    """A pose model that puts every keypoint at the crop centre plus a fixed offset per
    keypoint, confidence `conf` (per keypoint when an array)."""

    input_shape = [1, 3, 256, 192]

    def __init__(self, conf=0.9):
        self.conf = np.broadcast_to(np.asarray(conf, dtype=np.float32), (133,))
        self.calls = 0

    def __call__(self, img, center, scale):
        assert img.shape == (1, 3, 256, 192)
        self.calls += 1
        kp = np.zeros((1, 133, 3), dtype=np.float32)
        kp[0, :, 0] = center[0][0] + np.linspace(-20, 20, 133)
        kp[0, :, 1] = center[0][1] + np.linspace(-40, 40, 133)
        kp[0, :, 2] = self.conf
        return kp


class FakeDetector:
    """Returns the same person box on every frame, the way the Yolo wrapper hands it over."""

    threshold_conf = 0.05

    def __init__(self):
        self.seen_threshold = None

    def __call__(self, img, shape):
        self.seen_threshold = self.threshold_conf
        return [[{"bbox": np.array([30.0, 20.0, 90.0, 140.0, 0.9]), "track_id": 0, "person_count": 1}]]


@pytest.fixture(autouse=True)
def no_device(monkeypatch):
    monkeypatch.setattr(pose, "_to_device", lambda *models: None)


def frames():
    return torch.rand(B, H, W, 3)


# --- a scripted clip -----------------------------------------------------------------------------

# One detector row per frame of `frames()`, as (x1, y1, x2, y2, score, person_count); None is a
# frame the detector found nobody on.
DETECTOR_ROWS = (
    (30.0, 20.0, 90.0, 128.0, 0.9, 1),
    (31.0, 21.0, 91.0, 128.0, 0.88, 1),
    (32.0, 20.0, 92.0, 129.0, 0.9, 1),
    None,                                  # undetected
    (33.0, 22.0, 93.0, 129.0, 0.9, 1),
    (40.0, 30.0, 46.0, 128.0, 0.8, 1),     # 6 px wide: below MIN_BOX_SIDE
    (35.0, 20.0, 95.0, 130.0, 0.9, 1),
    (5.0, 21.0, 65.0, 128.0, 0.9, 1),      # 5 px from the left edge
    (36.0, 21.0, 96.0, 130.0, 0.9, 1),
    (36.0, 22.0, 96.0, 130.0, 0.9, 2),     # two people
    (37.0, 22.0, 97.0, 131.0, 0.9, 1),
    (38.0, 23.0, 98.0, 131.0, 0.1, 1),     # kept at the default threshold, dropped at 0.2
)


class ScriptedDetector:
    """A person detector that hands over one scripted row per call the way the Yolo wrapper
    does, and records every call: the image, the frame shape and the threshold it ran at. A
    row that is None, or scores below `threshold_conf`, is a frame without a person: the whole
    frame with score -1 and no person count."""

    threshold_conf = 0.05

    def __init__(self, rows=DETECTOR_ROWS):
        self.rows = rows
        self.calls = []

    def __call__(self, img, shape):
        self.calls.append((img.copy(), shape.copy(), self.threshold_conf))
        row = self.rows[len(self.calls) - 1]
        height, width = shape[0]
        if row is None or row[4] < self.threshold_conf:
            return [[{"bbox": np.array([0.0, 0.0, 1.0 * width, 1.0 * height, -1.0]), "track_id": -1}]]
        return [[{"bbox": np.array(row[:5], dtype=np.float64), "track_id": 0, "person_count": row[5]}]]


def _recording_layout():
    """133 keypoints in frame pixels inside the person box of DETECTOR_ROWS, confidence 0.9.
    The nose sits 0.4 px left of the frame, the right ankle 0.12 px inside its right edge and
    the left ankle below it."""
    k = np.arange(133)
    layout = np.stack([32.0 + (k % 14) * 4.0, 22.0 + (k // 14) * 10.0, np.full(133, 0.9)], axis=1)
    layout[0, 0] = -0.4
    layout[16, 0] = W - 0.12
    layout[15, 1] = H + 4.0
    return layout.astype(np.float32)


# (frame, keypoint): (dx, dy, confidence) on that frame. Keypoint 40 is unconfident for two
# frames between confident ones (RECOVERED), 60 jumps on frame 6 (REPLACED), 80 jumps on the
# last frame, with nothing after it (DROPPED); the right hip is unconfident on frame 0.
RECORDING_FAULTS = {
    (4, 40): (0.0, 0.0, 0.1), (5, 40): (0.0, 0.0, 0.1),
    (6, 60): (70.0, 0.0, 0.9),
    (11, 80): (70.0, 0.0, 0.9),
    (0, 12): (0.0, 0.0, 0.3),
}


class RecordingPose:
    """A pose model that records every call's crop, centre and scale and answers with
    `_recording_layout`, moved down `drift` px per call, whatever the crop: its scripted
    `faults` make the temporal layer recover, replace and drop."""

    input_shape = [1, 3, 256, 192]

    def __init__(self, faults=RECORDING_FAULTS, drift=0.25):
        self.layout = _recording_layout()
        self.faults = faults
        self.drift = drift
        self.calls = []

    def keypoints(self, i):
        kp = self.layout.copy()
        kp[:, 1] += np.float32(self.drift * i)
        for (frame, k), (dx, dy, conf) in self.faults.items():
            if frame == i:
                kp[k, 0] += np.float32(dx)
                kp[k, 1] += np.float32(dy)
                kp[k, 2] = conf
        return kp

    def __call__(self, img, center, scale):
        self.calls.append((img.copy(), np.array(center), np.array(scale)))
        return self.keypoints(len(self.calls) - 1)[None]
