"""The pose module on synthetic frames with fake models: the supplied-box path that skips the
detector, the key_frame_body_points string, the config. No ComfyUI and no real model:

    python -m pytest tests/test_pose.py
"""
import json
import sys
import types

import numpy as np
import pytest

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

from preprocess import pose  # noqa: E402

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


class NoDetector:
    def __call__(self, *args, **kwargs):
        raise AssertionError("the detector must not run when boxes are supplied")


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


def test_supplied_boxes_skip_the_detector():
    boxes = [(30.0 + i, 20.0, 90.0 + i, 140.0) for i in range(B)]
    model = FakePose()
    images, pose_data, out_boxes, _ = pose.pose_detection(frames(), NoDetector(), model, bboxes=boxes)
    assert model.calls == B
    assert images.shape == (B, H, W, 3)
    assert len(out_boxes) == B and len(pose_data["detections"]) == B
    assert all(d["score"] == 1.0 and d["persons"] == 1 for d in pose_data["detections"])


def test_supplied_boxes_need_no_detector_object():
    pose_data, _ = pose.detect(None, FakePose(), frames(), bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert len(pose_data["detections"]) == B


def test_a_single_supplied_box_is_used_on_every_frame():
    config = pose.PoseConfig(temporal=False, box_window=0)
    _, boxes = pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(30.0, 20.0, 90.0, 140.0, 0.5)], config=config)
    assert boxes == [(30.0, 20.0, 90.0, 140.0)] * B


def test_supplied_boxes_go_through_the_same_box_logic_as_detections():
    """Supplying the detector's own boxes gives what the detector gives."""
    detected, detected_boxes = pose.detect(FakeDetector(), FakePose(), frames())
    supplied, supplied_boxes = pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert supplied_boxes == detected_boxes


def test_the_wrong_number_of_supplied_boxes_raises():
    with pytest.raises(ValueError, match="3 boxes for 12 frames"):
        pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(0, 0, 10, 10)] * 3)


def test_an_inverted_supplied_box_raises():
    with pytest.raises(ValueError, match="x1 < x2"):
        pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(90, 20, 30, 140)])


def test_the_detection_threshold_reaches_the_detector_and_is_restored():
    detector = FakeDetector()
    pose.detect(detector, FakePose(), frames(), config=pose.PoseConfig(detection_threshold=0.2))
    assert detector.seen_threshold == 0.2 and detector.threshold_conf == 0.05


def test_a_confidence_scale_on_a_model_without_one_raises():
    with pytest.raises(ValueError, match="conf_scale"):
        pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(30, 20, 90, 140)],
                    config=pose.PoseConfig(confidence_scale=4.6))


def test_pose_data_keys():
    pose_data, _ = pose.detect(FakeDetector(), FakePose(), frames())
    assert set(pose_data) == {"pose_metas", "pose_metas_original", "detections", "keypoint_source", "pose_config"}
    assert pose_data["pose_config"] == pose.asdict(pose.PoseConfig())
    assert np.array(pose_data["keypoint_source"]).shape == (B, 133)


def test_temporal_off_marks_every_keypoint_measured():
    conf = np.full(133, 0.9)
    conf[7] = 0.1
    pose_data, _ = pose.detect(FakeDetector(), FakePose(conf), frames(), config=pose.PoseConfig(temporal=False))
    assert not np.any(pose_data["keypoint_source"])


def test_config_values_out_of_range_raise():
    with pytest.raises(ValueError, match="detection_threshold"):
        pose.PoseConfig(detection_threshold=1.5)
    with pytest.raises(ValueError, match="box_window"):
        pose.PoseConfig(box_window=-1)


def test_key_frame_body_points_is_the_points_editor_string():
    pose_data, _ = pose.detect(FakeDetector(), FakePose(), frames())
    text = pose.key_frame_body_points(pose_data, 0.5)
    points = json.loads(text)
    assert isinstance(points, list) and len(points) == len(pose.KEY_FRAME_BODY_POINTS)
    assert all(set(p) == {"x", "y"} and type(p["x"]) is int and type(p["y"]) is int for p in points)
    # frame 0's body keypoints in frame pixels, in the exported order
    meta = pose_data["pose_metas_original"][0]
    body = meta["keypoints_body"][list(pose.KEY_FRAME_BODY_POINTS)]
    assert points == [{"x": int(x * W), "y": int(y * H)} for x, y in body[:, :2]]
    # what easy-sam3 accepts: pixel coordinates inside the frame
    assert all(0 <= p["x"] < W and 0 <= p["y"] < H for p in points)


def test_key_frame_body_points_keeps_only_confident_points():
    conf = np.full(133, 0.9, dtype=np.float32)
    conf[[5, 6]] = 0.2  # the shoulders; the neck is their mean, so it drops too
    pose_data, _ = pose.detect(FakeDetector(), FakePose(conf), frames(), config=pose.PoseConfig(temporal=False))
    points = json.loads(pose.key_frame_body_points(pose_data, 0.5))
    body = pose_data["pose_metas_original"][0]["keypoints_body"]
    kept = [i for i in pose.KEY_FRAME_BODY_POINTS if body[i, 2] >= 0.5]
    assert len(points) == len(kept) == len(pose.KEY_FRAME_BODY_POINTS) - 3


def test_key_frame_body_points_is_an_empty_list_without_a_confident_point():
    pose_data, _ = pose.detect(FakeDetector(), FakePose(0.1), frames(), config=pose.PoseConfig(temporal=False))
    assert pose.key_frame_body_points(pose_data, 0.5) == "[]"


def test_draw_threshold_decides_what_is_drawn():
    pose_data, _ = pose.detect(FakeDetector(), FakePose(0.6), frames(), config=pose.PoseConfig(temporal=False))
    assert pose.draw(pose_data, draw_threshold=0.5).sum() > 0
    assert pose.draw(pose_data, draw_threshold=0.7, draw_head=False).sum() == 0


# --- precedence: config fields a run does not read ------------------------------------------

def not_used_lines(caplog):
    return [r.getMessage() for r in caplog.records if "not used" in r.getMessage()]


def test_supplied_boxes_ignore_the_detection_threshold_in_one_line(caplog):
    box = [(30.0, 20.0, 90.0, 140.0)]
    plain, _ = pose.detect(NoDetector(), FakePose(), frames(), bboxes=box)
    with caplog.at_level("INFO"):
        changed, _ = pose.detect(NoDetector(), FakePose(), frames(), bboxes=box,
                                 config=pose.PoseConfig(detection_threshold=0.5))
    assert not_used_lines(caplog) == ["[BCVideoNodes] pose_config.detection_threshold "
                                      "(bboxes connected, the detector does not run) not used"]
    assert plain["detections"] == changed["detections"]


def test_temporal_off_ignores_the_temporal_fields_in_one_line(caplog):
    with caplog.at_level("INFO"):
        pose.detect(FakeDetector(), FakePose(), frames(),
                    config=pose.PoseConfig(temporal=False, temporal_max_gap=3, box_window=2))
    (line,) = not_used_lines(caplog)
    assert "pose_config.temporal_max_gap (temporal off)" in line and "box_window" not in line


def test_the_default_config_ignores_nothing(caplog):
    with caplog.at_level("INFO"):
        pose.detect(NoDetector(), FakePose(), frames(), bboxes=[(30.0, 20.0, 90.0, 140.0)])
        pose.detect(FakeDetector(), FakePose(), frames(), config=pose.PoseConfig(detection_threshold=0.2))
    assert not_used_lines(caplog) == []
