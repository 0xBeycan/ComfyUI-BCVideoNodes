"""G4, G5 and G14: the pose pipeline, recorded from the production code (specs refactor plan
8.2) on a seeded 12-frame clip, scripted models and hand-built metas.

G4  `pose.detect`: the detector's and the pose model's every call, the models brought to the
    device, pose_metas_original with its dtypes, detections, keypoint_source (RECOVERED,
    REPLACED and DROPPED all fire), pose_config, the returned boxes, the log lines and the
    ProgressBar, in nine scenarios.
G5  `pose.draw`: the pose image pixels at 160x120 and 720x1280 (height x width) under each
    drawing switch.
G14 `key_frame_body_points`: the exact strings, the truncation cases kept and the points
    outside the frame left out.

The crops and the drawing go through cv2, so these digests hold for the cv2 build of the
ComfyUI venv only; they are in tests/goldens/test_pose_golden.json:

    python -m pytest tests/pipelines/test_pose_golden.py
"""
import json
import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from golden import check, digest, log_text  # noqa: E402
from pose_fakes import B, H, W, RecordingPose, ScriptedDetector, no_device, pose  # noqa: E402,F401

from bcvideonodes.libs import temporal  # noqa: E402
from bcvideonodes.libs.pose_utils.pose2d_utils import AAPoseMeta  # noqa: E402


def seeded_frames():
    return torch.from_numpy(np.random.default_rng(0).random((B, H, W, 3), dtype=np.float32))


def recording_progress_bar(events):
    class RecordingProgressBar:
        def __init__(self, total):
            events.append(("total", total))

        def update_absolute(self, value, total=None, preview=None):
            events.append(("update", value))

    return RecordingProgressBar


def capture_logs(caplog):
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="BCVideoNodes")
    caplog.clear()


class RTMWLike(RecordingPose):
    """RTMW's contract: a `conf_scale` divisor of a raw score, here the scripted confidence
    times 4.6, and the divisor each call ran with recorded."""

    architecture = "rtmw"
    conf_scale = 6.0

    def __call__(self, img, center, scale):
        kp = super().__call__(img, center, scale)
        self.calls[-1] += (self.conf_scale,)
        kp[..., 2] = np.clip(kp[..., 2] * np.float32(4.6) / np.float32(self.conf_scale), 0.0, 1.0)
        return kp


class ViTPoseLike(RecordingPose):
    """ViTPose's contract: confidences used as they are, `conf_scale` None."""

    architecture = "vitpose"
    conf_scale = None


BOXES = [(30.0 + i, 20.0, 90.0 + i, 128.0) for i in range(B)]
SCENARIOS = {
    "default": (RecordingPose, {}),
    "temporal_off": (RecordingPose, {"config": {"temporal": False}}),
    "box_window_0": (RecordingPose, {"config": {"box_window": 0}}),
    "supplied_list": (RecordingPose, {"bboxes": BOXES}),
    "supplied_single": (RecordingPose, {"bboxes": BOXES[0]}),
    "supplied_json": (RecordingPose, {"bboxes": json.dumps([{"startX": x1, "startY": y1, "endX": x2, "endY": y2}
                                                             for x1, y1, x2, y2 in BOXES])}),
    "detection_threshold_0.2": (RecordingPose, {"config": {"detection_threshold": 0.2}}),
    "confidence_scale_rtmw": (RTMWLike, {"config": {"confidence_scale": 4.6}}),
    "confidence_scale_vitpose": (ViTPoseLike, {"config": {"confidence_scale": 4.6}}),
}


def run_detect(name, caplog, monkeypatch):
    """pose.detect on the scenario, with the device calls and the ProgressBar recorded."""
    model_class, kwargs = SCENARIOS[name]
    kwargs = dict(kwargs)
    if "config" in kwargs:
        kwargs["config"] = pose.PoseConfig(**kwargs["config"])
    detector, model = ScriptedDetector(), model_class()
    devices, events = [], []
    monkeypatch.setattr(pose, "_to_device", lambda *models: devices.append(tuple(type(m).__name__ for m in models)))
    monkeypatch.setattr(pose, "ProgressBar", recording_progress_bar(events))
    capture_logs(caplog)
    pose_data, boxes = pose.detect(detector, model, seeded_frames(), **kwargs)
    return detector, model, pose_data, boxes, devices, events, log_text(caplog.records)


def metas_digest(metas):
    return digest([[(key, digest(value) if isinstance(value, np.ndarray) else repr(value))
                    for key, value in meta.items()] for meta in metas])


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_detect_golden(name, caplog, monkeypatch):
    detector, model, pose_data, boxes, devices, events, log = run_detect(name, caplog, monkeypatch)
    source = np.array(pose_data["keypoint_source"])
    if name == "default":
        # the clip reaches what G4 pins: every temporal label, an undetected frame, a box below
        # MIN_BOX_SIDE, the edge snap and a second person
        assert {temporal.RECOVERED, temporal.REPLACED, temporal.DROPPED} <= set(np.unique(source).tolist())
        assert [d["score"] for d in pose_data["detections"]][3:6:2] == [-1.0, -1.0]
        assert boxes[7][0] == 0.0 and pose_data["detections"][9]["persons"] == 2
    if "bboxes" in SCENARIOS[name][1]:
        assert detector.calls == []
    assert detector.threshold_conf == ScriptedDetector.threshold_conf
    if name == "confidence_scale_rtmw":
        assert {call[3] for call in model.calls} == {4.6} and model.conf_scale == RTMWLike.conf_scale
    calls = [(digest(img), digest(center), digest(scale), *rest) for img, center, scale, *rest in model.calls]
    check(__file__, f"{name}.detector_calls",
          digest([(digest(img), repr(shape.tolist()), threshold) for img, shape, threshold in detector.calls]))
    check(__file__, f"{name}.estimator_calls", digest(calls))
    check(__file__, f"{name}.to_device", repr(devices))
    check(__file__, f"{name}.pose_metas_original", metas_digest(pose_data["pose_metas_original"]))
    check(__file__, f"{name}.detections", digest(pose_data["detections"]))
    check(__file__, f"{name}.keypoint_source", digest(pose_data["keypoint_source"]))
    check(__file__, f"{name}.pose_config", digest(pose_data["pose_config"]))
    check(__file__, f"{name}.keys", repr(list(pose_data)))
    check(__file__, f"{name}.boxes", repr(boxes))
    check(__file__, f"{name}.log", digest(log))
    check(__file__, f"{name}.progress", digest(events))


# --- G5: the pose images -----------------------------------------------------------------------

SIZES = {"160x120": (160, 120), "720x1280": (720, 1280)}
# (body_stick_width, hand_stick_width, draw_head, draw_threshold)
SWITCHES = {
    "default": (-1, -1, True, 0.5),
    "body_0": (0, -1, True, 0.5),
    "hands_0": (-1, 0, True, 0.5),
    "head_false": (-1, -1, False, 0.5),
    "widths_3": (3, 3, True, 0.5),
    "widths_7": (7, 7, True, 0.5),
    "threshold_0": (-1, -1, True, 0.0),
    # at draw_threshold 0.0 body 0 and draw_head False stop hiding their parts (plan 12.D F13)
    "threshold_0_body_0": (0, -1, True, 0.0),
    "threshold_0_head_false": (-1, -1, False, 0.0),
}


def draw_metas(height, width):
    """Two frames of hand-built metas: body confidences around 0.5, the left hand at 0.55 and
    the right at 0.45, a body and a hand point at x = 0, a body point off the canvas."""
    metas = []
    for f, dtype in enumerate((np.float64, np.float32)):
        body = np.zeros((20, 3))
        body[:, 0] = np.linspace(0.2, 0.8, 20) + 0.01 * f
        body[:, 1] = np.linspace(0.1, 0.9, 20)
        body[:, 2] = np.resize([0.45, 0.5, 0.55, 0.6], 20)
        body[3, 0] = 0.0
        body[7, :2] = (1.1, 0.5)
        hands = []
        for x0, conf in ((0.3, 0.55), (0.6, 0.45)):
            hand = np.zeros((21, 3))
            hand[:, 0] = x0 + np.linspace(0.0, 0.1, 21)
            hand[:, 1] = 0.5 + np.linspace(0.0, 0.1, 21) * (-1) ** f
            hand[:, 2] = conf
            hands.append(hand)
        hands[0][4, 0] = 0.0
        face = np.zeros((69, 3))
        face[:, 0], face[:, 1], face[:, 2] = 0.5, 0.15, 0.9
        meta = {"width": width, "height": height, "keypoints_body": body.astype(dtype),
                "keypoints_left_hand": hands[0].astype(dtype), "keypoints_right_hand": hands[1].astype(dtype),
                "keypoints_face": face.astype(dtype)}
        metas.append(AAPoseMeta.from_humanapi_meta(meta))
    return metas


@pytest.mark.parametrize("switches", list(SWITCHES))
@pytest.mark.parametrize("size", list(SIZES))
def test_draw_golden(size, switches, caplog, monkeypatch):
    body, hand, head, threshold = SWITCHES[switches]
    events = []
    monkeypatch.setattr(pose, "ProgressBar", recording_progress_bar(events))
    capture_logs(caplog)
    images = pose.draw({"pose_metas": draw_metas(*SIZES[size])}, body_stick_width=body, hand_stick_width=hand,
                       draw_head=head, draw_threshold=threshold)
    check(__file__, f"draw.{size}.{switches}", digest(images))
    if switches == "default":
        check(__file__, f"draw.{size}.progress", digest(events))
        check(__file__, f"draw.{size}.log", digest(log_text(caplog.records)))


# --- G14: the key frame's body points ----------------------------------------------------------

def test_key_frame_body_points_golden(caplog, monkeypatch):
    _, _, pose_data, _, _, _, _ = run_detect("default", caplog, monkeypatch)
    # frame 0 holds the nose 0.4 px left of the frame (truncates to 0, kept), the right ankle
    # at W - 0.12 (truncates to W - 1, kept), the left ankle below the frame (left out) and the
    # right hip at confidence 0.3
    for threshold in (0.5, 0.0):
        check(__file__, f"key_frame_body_points.{threshold}", pose.key_frame_body_points(pose_data, threshold))
    body = np.zeros((20, 3))
    body[:, 2] = 0.9
    body[list(pose.KEY_FRAME_BODY_POINTS), :2] = [(0.5, 0.1), (1.02, 0.5), (0.25, 0.25), (0.75, 0.25),
                                                   (0.4, 0.6), (0.6, 0.6), (0.4, 0.9), (0.6, 0.9)]
    body[list(pose.KEY_FRAME_BODY_POINTS)[4], 2] = 0.4
    meta = {"pose_metas_original": [{"width": W, "height": H, "keypoints_body": body}]}
    for threshold in (0.5, 0.0):
        check(__file__, f"key_frame_body_points.x_1.02.{threshold}", pose.key_frame_body_points(meta, threshold))
