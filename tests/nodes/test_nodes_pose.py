"""The Pose Detection node: the pose_model widget, confidence_scale on models with and without a
divisor, and supplied boxes that never build the detector. Fake models, synthetic frames. The
node loads models/common/download.py, which imports folder_paths at its top, so this runs where
ComfyUI is importable (with the ComfyUI root on PYTHONPATH) and is skipped elsewhere:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_nodes_pose.py
"""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("folder_paths")

from names import nodes, spec  # noqa: E402
from pose_fakes import FakeDetector, FakePose, frames, loader, pose  # noqa: E402


def test_the_pose_model_widget_lists_the_loader_models():
    assert spec("BCVPoseDetection")["required"]["pose_model"][0] == list(loader.POSE_MODELS)


# --- confidence_scale ----------------------------------------------------------------------

class HeatmapPose(FakePose):
    """A pose model whose confidences are not a scaled score, as ViTPose declares it."""

    conf_scale = None


def test_confidence_scale_is_ignored_on_a_model_without_a_divisor(monkeypatch, caplog):
    monkeypatch.setattr(pose, "_to_device", lambda *models: None)
    images = frames()
    plain, _ = pose.detect(FakeDetector(), HeatmapPose(), images)
    with caplog.at_level("INFO"):
        scaled, _ = pose.detect(FakeDetector(), HeatmapPose(), images, config=pose.PoseConfig(confidence_scale=4.6))
    assert "confidence_scale 4.6 ignored: it applies to RTMW only" in caplog.text
    for a, b in zip(plain["pose_metas_original"], scaled["pose_metas_original"]):
        assert np.array_equal(a["keypoints_body"], b["keypoints_body"])


def test_vitpose_declares_no_divisor_and_rtmw_does():
    from pose_fakes import wrappers
    assert wrappers.ViTPose.conf_scale is None
    assert wrappers.RTMW.conf_scale == 4.6


# --- supplied boxes ------------------------------------------------------------------------

def test_supplied_boxes_never_build_the_detector(monkeypatch):
    monkeypatch.setattr(pose, "_to_device", lambda *models: None)
    built = []

    def build(cls, filename):
        built.append(filename)
        return FakePose() if cls is not loader.Yolo else FakeDetector()

    monkeypatch.setattr(loader, "_load", build)
    out = nodes.BCVPoseDetection().detect(frames(), "ViTPose-H", -1, -1, True, 0.5, bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert built == [loader.POSE_MODELS["ViTPose-H"][1]] and len(out[2]) == len(frames())
    built.clear()
    nodes.BCVPoseDetection().detect(frames(), "ViTPose-H", -1, -1, True, 0.5)
    assert built == [loader.DETECTOR_FILE, loader.POSE_MODELS["ViTPose-H"][1]]
