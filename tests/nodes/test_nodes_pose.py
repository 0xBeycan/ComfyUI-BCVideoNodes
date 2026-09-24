"""The Pose Detection node: supplied boxes that never build the detector. Fake models, synthetic frames. The
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

from names import nodes  # noqa: E402
from pose_fakes import FakeDetector, FakePose, frames, loader, pose  # noqa: E402


def test_supplied_boxes_never_build_the_detector(monkeypatch):
    monkeypatch.setattr(pose, "_to_device", lambda *models: None)
    built = []

    def build(cls, filename):
        built.append(filename)
        return FakePose() if cls is not loader.Yolo else FakeDetector()

    monkeypatch.setattr(loader, "_load", build)
    out = nodes.BCVPoseDetection().detect(frames(), -1, -1, True, 0.5, bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert built == [loader.POSE_FILE] and len(out[2]) == len(frames())
    built.clear()
    nodes.BCVPoseDetection().detect(frames(), -1, -1, True, 0.5)
    assert built == [loader.DETECTOR_FILE, loader.POSE_FILE]
