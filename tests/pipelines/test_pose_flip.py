"""ViTPose's test-time flip switch in PoseConfig, through pose.detect on fake models: on for
ViTPose, ignored in one line for RTMW. No ComfyUI, no real model:

    python -m pytest tests/pipelines/test_pose_flip.py
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from pose_fakes import FakeDetector, FakePose, FakeViTPose, frames, no_device, pose  # noqa: E402,F401


def test_flip_test_runs_the_mirror_and_default_off_does_not():
    model = FakeViTPose()
    pose.detect(FakeDetector(), model, frames(), config=pose.PoseConfig(temporal=False))
    assert model.calls == len(frames()) and model.runs == len(frames())
    model = FakeViTPose()
    pose.detect(FakeDetector(), model, frames(), config=pose.PoseConfig(temporal=False, flip_test=True))
    assert model.calls == 0 and model.runs == 2 * len(frames())


def test_rtmw_ignores_flip_test_in_one_line(caplog):
    model = FakePose()   # no heatmaps: stands in for RTMW
    model.architecture = "rtmw"
    with caplog.at_level("INFO"):
        plain, _ = pose.detect(FakeDetector(), FakePose(), frames(), config=pose.PoseConfig(temporal=False))
        flipped, _ = pose.detect(FakeDetector(), model, frames(), config=pose.PoseConfig(temporal=False, flip_test=True))
    lines = [r.getMessage() for r in caplog.records if "flip_test" in r.getMessage()]
    assert len(lines) == 1 and "ViTPose only" in lines[0]
    assert model.calls == len(frames())
    for a, b in zip(plain["pose_metas_original"], flipped["pose_metas_original"]):
        assert np.array_equal(a["keypoints_body"], b["keypoints_body"])
