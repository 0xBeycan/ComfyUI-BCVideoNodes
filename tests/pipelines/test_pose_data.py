"""The POSEDATA contract of libs/pose_data.py against the dicts the pose pipeline writes: the
keys of PoseData, PoseMeta and Detection, in order, are the keys `detect` and `pose_detection`
produce, so the annotations and the runtime dicts cannot drift apart. Runs on pose_fakes' scripted
clip and models (ScriptedDetector and RecordingPose):

    python -m pytest tests/pipelines/test_pose_data.py
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from pose_fakes import B, H, W, RecordingPose, ScriptedDetector, no_device, pose  # noqa: E402,F401

from bcvideonodes.libs.pose_data import Detection, PoseData, PoseMeta  # noqa: E402

def seeded_frames():
    return torch.from_numpy(np.random.default_rng(0).random((B, H, W, 3), dtype=np.float32))


def assert_frames_follow_the_contract(pose_data):
    assert len(pose_data["pose_metas_original"]) == len(pose_data["detections"]) == B
    for meta in pose_data["pose_metas_original"]:
        assert list(meta) == list(PoseMeta.__annotations__)
    for detection in pose_data["detections"]:
        assert list(detection) == list(Detection.__annotations__)


def test_detect_writes_the_first_four_pose_data_keys_in_order():
    pose_data, _ = pose.detect(ScriptedDetector(), RecordingPose(), seeded_frames())
    assert list(pose_data) == list(PoseData.__annotations__)[:4]
    assert_frames_follow_the_contract(pose_data)


def test_pose_detection_writes_all_five_pose_data_keys_in_order():
    _, pose_data, _, _ = pose.pose_detection(seeded_frames(), ScriptedDetector(), RecordingPose())
    assert list(pose_data) == list(PoseData.__annotations__)
    assert_frames_follow_the_contract(pose_data)
