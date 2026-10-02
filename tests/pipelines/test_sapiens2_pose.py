"""The Sapiens2 Pose pipeline (pipelines/sapiens2_pose.py) with stand-in models: the hybrid keypoints
(the face rows 23-90 from ViTPose-H, everything else from Sapiens2), Pose Detection's flow around them
(the boxes, the pose_data contract, the drawing outputs) and the Sapiens2 batches. No ComfyUI and no
real model:

    python -m pytest tests/pipelines/test_sapiens2_pose.py
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from pose_fakes import B, H, W, FakeDetector, RecordingPose, frames, no_device, pose  # noqa: E402,F401
from sapiens2_fakes import FakeSapiens2, sapiens2  # noqa: E402

FACE = list(range(23, 91))
NOT_FACE = [i for i in range(133) if i not in FACE]


def test_the_face_rows_are_vitposes_and_every_other_row_is_sapiens2s():
    images_np = frames().numpy()
    boxes = [np.array([30.0 + i, 20.0, 90.0 + i, 140.0, 1.0]) for i in range(B)]
    vitpose, model = RecordingPose(), FakeSapiens2()
    keypoints = sapiens2.hybrid_keypoints(model, vitpose, images_np, boxes)
    assert keypoints.shape == (B, 133, 3)
    # ViTPose-H ran once per frame, on Pose Detection's crop of the same box
    assert len(vitpose.calls) == B
    assert [tuple(call[1][0]) for call in vitpose.calls] == [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes]
    for i in range(B):
        assert np.array_equal(keypoints[i, FACE], vitpose.keypoints(i)[FACE])
    # the Sapiens2 stand-in's keypoints: its crop centre (the box centre) plus its per-keypoint offset
    centres = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in boxes])
    expected_x = centres[:, :1] + np.linspace(-30, 30, 133)[None]
    assert np.allclose(keypoints[:, NOT_FACE, 0], expected_x[:, NOT_FACE], atol=1e-4)
    assert (keypoints[:, NOT_FACE, 2] == np.float32(0.7)).all()


def test_a_half_clip_gives_both_models_the_float32_runs_crops(monkeypatch):
    # Load Video at precision fp16: both models' frames are read one at a time, each as the float32
    # clip's levels
    from video_input_fakes import levels, record_reads, video

    class Recording(FakeSapiens2):
        def __init__(self):
            super().__init__()
            self.crops = []

        def __call__(self, crops, centers, scales):
            self.crops.append(crops.copy())
            return super().__call__(crops, centers, scales)

    def run(images_np):
        model, vitpose = Recording(), RecordingPose()
        return sapiens2.hybrid_keypoints(model, vitpose, images_np, boxes), model.crops, vitpose.calls

    exact = levels(B, H, W, 3)
    boxes = [np.array([30.0 + i, 20.0, 90.0 + i, 140.0, 1.0]) for i in range(B)]
    keypoints, crops, calls = run(exact.numpy())
    reads = record_reads(monkeypatch)
    half_keypoints, half_crops, half_calls = run(video.as_numpy(exact.half()))
    assert reads == [(H, W, 3)] * (2 * B)
    assert np.array_equal(half_keypoints, keypoints)
    assert all(np.array_equal(a, b) for a, b in zip(half_crops, crops))
    assert all(np.array_equal(a[0], b[0]) for a, b in zip(half_calls, calls))


def test_sapiens2_runs_in_batches_and_reports_progress():
    model, done = FakeSapiens2(), []
    boxes = [np.array([30.0, 20.0, 90.0, 140.0, 1.0])] * B
    sapiens2.hybrid_keypoints(model, RecordingPose(), frames().numpy(), boxes, done.append)
    size = sapiens2.BATCH_SIZE
    assert model.calls == [min(size, B - start) for start in range(0, B, size)]
    assert done == [min(start + size, B) for start in range(0, B, size)]


def test_the_node_pipeline_is_pose_detections_flow_with_the_hybrid_keypoints():
    images, vitpose, model = frames(), RecordingPose(), FakeSapiens2()
    config = pose.PoseConfig(box_window=2)
    out = sapiens2.sapiens2_pose(images, FakeDetector(), model, vitpose, config=config, draw_threshold=0.4)
    pose_images, pose_data, boxes, key_points = out
    # Pose Detection's boxes and pose_data keys, from the same detector and config
    detected, detected_boxes = pose.detect(FakeDetector(), RecordingPose(), images, config=config)
    assert boxes == detected_boxes
    assert list(pose_data) == ["pose_metas", "pose_metas_original", "detections", "pose_config", "draw_threshold"]
    assert pose_data["detections"] == detected["detections"] and pose_data["pose_config"] == pose.asdict(config)
    assert pose_data["draw_threshold"] == 0.4 and pose_images.shape == (B, H, W, 3)
    # the face in pose_data is ViTPose's: keypoints_face rows 1-68 are COCO-WholeBody 23-90
    meta = pose_data["pose_metas_original"][3]
    face = vitpose.keypoints(3)[FACE]
    assert np.allclose(meta["keypoints_face"][1:, :2], face[:, :2] / [W, H], atol=1e-6)
    assert np.array_equal(meta["keypoints_face"][1:, 2], face[:, 2])
    # the body is Sapiens2's: the nose at the stand-in's crop centre plus its first offset
    x1, y1, x2, y2 = boxes[3]
    assert meta["keypoints_body"][0, 0] == pytest.approx(((x1 + x2) / 2 - 30) / W, abs=1e-6)
    assert meta["keypoints_body"][0, 2] == np.float32(0.7)
    assert key_points == pose.key_frame_body_points(pose_data, 0.4)


def test_supplied_boxes_skip_the_detector():
    class NoDetector:
        def __call__(self, *args):
            raise AssertionError("the detector must not run when boxes are supplied")

    _, pose_data, boxes, _ = sapiens2.sapiens2_pose(frames(), NoDetector(), FakeSapiens2(), RecordingPose(),
                                                    bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert boxes == [(30.0, 20.0, 90.0, 140.0)] * B
    assert all(d["score"] == 1.0 for d in pose_data["detections"])


def test_the_detector_and_both_pose_models_go_to_the_device_together(monkeypatch):
    loaded = []
    monkeypatch.setattr(pose, "_to_device", lambda *models: loaded.append(models))
    detector, vitpose, model = FakeDetector(), RecordingPose(), FakeSapiens2()
    sapiens2.sapiens2_pose(frames(), detector, model, vitpose)
    sapiens2.sapiens2_pose(frames(), None, model, vitpose, bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert loaded == [(detector, model, vitpose), (model, vitpose)]
