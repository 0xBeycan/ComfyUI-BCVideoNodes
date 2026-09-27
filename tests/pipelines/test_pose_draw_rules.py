"""The Pose Config draw rule of the pose pipeline: forearm_limit 0 draws the pose images as the
pipeline without it does, and a limit above 0 leaves the overlong forearm's wrist and hand out of the
images and not out of pose_data. Scripted detector and pose model (pose_fakes), hand-built
keypoints; no ComfyUI server and no real model:

    python -m pytest tests/pipelines/test_pose_draw_rules.py
"""
import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from pose_fakes import B, H, W, RecordingPose, ScriptedDetector, no_device, pose  # noqa: E402,F401


def seeded_frames():
    return torch.from_numpy(np.random.default_rng(3).random((B, H, W, 3), dtype=np.float32))


def test_forearm_limit_is_off_by_default_and_follows_the_measured_tunables():
    config = pose.PoseConfig()
    assert config.forearm_limit == 0.0
    # it travels in pose_data with the rest
    assert list(pose.asdict(config)) == ["min_keypoint_conf", "detection_threshold", "box_window", "forearm_limit"]


# -- the draw rule acts on the pose images only -----------------------------------------------------

HAND = [(50.0 + (k % 5) * 3, 0.0 + (k // 5) * 3) for k in range(21)]
# the right forearm's length per frame: 20 px, and 50 px on the last frame (over 2.0 x the 20 px
# median, under 3.0 x it)
RIGHT_FOREARM = (20.0, 20.0, 20.0, 20.0, 50.0)


def overlong_forearm_pose_data(right=RIGHT_FOREARM, left=None):
    """One frame per `right` forearm length, the right hand drawn below the right wrist, as detect
    writes pose_data; the left forearm `left` long on each frame, 20 px when None."""
    def frame(right_length, left_length):
        body = np.zeros((20, 3), np.float32)
        for j, (x, y) in {1: (60, 30), 2: (50, 30), 5: (70, 30), 3: (50, 55), 4: (50, 55 + right_length),
                          6: (70, 55), 7: (70, 55 + left_length)}.items():
            body[j] = (x / W, y / H, 0.9)
        wrist_y = 55 + right_length + 3
        right_hand = np.array([(x / W, (y + wrist_y) / H, 0.9) for x, y in HAND], np.float32)
        left_hand = np.array([((x + 20) / W, (y + 78) / H, 0.9) for x, y in HAND], np.float32)
        return {"width": W, "height": H, "keypoints_body": body, "keypoints_left_hand": left_hand,
                "keypoints_right_hand": right_hand, "keypoints_face": np.zeros((69, 3), np.float32)}

    originals = [frame(r, l) for r, l in zip(right, left or [20.0] * len(right))]
    return {"pose_metas": [pose.AAPoseMeta.from_humanapi_meta(m) for m in originals],
            "pose_metas_original": originals, "draw_threshold": 0.5}


def vendored(meta, hide_right_wrist=False):
    """The vendored drawing of one AAPoseMeta, with the right wrist's and the right hand's
    confidences set below any threshold by hand when asked: the image the rule must produce."""
    if hide_right_wrist:
        meta = copy.copy(meta)
        meta.kps_body_p = meta.kps_body_p.copy()
        meta.kps_body_p[4] = -np.inf
        meta.kps_rhand_p = np.full_like(meta.kps_rhand_p, -np.inf)
    canvas = np.zeros((H, W, 3), np.uint8)
    image = pose.draw_aapose_by_meta_new(canvas, meta, threshold=0.5, draw_body=True, draw_hand=True, draw_head=True,
                                         body_stick_width=-1, hand_stick_width=-1)
    return torch.from_numpy(image).float() / 255.0


def test_at_forearm_limit_0_the_pose_images_are_the_vendored_drawing():
    pose_data = overlong_forearm_pose_data()
    for images in (pose.draw(pose_data), pose.draw(pose_data, forearm_limit=0.0)):
        assert all(torch.equal(images[i], vendored(meta)) for i, meta in enumerate(pose_data["pose_metas"]))


def test_forearm_limit_leaves_the_wrist_and_hand_out_of_the_images_and_not_out_of_pose_data():
    pose_data = overlong_forearm_pose_data()
    before = [(m.kps_rhand_p.copy(), m.kps_body_p.copy()) for m in pose_data["pose_metas"]]
    originals = [{k: np.array(v, copy=True) for k, v in m.items()} for m in pose_data["pose_metas_original"]]
    images = pose.draw(pose_data, forearm_limit=2.0)
    metas = pose_data["pose_metas"]
    assert all(torch.equal(images[i], vendored(metas[i])) for i in range(4))
    assert torch.equal(images[4], vendored(metas[4], hide_right_wrist=True))
    assert not torch.equal(images[4], vendored(metas[4]))
    assert all(np.array_equal(m.kps_rhand_p, rhand) and np.array_equal(m.kps_body_p, body)
               for m, (rhand, body) in zip(metas, before))
    assert all(all(np.array_equal(m[k], v) for k, v in o.items() if isinstance(v, np.ndarray))
               for m, o in zip(pose_data["pose_metas_original"], originals))


def test_a_larger_forearm_limit_hides_fewer_wrists():
    # 50 px is not over 3.0 x the 20 px median: nothing is left out
    pose_data = overlong_forearm_pose_data()
    images = pose.draw(pose_data, forearm_limit=3.0)
    assert all(torch.equal(images[i], vendored(meta)) for i, meta in enumerate(pose_data["pose_metas"]))


def test_pose_detection_draws_with_the_configs_forearm_limit(caplog):
    with caplog.at_level("INFO"):
        pose.pose_detection(seeded_frames(), ScriptedDetector(), RecordingPose(),
                            config=pose.PoseConfig(forearm_limit=2.0))
    done = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[BCVideoNodes] drawing ")
            and " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(forearm_rule frames 0)")
    assert forearm_warnings(caplog) == []
    caplog.clear()
    with caplog.at_level("INFO"):
        pose.pose_detection(seeded_frames(), ScriptedDetector(), RecordingPose())
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage() and "drawing" in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("s")


# -- the warning: what the pose images leave out, one line per side ----------------------------------

def forearm_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "forearm_limit" in r.getMessage()]


def test_forearm_limit_warns_once_per_side_it_fired_on(caplog):
    # 12 frames, both forearms 20 px (the median); the right one 41, 50, 45 and 61 px on frames 3,
    # 6, 7 and 10 (2.05, 2.50, 2.25 and 3.05 x), the left one 60 px on frame 0 (3.00 x)
    right, left = [20.0] * 12, [60.0] + [20.0] * 11
    right[3], right[6], right[7], right[10] = 41.0, 50.0, 45.0, 61.0
    with caplog.at_level("INFO"):
        pose.draw(overlong_forearm_pose_data(right, left), forearm_limit=2.0)
    assert forearm_warnings(caplog) == [
        "[BCVideoNodes] forearm_limit 2.0: right wrist and hand left out of the pose images on 4 frames "
        "(3, 6-7, 10): forearm 2.05-3.05x its clip median of 20 px; pose_data keeps the keypoints",
        "[BCVideoNodes] forearm_limit 2.0: left wrist and hand left out of the pose images on 1 frame "
        "(0): forearm 3.00x its clip median of 20 px; pose_data keeps the keypoints"]
    # the step line keeps its format: the frames any side fired on
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(forearm_rule frames 5)")


def test_no_warning_at_forearm_limit_0_or_when_nothing_is_over_it(caplog):
    # the 50 px forearm is 2.5 x its median: over 2.0, not over 3.0
    with caplog.at_level("INFO"):
        pose.draw(overlong_forearm_pose_data())
        pose.draw(overlong_forearm_pose_data(), forearm_limit=0.0)
        pose.draw(overlong_forearm_pose_data(), forearm_limit=3.0)
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
    with caplog.at_level("INFO"):
        pose.draw(overlong_forearm_pose_data(), forearm_limit=2.0)
    assert forearm_warnings(caplog) == [
        "[BCVideoNodes] forearm_limit 2.0: right wrist and hand left out of the pose images on 1 frame "
        "(4): forearm 2.50x its clip median of 20 px; pose_data keeps the keypoints"]
