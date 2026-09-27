"""The Pose Config draw rules of the pose pipeline: limb_dedup and back_view_face off and
forearm_limit 0 draw the pose images as the pipeline without them does; limb_dedup on leaves a hand
drawn on the other hand and the elbow, wrist and hand of an arm drawn on the other arm out of the
images, a forearm_limit above 0 the overlong forearm's wrist and hand, back_view_face on the nose and
eyes of a face on the back of the head, and none of them out of pose_data. Scripted detector and pose
model (pose_fakes), hand-built keypoints; no ComfyUI server and no real model:

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


def test_the_draw_rules_are_off_by_default_and_follow_the_measured_tunables():
    config = pose.PoseConfig()
    assert config.forearm_limit == 0.0 and config.limb_dedup is False and config.back_view_face is False
    # they travel in pose_data with the rest
    assert list(pose.asdict(config)) == ["min_keypoint_conf", "detection_threshold", "box_window", "forearm_limit",
                                         "limb_dedup", "back_view_face"]
    # limb_dedup is the one switch: the separate hand and arm switches are gone
    for gone in ("hand_dedup", "mirror_rule"):
        with pytest.raises(TypeError, match=gone):
            pose.PoseConfig(**{gone: True})


# -- the draw rule acts on the pose images only -----------------------------------------------------

HAND = [(50.0 + (k % 5) * 3, 0.0 + (k // 5) * 3) for k in range(21)]
# the right forearm's length per frame: 20 px, and 50 px on the last frame (over 2.0 x the 20 px
# median, under 3.0 x it)
RIGHT_FOREARM = (20.0, 20.0, 20.0, 20.0, 50.0)


# each side's (elbow, wrist) in the AAPose layout
ARMS = {"right": (3, 4), "left": (6, 7)}


def overlong_forearm_pose_data(right=RIGHT_FOREARM, left=None, copies=None, mirrors=None, heads=None):
    """One frame per `right` forearm length, the right hand drawn below the right wrist, as detect
    writes pose_data; the left forearm `left` long on each frame, 20 px when None, the left hand 20 px
    right of the right hand's place at 20 px. The shoulders are 20 px apart and the nose, not drawn,
    10 px above the neck: the body scale is the shoulders' 20 px, the arm test's reach 4 px, and the
    two arms 20 px apart are two arms. `mirrors` maps a frame to the side whose elbow and wrist are
    drawn on the other arm there (2 px right of the other side's) with its elbow less confident (0.6).
    `copies` maps a frame to the side whose hand is drawn on the other hand there (2 px right of it:
    2 / 17 px of the hand's diagonal) with that side's elbow under the 0.5 draw threshold (0.3).
    `heads` maps a frame to (seen, jaw): its nose, eyes and ears drawn (0.9) above the neck, the 17
    jaw-line face keypoints (keypoints_face rows 1-17) `jaw` confident, and when `seen` is "back" the
    two shoulders swapped (the left one on the image left: a body seen from behind)."""
    def frame(i, right_length, left_length):
        body = np.zeros((20, 3), np.float32)
        for j, (x, y) in {1: (60, 30), 2: (50, 30), 5: (70, 30), 3: (50, 55), 4: (50, 55 + right_length),
                          6: (70, 55), 7: (70, 55 + left_length)}.items():
            body[j] = (x / W, y / H, 0.9)
        body[0] = (60 / W, 20 / H, 0.0)
        mirror = (mirrors or {}).get(i)
        if mirror:
            (elbow, wrist), other = ARMS[mirror], ARMS["left" if mirror == "right" else "right"]
            body[[elbow, wrist]] = body[list(other)] + np.array([2 / W, 0, 0], np.float32)
            body[elbow, 2] = 0.6
        face = np.zeros((69, 3), np.float32)
        head = (heads or {}).get(i)
        if head:
            seen, jaw = head
            for j, (x, y) in {0: (60, 20), 14: (57, 17), 15: (63, 17), 16: (54, 19), 17: (66, 19)}.items():
                body[j] = (x / W, y / H, 0.9)
            if seen == "back":
                body[[2, 5]] = body[[5, 2]]
            face[1:18, 2] = jaw
        wrist_y = 55 + right_length + 3
        right_hand = np.array([(x / W, (y + wrist_y) / H, 0.9) for x, y in HAND], np.float32)
        left_hand = np.array([((x + 20) / W, (y + 78) / H, 0.9) for x, y in HAND], np.float32)
        copy = (copies or {}).get(i)
        if copy == "left":
            left_hand, body[6, 2] = right_hand + np.array([2 / W, 0, 0], np.float32), 0.3
        elif copy == "right":
            right_hand, body[3, 2] = left_hand + np.array([2 / W, 0, 0], np.float32), 0.3
        return {"width": W, "height": H, "keypoints_body": body, "keypoints_left_hand": left_hand,
                "keypoints_right_hand": right_hand, "keypoints_face": face}

    originals = [frame(i, r, l) for i, (r, l) in enumerate(zip(right, left or [20.0] * len(right)))]
    return {"pose_metas": [pose.AAPoseMeta.from_humanapi_meta(m) for m in originals],
            "pose_metas_original": originals, "draw_threshold": 0.5}


def vendored(meta, hide_right_wrist=False, hide_left_hand=False, hide_left_arm=False, hide_face=False):
    """The vendored drawing of one AAPoseMeta, with the right wrist's and the right hand's
    confidences, the left hand's, the left elbow's, wrist's and hand's, or the nose's and the eyes'
    set below any threshold by hand when asked: the image the rules must produce."""
    meta = copy.copy(meta)
    meta.kps_body_p = meta.kps_body_p.copy()
    if hide_face:
        meta.kps_body_p[[0, 14, 15]] = -np.inf
    if hide_right_wrist:
        meta.kps_body_p[4] = -np.inf
        meta.kps_rhand_p = np.full_like(meta.kps_rhand_p, -np.inf)
    if hide_left_arm:
        meta.kps_body_p[[6, 7]] = -np.inf
    if hide_left_hand or hide_left_arm:
        meta.kps_lhand_p = np.full_like(meta.kps_lhand_p, -np.inf)
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


def test_pose_detection_draws_with_the_configs_draw_rules(caplog):
    with caplog.at_level("INFO"):
        pose.pose_detection(seeded_frames(), ScriptedDetector(), RecordingPose(),
                            config=pose.PoseConfig(forearm_limit=2.0))
    done = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[BCVideoNodes] drawing ")
            and " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(forearm_rule frames 0)")
    caplog.clear()
    with caplog.at_level("INFO"):
        pose.pose_detection(seeded_frames(), ScriptedDetector(), RecordingPose(),
                            config=pose.PoseConfig(forearm_limit=2.0, limb_dedup=True))
    done = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[BCVideoNodes] drawing ")
            and " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(limb_dedup frames 0, forearm_rule frames 0)")
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
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


# -- limb_dedup: a hand drawn on the other hand, an arm drawn on the other arm ----------------------

# the left hand drawn on the right hand with the left elbow not drawn on frames 1-2 (each the other's
# broken neighbour: both fire) and on frame 4 alone (its neighbours' left arms intact: it does not)
COPIES = {1: "left", 2: "left", 4: "left"}
# the left arm drawn on the right arm on frames 1-2 and on frame 4 alone: each frame is read alone,
# all three fire
MIRRORS = {1: "left", 2: "left", 4: "left"}


def union_pose_data():
    """6 frames: the left arm on the right arm on frame 2 (the arm test), the left hand on the right
    hand on frame 3 (the hand test, frame 4 its broken neighbour), both on frame 4; the right forearm
    50 px on frame 4 (2.5 x its 20 px median)."""
    return overlong_forearm_pose_data(right=RIGHT_FOREARM + (20.0,), copies={3: "left", 4: "left"},
                                      mirrors={2: "left", 4: "left"})


def limb_dedup_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "limb_dedup" in r.getMessage()]


def test_limb_dedup_off_draws_the_vendored_images():
    for pose_data in (overlong_forearm_pose_data(right=[20.0] * 6, copies=COPIES),
                      overlong_forearm_pose_data(right=[20.0] * 6, mirrors=MIRRORS), union_pose_data()):
        for images in (pose.draw(pose_data), pose.draw(pose_data, limb_dedup=False)):
            assert all(torch.equal(images[i], vendored(meta)) for i, meta in enumerate(pose_data["pose_metas"]))


def test_limb_dedup_leaves_a_hand_copy_out_of_the_images_and_not_out_of_pose_data():
    pose_data = overlong_forearm_pose_data(right=[20.0] * 6, copies=COPIES)
    metas = pose_data["pose_metas"]
    before = [(m.kps_lhand_p.copy(), m.kps_rhand_p.copy(), m.kps_body_p.copy()) for m in metas]
    originals = [{k: np.array(v, copy=True) for k, v in m.items()} for m in pose_data["pose_metas_original"]]
    images = pose.draw(pose_data, limb_dedup=True)
    assert all(torch.equal(images[i], vendored(metas[i], hide_left_hand=i in (1, 2))) for i in range(6))
    assert not torch.equal(images[1], vendored(metas[1]))
    assert all(np.array_equal(m.kps_lhand_p, lhand) and np.array_equal(m.kps_rhand_p, rhand)
               and np.array_equal(m.kps_body_p, body) for m, (lhand, rhand, body) in zip(metas, before))
    assert all(all(np.array_equal(m[k], v) for k, v in o.items() if isinstance(v, np.ndarray))
               for m, o in zip(pose_data["pose_metas_original"], originals))


def test_limb_dedup_leaves_an_arm_copy_out_of_the_images_and_not_out_of_pose_data():
    pose_data = overlong_forearm_pose_data(right=[20.0] * 6, mirrors=MIRRORS)
    metas = pose_data["pose_metas"]
    before = [(m.kps_lhand_p.copy(), m.kps_rhand_p.copy(), m.kps_body_p.copy()) for m in metas]
    originals = [{k: np.array(v, copy=True) for k, v in m.items()} for m in pose_data["pose_metas_original"]]
    images = pose.draw(pose_data, limb_dedup=True)
    assert all(torch.equal(images[i], vendored(metas[i], hide_left_arm=i in MIRRORS)) for i in range(6))
    assert not torch.equal(images[1], vendored(metas[1]))
    assert all(np.array_equal(m.kps_lhand_p, lhand) and np.array_equal(m.kps_rhand_p, rhand)
               and np.array_equal(m.kps_body_p, body) for m, (lhand, rhand, body) in zip(metas, before))
    assert all(all(np.array_equal(m[k], v) for k, v in o.items() if isinstance(v, np.ndarray))
               for m, o in zip(pose_data["pose_metas_original"], originals))


def test_limb_dedup_leaves_out_what_either_test_names(caplog):
    pose_data = union_pose_data()
    metas = pose_data["pose_metas"]
    with caplog.at_level("INFO"):
        images = pose.draw(pose_data, limb_dedup=True)
    assert all(torch.equal(images[i], vendored(metas[i])) for i in (0, 1, 5))
    assert torch.equal(images[2], vendored(metas[2], hide_left_arm=True))
    assert torch.equal(images[3], vendored(metas[3], hide_left_hand=True))
    assert torch.equal(images[4], vendored(metas[4], hide_left_arm=True))
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
        "[BCVideoNodes] limb_dedup: left hand left out of the pose images on 2 frames (3-4): drawn on the other "
        "hand; pose_data keeps the keypoints",
        "[BCVideoNodes] limb_dedup: left elbow, wrist and hand left out of the pose images on 2 frames (2, 4): "
        "drawn on the other arm; pose_data keeps the keypoints"]
    # the step line counts the frames limb_dedup left anything out of
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(limb_dedup frames 3)")


def test_limb_dedup_and_forearm_limit_leave_out_what_either_names(caplog):
    pose_data = union_pose_data()
    metas = pose_data["pose_metas"]
    with caplog.at_level("INFO"):
        images = pose.draw(pose_data, forearm_limit=2.0, limb_dedup=True)
    assert all(torch.equal(images[i], vendored(metas[i])) for i in (0, 1, 5))
    assert torch.equal(images[2], vendored(metas[2], hide_left_arm=True))
    assert torch.equal(images[3], vendored(metas[3], hide_left_hand=True))
    assert torch.equal(images[4], vendored(metas[4], hide_right_wrist=True, hide_left_arm=True))
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
        "[BCVideoNodes] limb_dedup: left hand left out of the pose images on 2 frames (3-4): drawn on the other "
        "hand; pose_data keeps the keypoints",
        "[BCVideoNodes] limb_dedup: left elbow, wrist and hand left out of the pose images on 2 frames (2, 4): "
        "drawn on the other arm; pose_data keeps the keypoints",
        "[BCVideoNodes] forearm_limit 2.0: right wrist and hand left out of the pose images on 1 frame (4): "
        "forearm 2.50x its clip median of 20 px; pose_data keeps the keypoints"]
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(limb_dedup frames 3, forearm_rule frames 1)")


def test_a_hand_copy_and_an_overlong_forearm_on_one_frame_leave_out_both(caplog):
    # the right forearm 50 px on frame 4 (2.5 x its 20 px median), the left hand on the right hand
    # on frames 3-4
    pose_data = overlong_forearm_pose_data(copies={3: "left", 4: "left"})
    metas = pose_data["pose_metas"]
    with caplog.at_level("INFO"):
        images = pose.draw(pose_data, forearm_limit=2.0, limb_dedup=True)
    assert all(torch.equal(images[i], vendored(metas[i])) for i in range(3))
    assert torch.equal(images[3], vendored(metas[3], hide_left_hand=True))
    assert torch.equal(images[4], vendored(metas[4], hide_right_wrist=True, hide_left_hand=True))
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
        "[BCVideoNodes] limb_dedup: left hand left out of the pose images on 2 frames (3-4): drawn on the other "
        "hand; pose_data keeps the keypoints",
        "[BCVideoNodes] forearm_limit 2.0: right wrist and hand left out of the pose images on 1 frame (4): "
        "forearm 2.50x its clip median of 20 px; pose_data keeps the keypoints"]
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(limb_dedup frames 2, forearm_rule frames 1)")


def test_limb_dedup_warns_once_per_side_it_left_a_hand_out_on(caplog):
    # 12 frames: the left hand on the right one on frames 0-1, and on frame 10 alone (does not fire);
    # the right hand on the left one on frames 5-7
    copies = {0: "left", 1: "left", 5: "right", 6: "right", 7: "right", 10: "left"}
    with caplog.at_level("INFO"):
        pose.draw(overlong_forearm_pose_data(right=[20.0] * 12, copies=copies), limb_dedup=True)
    assert limb_dedup_warnings(caplog) == [
        "[BCVideoNodes] limb_dedup: right hand left out of the pose images on 3 frames (5-7): drawn on the other "
        "hand; pose_data keeps the keypoints",
        "[BCVideoNodes] limb_dedup: left hand left out of the pose images on 2 frames (0-1): drawn on the other "
        "hand; pose_data keeps the keypoints"]
    # the step line keeps its format: the frames any side fired on
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(limb_dedup frames 5)")


def test_limb_dedup_warns_once_per_side_it_left_an_arm_out_on(caplog):
    # 12 frames: the left arm on the right one on frames 0-1 and 10, the right arm on the left one on
    # frames 5-7
    mirrors = {0: "left", 1: "left", 5: "right", 6: "right", 7: "right", 10: "left"}
    with caplog.at_level("INFO"):
        pose.draw(overlong_forearm_pose_data(right=[20.0] * 12, mirrors=mirrors), limb_dedup=True)
    assert limb_dedup_warnings(caplog) == [
        "[BCVideoNodes] limb_dedup: right elbow, wrist and hand left out of the pose images on 3 frames (5-7): "
        "drawn on the other arm; pose_data keeps the keypoints",
        "[BCVideoNodes] limb_dedup: left elbow, wrist and hand left out of the pose images on 3 frames (0-1, 10): "
        "drawn on the other arm; pose_data keeps the keypoints"]
    # the step line keeps its format: the frames any side fired on
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(limb_dedup frames 6)")


def test_no_limb_dedup_warning_when_off_or_when_nothing_fires(caplog):
    with caplog.at_level("INFO"):
        pose.draw(union_pose_data())
        pose.draw(union_pose_data(), limb_dedup=False)
        # the hand copy on frame 4 alone: a one-frame dip of the left arm
        pose.draw(overlong_forearm_pose_data(right=[20.0] * 6, copies={4: "left"}), limb_dedup=True)
        # the two arms 20 px apart and the two hands apart on every frame
        pose.draw(overlong_forearm_pose_data(right=[20.0] * 6), limb_dedup=True)
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert [line.endswith("s") for line in done[:2]] == [True, True]
    assert [line.endswith("(limb_dedup frames 0)") for line in done[2:]] == [True, True]


# -- back_view_face: a face drawn on the back of the head --------------------------------------------

# frames 1-2 and 4: the body seen from behind, its jaw line 0.6 (under 0.835): the nose and eyes are
# left out; frame 3: seen from the front, and frame 5: from behind with the jaw line 0.9: the face is
# kept; frame 0 has no head drawn
HEADS = {1: ("back", 0.6), 2: ("back", 0.6), 3: ("front", 0.6), 4: ("back", 0.6), 5: ("back", 0.9)}


def back_view_face_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "back_view_face" in r.getMessage()]


def test_back_view_face_off_draws_the_vendored_images():
    pose_data = overlong_forearm_pose_data(right=[20.0] * 6, heads=HEADS)
    for images in (pose.draw(pose_data), pose.draw(pose_data, back_view_face=False)):
        assert all(torch.equal(images[i], vendored(meta)) for i, meta in enumerate(pose_data["pose_metas"]))


def test_back_view_face_leaves_the_nose_and_eyes_out_of_the_images_and_not_out_of_pose_data(caplog):
    pose_data = overlong_forearm_pose_data(right=[20.0] * 6, heads=HEADS)
    metas = pose_data["pose_metas"]
    before = [m.kps_body_p.copy() for m in metas]
    originals = [{k: np.array(v, copy=True) for k, v in m.items()} for m in pose_data["pose_metas_original"]]
    with caplog.at_level("INFO"):
        images = pose.draw(pose_data, back_view_face=True)
    assert all(torch.equal(images[i], vendored(metas[i], hide_face=i in (1, 2, 4))) for i in range(6))
    assert not torch.equal(images[1], vendored(metas[1]))
    # the ears' dots stay, around (54, 19) and (66, 19); around the nose (60, 20) nothing is left
    assert all(images[i][16:23, 51:58].sum() > 0 and images[i][16:23, 63:70].sum() > 0 for i in (1, 2, 4))
    assert all(images[i][18:23, 58:63].sum() == 0 for i in (1, 2, 4))
    assert all(images[i][18:23, 58:63].sum() > 0 for i in (3, 5))
    assert all(np.array_equal(m.kps_body_p, body) for m, body in zip(metas, before))
    assert all(all(np.array_equal(m[k], v) for k, v in o.items() if isinstance(v, np.ndarray))
               for m, o in zip(pose_data["pose_metas_original"], originals))
    assert back_view_face_warnings(caplog) == [
        "[BCVideoNodes] back_view_face: nose and eyes left out of the pose images on 3 frames (1-2, 4): seen from "
        "behind; pose_data keeps the keypoints"]
    # the step line counts the frames back_view_face left the nose and eyes out of
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(back_view_face frames 3)")


def test_back_view_face_on_one_frame(caplog):
    with caplog.at_level("INFO"):
        images = pose.draw(overlong_forearm_pose_data(right=[20.0], heads={0: ("back", 0.6)}), back_view_face=True)
    assert back_view_face_warnings(caplog) == [
        "[BCVideoNodes] back_view_face: nose and eyes left out of the pose images on 1 frame (0): seen from behind; "
        "pose_data keeps the keypoints"]
    assert images[0][18:23, 58:63].sum() == 0


def test_no_back_view_face_warning_when_off_or_when_nothing_is_left_out(caplog):
    with caplog.at_level("INFO"):
        pose.draw(overlong_forearm_pose_data(right=[20.0] * 6, heads=HEADS))
        pose.draw(overlong_forearm_pose_data(right=[20.0] * 6, heads=HEADS), back_view_face=False)
        # seen from the front, or the jaw line over the cut
        pose.draw(overlong_forearm_pose_data(right=[20.0] * 6, heads={3: ("front", 0.6), 5: ("back", 0.9)}),
                  back_view_face=True)
        # the nose and eyes (0.9) are not drawn at a draw threshold of 0.95: nothing to leave out
        pose.draw(overlong_forearm_pose_data(right=[20.0] * 6, heads=HEADS), draw_threshold=0.95, back_view_face=True)
    assert [r for r in caplog.records if r.levelname == "WARNING"] == []
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert [line.endswith("s") for line in done[:2]] == [True, True]
    assert [line.endswith("(back_view_face frames 0)") for line in done[2:]] == [True, True]


def test_back_view_face_limb_dedup_and_forearm_limit_leave_out_what_any_names(caplog):
    # union_pose_data's parts (the left arm on the right arm on frames 2 and 4, the left hand on the
    # right hand on frames 3-4, the right forearm overlong on frame 4) and a face on the back of the
    # head on frames 1 and 4
    pose_data = overlong_forearm_pose_data(right=RIGHT_FOREARM + (20.0,), copies={3: "left", 4: "left"},
                                           mirrors={2: "left", 4: "left"}, heads={1: ("back", 0.6), 4: ("back", 0.6)})
    metas = pose_data["pose_metas"]
    with caplog.at_level("INFO"):
        images = pose.draw(pose_data, forearm_limit=2.0, limb_dedup=True, back_view_face=True)
    assert all(torch.equal(images[i], vendored(metas[i])) for i in (0, 5))
    assert torch.equal(images[1], vendored(metas[1], hide_face=True))
    assert torch.equal(images[2], vendored(metas[2], hide_left_arm=True))
    assert torch.equal(images[3], vendored(metas[3], hide_left_hand=True))
    assert torch.equal(images[4], vendored(metas[4], hide_right_wrist=True, hide_left_arm=True, hide_face=True))
    assert [r.getMessage() for r in caplog.records if r.levelname == "WARNING"] == [
        "[BCVideoNodes] limb_dedup: left hand left out of the pose images on 2 frames (3-4): drawn on the other "
        "hand; pose_data keeps the keypoints",
        "[BCVideoNodes] limb_dedup: left elbow, wrist and hand left out of the pose images on 2 frames (2, 4): "
        "drawn on the other arm; pose_data keeps the keypoints",
        "[BCVideoNodes] forearm_limit 2.0: right wrist and hand left out of the pose images on 1 frame (4): "
        "forearm 2.50x its clip median of 20 px; pose_data keeps the keypoints",
        "[BCVideoNodes] back_view_face: nose and eyes left out of the pose images on 2 frames (1, 4): seen from "
        "behind; pose_data keeps the keypoints"]
    done = [r.getMessage() for r in caplog.records if " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(limb_dedup frames 3, forearm_rule frames 1, back_view_face frames 2)")


def test_pose_detection_reads_the_jaw_line_from_the_models_keypoints_23_to_39(caplog):
    # RecordingPose's layout has the left shoulder (COCO-WholeBody 5, x 52) on the image left of the
    # right one (6, x 56): seen from behind, the nose and eyes 0.9. Its jaw line (COCO-WholeBody 23-39)
    # at 0.83 leaves them out of every frame, with the right heel (22) and keypoint 40 at 1.0: one row
    # off, the mean would be (1.0 + 16 x 0.83) / 17 = 0.84. The heel and the rest of the face (40-90)
    # at 0 with the jaw line at 0.9 leave them in
    def detection(conf):
        model = RecordingPose()
        for keypoints, value in conf.items():
            model.layout[keypoints, 2] = value
        with caplog.at_level("INFO"):
            pose.pose_detection(seeded_frames(), ScriptedDetector(), model, config=pose.PoseConfig(back_view_face=True))

    detection({slice(23, 40): 0.83, 22: 1.0, 40: 1.0})
    assert back_view_face_warnings(caplog) == [
        "[BCVideoNodes] back_view_face: nose and eyes left out of the pose images on 12 frames (0-11): seen from "
        "behind; pose_data keeps the keypoints"]
    caplog.clear()
    detection({22: 0.0, slice(40, 91): 0.0})
    assert back_view_face_warnings(caplog) == []
    done = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[BCVideoNodes] drawing ")
            and " done in " in r.getMessage()]
    assert len(done) == 1 and done[0].endswith("(back_view_face frames 0)")
