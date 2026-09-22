"""The guard on synthetic clips: a clean clip passes, each injected fault fires its own check
on the injected frames only. Every test here goes the way the WanAnimate Preprocess Guard
goes - both groups, then `combine_guards`; tests/test_guard_split.py runs each group alone.
Runs without ComfyUI or any model:

    python -m pytest tests/test_guard.py
"""
import json

import numpy as np
import pytest
import torch

from preprocess import guard

N, H, W = 40, 320, 240
POSE_CONFIG = {"min_keypoint_conf": 0.3}   # the part of Pose Detection's config the guards read
POSE = guard.PoseGuardConfig(min_pose_completeness=0.6, max_torso_jump=0.25)
MASK = guard.MaskGuardConfig(min_mask_to_box=0.15, max_mask_outside_box=0.10,
                             min_keypoint_recall=0.9, min_mask_iou=0.6)


def clip():
    """A person-sized rectangle drifting slowly across the frame, with keypoints inside it."""
    masks = torch.zeros(N, H, W)
    metas, detections = [], []
    for i in range(N):
        x1, y1 = 60 + i, 40
        x2, y2 = x1 + 100, y1 + 240
        masks[i, y1:y2, x1:x2] = 1.0
        # 20 body keypoints spread inside the rectangle, all confident
        xs = np.linspace(x1 + 10, x2 - 10, 5)
        ys = np.linspace(y1 + 10, y2 - 10, 4)
        pts = np.array([(x, y, 0.9) for y in ys for x in xs])
        pts[:, 0] /= W
        pts[:, 1] /= H
        metas.append({"width": W, "height": H, "keypoints_body": pts})
        detections.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "score": 0.95, "persons": 1})
    return masks, {"pose_metas_original": metas, "detections": detections, "pose_config": dict(POSE_CONFIG)}


def drop_keypoints(pose_data, frames, indices):
    """Make the named body keypoints unconfident on `frames`, as a pose model losing them."""
    for i in frames:
        pts = pose_data["pose_metas_original"][i]["keypoints_body"].copy()
        pts[indices, 2] = 0.05
        pose_data["pose_metas_original"][i]["keypoints_body"] = pts


def guard_run(masks, pose_data, pose_guard=True, mask_guard=True):
    """Both groups and the combined result, without stopping: (report, passed, metrics, timeline)."""
    _, _, pose_metrics, _ = guard.check_pose(pose_data, POSE, pose_guard, stop_on_fail=False)
    _, _, mask_metrics, _ = guard.check_mask(masks, pose_data, MASK, mask_guard, stop_on_fail=False)
    report, metrics, timeline = guard.combine_guards(pose_metrics, mask_metrics, stop_on_fail=False)
    return report, report.startswith("Preprocess guard: passed"), metrics, timeline


def run(masks, pose_data, pose_guard=True, mask_guard=True):
    report, passed, metrics, timeline = guard_run(masks, pose_data, pose_guard, mask_guard)
    flags = json.loads(metrics)["flags"]
    assert timeline.shape[0] == 1 and timeline.shape[-1] == 3
    return passed, flags, report


def fails_only(flags, name, frames):
    failing = {k: v for k, v in flags.items() if k not in guard.WARNINGS}
    assert list(failing) == [name] or set(failing) - {name} <= {"mask_unstable"}, failing
    assert failing[name] == frames


def test_clean_clip_passes():
    masks, pose_data = clip()
    passed, flags, report = run(masks, pose_data)
    assert passed and not {k for k in flags if k not in guard.WARNINGS}, report


def test_the_beta_label_is_off():
    masks, pose_data = clip()
    _, _, report = run(masks, pose_data)
    assert report.startswith("Preprocess guard: ") and "beta" not in report


def test_empty_start_is_mask_empty():
    masks, pose_data = clip()
    masks[:10] = 0
    passed, flags, _ = run(masks, pose_data)
    assert not passed
    fails_only(flags, "mask_empty", list(range(10)))


def test_empty_mask_on_an_undetected_frame_is_not_a_mask_failure():
    masks, pose_data = clip()
    masks[7] = 0
    pose_data["detections"][7] = {"bbox": [0.0, 0.0, float(W), float(H)], "score": -1.0, "persons": 0}
    pts = pose_data["pose_metas_original"][7]["keypoints_body"].copy()
    pts[:, 2] = 0.05
    pose_data["pose_metas_original"][7]["keypoints_body"] = pts
    report, passed, metrics, _ = guard_run(masks, pose_data, False, True)
    flags = json.loads(metrics)["flags"]
    assert passed and "mask_empty" not in flags, report


def test_mask_dying_mid_clip():
    masks, pose_data = clip()
    masks[25:] = 0
    passed, flags, _ = run(masks, pose_data)
    assert not passed and flags["mask_empty"] == list(range(25, N))


def test_cut_limb_names_the_keypoints():
    masks, pose_data = clip()
    masks[20:25, 200:, :] = 0  # the lowest keypoint row sits at y ~ 270
    passed, flags, report = run(masks, pose_data)
    assert not passed
    assert flags["mask_missing_keypoints"] == list(range(20, 25))
    assert "missed:" in report


def test_a_split_off_piece_holding_keypoints_is_not_a_second_object():
    masks, pose_data = clip()
    masks[30, 200:210, :] = 0     # a gap across the body; the lower piece keeps its keypoints
    passed, flags, report = run(masks, pose_data)
    assert "mask_fragmented" not in flags and "mask_specks" not in flags, report


def test_a_hand_the_frame_edge_cut_off_is_not_a_second_object():
    masks, pose_data = clip()
    masks[30, 40:, 90:190] = 1.0      # the body runs off the bottom edge
    masks[30, 0:20, W - 30:] = 1.0    # her hand comes back in at the top right corner
    pts = pose_data["pose_metas_original"][30]["keypoints_body"].copy()
    pts[4] = ((W - 15) / W, 10 / H, 0.9)   # the wrist is in that corner, and confident
    pose_data["pose_metas_original"][30]["keypoints_body"] = pts
    passed, flags, report = run(masks, pose_data)
    assert "mask_fragmented" not in flags and "mask_specks" not in flags, report


def test_an_object_at_the_frame_edge_is_still_a_second_object():
    # the body running off an edge must not excuse everything else that touches one: this is
    # the annexed-object case, and the guard has to keep reporting it
    masks, pose_data = clip()
    masks[30, 40:, 90:190] = 1.0      # the body runs off the bottom edge
    masks[30, 170:250, 0:30] = 1.0    # an object beside her, against the left edge
    passed, flags, report = run(masks, pose_data)
    assert not passed and 30 in flags["mask_fragmented"], report


def test_leak_outside_box():
    masks, pose_data = clip()
    masks[30, :, 200:] = 1.0  # a stripe far from the person
    passed, flags, _ = run(masks, pose_data)
    assert not passed and 30 in flags["mask_leak"] and 30 in flags["mask_fragmented"]


def test_missing_detection_is_only_a_warning():
    masks, pose_data = clip()
    pose_data["detections"][5] = {"bbox": [0.0, 0.0, float(W), float(H)], "score": -1.0, "persons": 0}
    passed, flags, report = run(masks, pose_data)
    # the tracker carries the mask through a frame the detector missed, so it is reported
    # and measured but stops nothing
    assert flags["no_detection"] == [5] and passed, report
    assert "no_detection (warning)" in report


def test_second_person_is_only_a_warning():
    masks, pose_data = clip()
    for i in (12, 13, 14):
        pose_data["detections"][i]["persons"] = 2
    passed, flags, report = run(masks, pose_data)
    assert flags["multi_person"] == [12, 13, 14] and passed, report
    assert "multi_person (warning)" in report


def test_subject_switch():
    masks, pose_data = clip()
    pose_data["detections"][15]["bbox"] = [0.0, 0.0, 40.0, 40.0]
    passed, flags, _ = run(masks, pose_data)
    assert not passed and 15 in flags["subject_switch"]


def test_torso_jump():
    masks, pose_data = clip()
    pts = pose_data["pose_metas_original"][12]["keypoints_body"].copy()
    pts[guard.TORSO, 0] += 0.5
    pose_data["pose_metas_original"][12]["keypoints_body"] = pts
    passed, flags, _ = run(masks, pose_data)
    assert not passed and 12 in flags["pose_jump"]


LEGS = [8, 9, 10, 11, 12, 13, 18, 19]   # the keypoints of both legs and both feet


def test_collapsed_skeleton_is_pose_incomplete():
    masks, pose_data = clip()
    drop_keypoints(pose_data, range(18, 24), LEGS)
    passed, flags, report = run(masks, pose_data)
    assert not passed
    assert flags["pose_incomplete"] == list(range(18, 24)), flags
    assert "missed:" in report and "r_hip-r_knee" in report


def test_a_pose_the_whole_clip_lacks_is_not_incomplete():
    # the legs are out of shot for the entire clip: every frame is as complete as its
    # neighbours, which is what completeness means - it is not a defect
    masks, pose_data = clip()
    drop_keypoints(pose_data, range(N), LEGS)
    passed, flags, report = run(masks, pose_data)
    assert passed and "pose_incomplete" not in flags, report


def test_completeness_does_not_move_with_the_confidence_scale():
    # a pose model whose scores are uniformly lower is not a broken pose model
    masks, pose_data = clip()
    for i in range(N):
        pts = pose_data["pose_metas_original"][i]["keypoints_body"].copy()
        pts[:, 2] = 0.31
        pose_data["pose_metas_original"][i]["keypoints_body"] = pts
    passed, flags, report = run(masks, pose_data)
    assert passed and "pose_incomplete" not in flags, report


def test_guards_off_never_raise_but_still_report():
    masks, pose_data = clip()
    masks[:10] = 0
    passed, flags, report = run(masks, pose_data, pose_guard=False, mask_guard=False)
    assert passed and flags["mask_empty"] == list(range(10)) and "(off)" in report


def test_a_failed_check_stops_the_combined_run():
    masks, pose_data = clip()
    masks[:10] = 0
    _, _, pose_metrics, _ = guard.check_pose(pose_data, POSE, stop_on_fail=False)
    _, _, mask_metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    with pytest.raises(guard.GuardFailed, match="Preprocess guard: FAILED"):
        guard.combine_guards(pose_metrics, mask_metrics)


def test_flags_are_listed_in_the_order_they_first_fired_across_both_groups():
    masks, pose_data = clip()
    masks[3] = 0                                       # mask_empty first, on frame 3
    pose_data["detections"][15]["bbox"] = [0.0, 0.0, 40.0, 40.0]   # subject_switch on 15
    for i in (3, 20):
        pose_data["detections"][i]["persons"] = 2      # multi_person also on 3, a pose check
    _, flags, report = run(masks, pose_data)
    names = list(flags)
    # on frame 3 the pose checks come first, as a single guard over both would test them
    assert names.index("multi_person") < names.index("mask_empty") < names.index("subject_switch"), names


def test_combined_rows_hold_every_measurement_in_one_order():
    masks, pose_data = clip()
    _, _, metrics, _ = guard_run(masks, pose_data)
    record = json.loads(metrics)
    assert list(record) == ["thresholds", "enabled", "flags", "frames"]
    assert list(record["thresholds"]) == ["min_keypoint_conf", "min_pose_completeness", "max_torso_jump",
                                          "min_mask_to_box", "max_mask_outside_box", "min_keypoint_recall",
                                          "min_mask_iou"]
    assert all(tuple(row) == guard.PREPROCESS_ROW for row in record["frames"])


def test_frame_count_mismatch_is_an_error():
    masks, pose_data = clip()
    with pytest.raises(ValueError):
        guard.check_mask(masks[:5], pose_data, MASK)


def test_foreign_pose_data_is_an_error():
    masks, _ = clip()
    with pytest.raises(ValueError, match="Pose Detection"):
        guard.check_mask(masks, {"something": 1}, MASK)
    with pytest.raises(ValueError, match="Pose Detection"):
        guard.check_pose({"something": 1}, POSE)


def test_resized_mask_is_an_error():
    masks, pose_data = clip()
    small = torch.nn.functional.interpolate(masks[None], size=(H // 2, W // 2))[0]
    with pytest.raises(ValueError, match="before any resize"):
        guard.check_mask(small, pose_data, MASK)


def test_single_frame_2d_mask_is_accepted():
    masks, pose_data = clip()
    pose_data = {"pose_metas_original": pose_data["pose_metas_original"][:1], "detections": pose_data["detections"][:1],
                 "pose_config": pose_data["pose_config"]}
    passed, flags, _ = run(masks[0], pose_data)
    assert passed
