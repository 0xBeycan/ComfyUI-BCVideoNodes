"""The guard on synthetic clips: a clean clip passes, each injected fault fires its own check
on the injected frames only. Every test here goes the way the WanAnimate Preprocess Guard
goes - both groups, then `combine_guards`; tests/pipelines/test_guard_split.py runs each group alone.
Runs without ComfyUI or any model:

    python -m pytest tests/pipelines/test_guard.py
"""
import json

import numpy as np
import pytest
import torch

from bcvideonodes.pipelines import guard
from guard_fakes import DRAW_THRESHOLD, LEGS, MASK, N, POSE, H, W, clip, drop_keypoints


def move_keypoint(pose_data, i, index, dx=0.0, dy=0.0):
    """Shift one body keypoint on frame `i` by (dx, dy) of the frame size."""
    pts = pose_data["pose_metas_original"][i]["keypoints_body"].copy()
    pts[index, 0] += dx
    pts[index, 1] += dy
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


def test_a_missed_detection_is_data_not_a_check():
    masks, pose_data = clip()
    pose_data["detections"][5] = {"bbox": [0.0, 0.0, float(W), float(H)], "score": -1.0, "persons": 0}
    passed, flags, report = run(masks, pose_data)
    # the tracker carries the mask through a frame the detector missed: the guard judges the
    # pose and the mask, and the detector's miss is only in the metrics
    assert passed and not {k for k in flags if k not in guard.WARNINGS}, report
    _, _, metrics, _ = guard_run(masks, pose_data)
    assert json.loads(metrics)["frames"][5]["detected"] is False


def test_a_second_person_is_data_not_a_check():
    # the pipeline draws one person by design; the detector's count is kept as data
    masks, pose_data = clip()
    for i in (12, 13, 14):
        pose_data["detections"][i]["persons"] = 2
    passed, flags, report = run(masks, pose_data)
    assert passed and not flags, report
    _, _, metrics, _ = guard_run(masks, pose_data)
    assert [row["persons"] for row in json.loads(metrics)["frames"][11:15]] == [1, 2, 2, 2]


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


def test_completeness_does_not_move_with_uniformly_lower_scores():
    # a pose model whose scores are uniformly lower is not a broken pose model, as long as
    # its skeleton is still drawn
    masks, pose_data = clip()
    drop_keypoints(pose_data, range(N), list(range(20)), conf=DRAW_THRESHOLD + 0.01)
    passed, flags, report = run(masks, pose_data)
    assert passed and "pose_incomplete" not in flags, report


def test_the_guard_counts_what_is_drawn_not_what_min_keypoint_conf_finds():
    # legs at 0.4 are found at min_keypoint_conf 0.3 but not drawn at 0.5: the diffusion model
    # never sees them, so the frame lost them
    masks, pose_data = clip()
    drop_keypoints(pose_data, range(18, 24), LEGS, conf=0.4)
    passed, flags, report = run(masks, pose_data)
    assert not passed and flags["pose_incomplete"] == list(range(18, 24)), report


def test_a_limb_spike_is_a_pose_fault():
    masks, pose_data = clip()
    move_keypoint(pose_data, 20, 4, dy=-0.2)   # the right wrist jumps a fifth of the frame and comes back
    passed, flags, report = run(masks, pose_data)
    assert not passed and flags["pose_spike"] == [20], report
    assert "r_wrist" in report


def test_a_limb_that_moves_there_and_stays_is_not_a_spike():
    masks, pose_data = clip()
    for i in range(20, N):
        move_keypoint(pose_data, i, 4, dy=-0.1)
    passed, flags, report = run(masks, pose_data)
    assert "pose_spike" not in flags, report


def test_a_limb_missing_for_a_stretch_is_a_warning():
    masks, pose_data = clip()
    drop_keypoints(pose_data, range(15, 25), [4])   # the right forearm is gone for 10 frames
    passed, flags, report = run(masks, pose_data)
    assert passed and flags["pose_limb_gap"] == list(range(15, 25)), report
    assert "pose_limb_gap (warning)" in report and "r_elbow-r_wrist" in report


def test_a_body_the_pose_lost_for_longer_than_the_window_is_body_not_drawn():
    # the lower half of the skeleton is gone for 30 frames: completeness only sees the edges of
    # the stretch, since its middle is expected by neighbours that lost it too; the mask still
    # shows the body there
    masks, pose_data = clip()
    drop_keypoints(pose_data, range(5, 35), list(range(10, 20)))
    passed, flags, report = run(masks, pose_data)
    assert not passed and flags["body_not_drawn"] == list(range(5, 35)), report
    assert set(flags.get("pose_incomplete", [])) < set(range(5, 35))


def test_one_limb_end_outside_the_mask_is_a_warning():
    masks, pose_data = clip()
    move_keypoint(pose_data, 20, 4, dx=0.2)   # the right wrist off the body, alone
    move_keypoint(pose_data, 21, 4, dx=0.2)
    passed, flags, report = run(masks, pose_data)
    assert flags["mask_missed_limb"] == [20, 21] and "mask_missing_keypoints" not in flags, report
    assert "mask_missed_limb (warning)" in report


def test_background_attached_to_the_body_on_one_frame_is_a_warning():
    masks, pose_data = clip()
    x2 = 60 + 20 + 100
    masks[20, 200:280, x2:x2 + 40] = 1.0   # a patch of background joined to her side
    passed, flags, report = run(masks, pose_data)
    assert passed and flags["mask_attached_leak"] == [20], report


def test_a_hand_holds_its_piece_of_mask():
    # the hand the text banner cut off the arm: no body keypoint in it, but her hand keypoints
    masks, pose_data = clip()
    masks[30, 40:, 90:190] = 1.0
    masks[30, 0:20, W - 30:] = 1.0
    passed, flags, _ = run(masks, pose_data)
    assert 30 in flags.get("mask_fragmented", []) + flags.get("mask_specks", [])
    hand = np.tile([(W - 15) / W, 10 / H, 0.9], (21, 1))
    pose_data["pose_metas_original"][30]["keypoints_right_hand"] = hand
    passed, flags, report = run(masks, pose_data)
    assert 30 not in flags.get("mask_fragmented", []) + flags.get("mask_specks", []), report


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
    move_keypoint(pose_data, 3, 4, dy=-0.2)            # pose_spike also on 3, a pose check
    _, flags, report = run(masks, pose_data)
    names = list(flags)
    # on frame 3 the pose checks come first, as a single guard over both would test them
    assert names.index("pose_spike") < names.index("mask_empty") < names.index("subject_switch"), names


def test_combined_rows_hold_every_measurement_in_one_order():
    masks, pose_data = clip()
    _, _, metrics, _ = guard_run(masks, pose_data)
    record = json.loads(metrics)
    assert list(record) == ["thresholds", "enabled", "flags", "frames"]
    assert list(record["thresholds"]) == ["draw_threshold", "min_pose_completeness", "max_torso_jump",
                                          "max_limb_spike", "min_mask_to_box", "max_mask_outside_box",
                                          "max_attached_leak", "min_keypoint_recall", "max_body_not_drawn",
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
                 "pose_config": pose_data["pose_config"], "draw_threshold": DRAW_THRESHOLD}
    passed, flags, _ = run(masks[0], pose_data)
    assert passed
