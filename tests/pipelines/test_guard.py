"""The guard on synthetic clips: a clean clip passes, each injected fault fires its own check
on the injected frames only. Every test here goes the way the WanAnimate Preprocess Guard
goes - both groups, then `combine_guards`; tests/pipelines/test_guard_split.py runs each group alone.
Runs without ComfyUI or any model:

    python -m pytest tests/pipelines/test_guard.py
"""
import dataclasses
import json

import numpy as np
import pytest
import torch

from bcvideonodes.pipelines import guard
from guard_fakes import DRAW_THRESHOLD, LEGS, MASK, N, POSE, H, W, clip, drop_keypoints, hand, origin, place_keypoints


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
    masks[20:25, 200:, :] = 0  # the knees, ankles and feet sit at y 225 to 274
    passed, flags, report = run(masks, pose_data)
    assert passed and flags["mask_missing_keypoints"] == list(range(20, 25)), report
    assert "mask_missing_keypoints (warning)" in report and "missed:" in report


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
    assert passed and flags["pose_incomplete"] == list(range(18, 24)), flags
    assert "pose_incomplete (warning)" in report and "missed:" in report and "r_hip-r_knee" in report


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
    assert flags["pose_incomplete"] == list(range(18, 24)), report


def test_a_limb_spike_is_a_warning():
    masks, pose_data = clip()
    move_keypoint(pose_data, 20, 4, dy=-0.2)   # the right wrist jumps a fifth of the frame and comes back
    passed, flags, report = run(masks, pose_data)
    assert passed and flags["pose_spike"] == [20], report
    assert "pose_spike (warning)" in report and "r_wrist" in report


def test_a_spike_is_flagged_over_its_window():
    masks, pose_data = clip()
    for i in (20, 21, 22):                      # out for three frames, back on the fourth
        move_keypoint(pose_data, i, 4, dy=-0.2)
    _, flags, report = run(masks, pose_data)
    assert flags["pose_spike"] == [20, 21, 22], report


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
    drop_keypoints(pose_data, range(5, 35), list(range(8, 20)))
    passed, flags, report = run(masks, pose_data)
    assert passed and flags["body_not_drawn"] == list(range(5, 35)), report
    assert "body_not_drawn (warning)" in report
    assert set(flags.get("pose_incomplete", [])) < set(range(5, 35))


def cut_hand(masks, frames):
    """The mask loses the right hand on `frames`: a 30 x 45 block at the body's side around the
    right wrist, whose forearm reaches it from inside the mask."""
    for i in frames:
        x1, y1 = origin(i)
        masks[i, y1 + 90:y1 + 135, x1:x1 + 30] = 0


def test_a_limb_end_the_mask_lost_is_a_warning():
    masks, pose_data = clip()
    cut_hand(masks, (20, 21))
    passed, flags, report = run(masks, pose_data)
    assert passed and flags["mask_missed_limb"] == [20, 21] and "mask_missing_keypoints" not in flags, report
    assert "mask_missed_limb (warning)" in report


def test_a_spiking_limb_end_is_a_pose_error_not_a_lost_limb():
    masks, pose_data = clip()
    move_keypoint(pose_data, 20, 4, dx=-0.2)   # the right wrist jumps off the body for a frame
    _, flags, report = run(masks, pose_data)
    assert flags["pose_spike"] == [20] and "mask_missed_limb" not in flags, report


def test_a_wrist_whose_hand_folds_back_over_the_forearm_is_a_pose_error_not_a_lost_limb():
    # the wrist keypoint 12 px beside the body (less than a spike's jump): with its hand beyond it
    # the mask lost the hand; with its hand pointing back at the elbow no wrist bends that way,
    # the wrist keypoint is off
    masks, pose_data = clip()
    for i in (20, 21):
        x1, y1 = origin(i)
        place_keypoints(pose_data, [i], 4, x1 - 12, y1 + 112)
        pose_data["pose_metas_original"][i]["keypoints_right_hand"] = hand(x1 - 20, y1 + 126)
    _, flags, report = run(masks, pose_data)
    assert flags["mask_missed_limb"] == [20, 21] and "pose_spike" not in flags, report
    for i in (20, 21):
        x1, y1 = origin(i)
        pose_data["pose_metas_original"][i]["keypoints_right_hand"] = hand(x1 + 2, y1 + 100)
    _, flags, report = run(masks, pose_data)
    assert "mask_missed_limb" not in flags, report


def test_a_limb_end_in_a_notch_is_not_a_lost_limb():
    # a 12 x 12 cut at the mask's edge around the wrist: most of the hand is still in the mask
    masks, pose_data = clip()
    for i in (20, 21):
        x1, y1 = origin(i)
        masks[i, y1 + 106:y1 + 118, x1:x1 + 12] = 0
    _, flags, report = run(masks, pose_data)
    assert "mask_missed_limb" not in flags, report
    _, _, metrics, _ = guard_run(masks, pose_data)
    assert json.loads(metrics)["frames"][20]["missed_keypoints"] == ["r_wrist"]


def test_a_forearm_drawn_along_the_outside_of_the_mask_is_fast_motion_not_a_lost_limb():
    # the elbow on the mask's edge, the wrist 8 px beside it 90 px lower: the whole forearm runs
    # along the outside of the mask, where a mask that cut off the hand would leave the forearm in
    masks, pose_data = clip()
    for i in range(15, 26):
        x1, y1 = origin(i)
        place_keypoints(pose_data, [i], 3, x1 + 1, y1 + 60)
        place_keypoints(pose_data, [i], 4, x1 - 8, y1 + 150)
    _, flags, report = run(masks, pose_data)
    assert "mask_missed_limb" not in flags and "pose_spike" not in flags, report
    _, _, metrics, _ = guard_run(masks, pose_data)
    assert json.loads(metrics)["frames"][20]["missed_keypoints"] == ["r_wrist"]


def test_a_limb_out_of_the_frame_is_no_gap():
    # the right forearm leaves the shot: the pose model places the wrist beyond the left edge
    masks, pose_data = clip()
    for i in range(15, 25):
        place_keypoints(pose_data, [i], 4, -10, origin(i)[1] + 112, conf=0.05)
    _, flags, report = run(masks, pose_data)
    assert "pose_limb_gap" not in flags and "pose_incomplete" not in flags, report


def test_the_frame_edge_reaches_a_limb_width_into_the_frame():
    # a quarter of the 56 px body scale: 10 px from the edge is out of the shot, 20 px is in it
    masks, pose_data = clip()
    for i in range(15, 25):
        place_keypoints(pose_data, [i], 4, 10, origin(i)[1] + 112, conf=0.05)
    _, flags, report = run(masks, pose_data)
    assert "pose_limb_gap" not in flags, report
    for i in range(15, 25):
        place_keypoints(pose_data, [i], 4, 20, origin(i)[1] + 112)
    _, flags, report = run(masks, pose_data)
    assert flags["pose_limb_gap"] == list(range(15, 25)), report


def test_knees_below_the_frame_with_the_thighs_filling_it_are_neither_missing_nor_body_not_drawn():
    masks, pose_data = clip()
    for i in range(10, 21):
        x1, y1 = origin(i)
        masks[i, y1:, x1:x1 + 100] = 1.0             # the body runs off the bottom edge
        for j, x in ((9, 35), (12, 65)):             # the knees below it
            place_keypoints(pose_data, [i], j, x1 + x, H + 5, conf=0.05)
        for j, x in ((10, 35), (13, 65), (18, 70), (19, 30)):
            place_keypoints(pose_data, [i], j, x1 + x, H + 20, conf=0.05)
    passed, flags, report = run(masks, pose_data)
    assert not {"pose_limb_gap", "pose_incomplete", "body_not_drawn"} & set(flags), report


def test_a_limb_hidden_behind_the_body_is_no_gap():
    # the right forearm behind her back: the pose model places the wrist on the torso
    masks, pose_data = clip()
    for i in range(15, 25):
        x1, y1 = origin(i)
        place_keypoints(pose_data, [i], 4, x1 + 45, y1 + 100, conf=0.05)
    _, flags, report = run(masks, pose_data)
    assert "pose_limb_gap" not in flags, report


def test_a_gap_is_the_frames_of_the_stretch_the_limb_is_missing_on():
    # the right forearm is undrawn on 15-24; hidden behind her back on the first frames of that,
    # in the open on the rest: the gap is the rest, when it runs for GAP_MIN (5) frames
    for hidden, flagged in ((range(15, 20), list(range(20, 25))), (range(15, 21), None)):
        masks, pose_data = clip()
        drop_keypoints(pose_data, range(15, 25), [4])
        for i in hidden:
            x1, y1 = origin(i)
            place_keypoints(pose_data, [i], 4, x1 + 45, y1 + 100)
        _, flags, report = run(masks, pose_data)
        assert flags.get("pose_limb_gap") == flagged, report


def test_a_limb_never_drawn_is_no_gap_wherever_the_model_places_it():
    # legs out of the shot for the whole clip: their unseen keypoints wander from the frame edge
    # into the open and back, which is no limb the pose lost
    masks, pose_data = clip()
    drop_keypoints(pose_data, range(N), LEGS)
    for i in range(N):
        y = origin(i)[1] + 185 if 15 <= i < 25 else H - 2
        for j in (9, 10, 12, 13, 18, 19):
            place_keypoints(pose_data, [i], j, origin(i)[0] + 50, y)
    _, flags, report = run(masks, pose_data)
    assert "pose_limb_gap" not in flags and "pose_incomplete" not in flags, report


@pytest.mark.parametrize("r_shoulder, l_shoulder", [((78, 48), (22, 48)), ((45, 48), (55, 48))],
                         ids=["back to the camera", "profile"])
def test_a_person_not_facing_the_camera_misses_no_limb(r_shoulder, l_shoulder):
    masks, pose_data = clip()
    drop_keypoints(pose_data, range(15, 25), [4])
    for i in range(15, 25):
        x1, y1 = origin(i)
        place_keypoints(pose_data, [i], 2, x1 + r_shoulder[0], y1 + r_shoulder[1])
        place_keypoints(pose_data, [i], 5, x1 + l_shoulder[0], y1 + l_shoulder[1])
    _, flags, report = run(masks, pose_data)
    assert "pose_limb_gap" not in flags and "pose_incomplete" not in flags, report


def block(masks, frames, rows, cols):
    """Zero `rows` x `cols` (ranges relative to each frame's rectangle) of the mask on `frames`."""
    for i in frames:
        x1, y1 = origin(i)
        masks[i, y1 + rows[0]:y1 + rows[1], x1 + cols[0]:x1 + cols[1]] = 0


def test_a_region_dropped_for_one_frame_is_mask_loss():
    masks, pose_data = clip()
    block(masks, [20], (150, 180), (40, 70))     # between the knees: no keypoint in it
    passed, flags, report = run(masks, pose_data)
    assert passed and flags == {"mask_loss": [20]}, report
    _, _, metrics, _ = guard_run(masks, pose_data)
    frames = json.loads(metrics)["frames"]
    assert frames[20]["mask_loss"] == pytest.approx(15 / 240)   # the 30 x 30 hole holds a disc of radius 15
    assert frames[0]["mask_loss"] is None and frames[N - 1]["mask_loss"] is None
    assert "mask_loss (warning)" in report


def test_mask_loss_is_the_thickness_not_the_area():
    # a sliver 2 px wide down the whole body is 480 px, a 12 x 12 block 144 px: the block holds a
    # disc of radius 6, the sliver of radius 1
    masks, pose_data = clip()
    block(masks, [12], (0, 240), (98, 100))
    block(masks, [28], (150, 162), (44, 56))
    _, flags, report = run(masks, pose_data)
    assert flags == {"mask_loss": [28]}, report


def test_a_region_lost_for_good_is_not_mask_loss():
    masks, pose_data = clip()
    block(masks, range(20, 31), (150, 180), (40, 70))
    _, flags, report = run(masks, pose_data)
    assert "mask_loss" not in flags, report


def test_the_mask_loss_threshold_is_the_config_s():
    # the 30 x 30 hole holds a disc of radius 15 (0.0625 of the shorter side); the left thigh
    # crosses it, so half the threshold is enough: 0.1 flags it, 0.2 does not
    masks, pose_data = clip()
    block(masks, [20], (150, 180), (40, 70))
    for threshold, flagged in ((0.1, [20]), (0.2, None)):
        loose = guard.MaskGuardConfig(**{**dataclasses.asdict(MASK), "max_mask_loss": threshold})
        _, _, metrics, _ = guard.check_mask(masks, pose_data, loose, stop_on_fail=False)
        record = json.loads(metrics)
        assert record["flags"].get("mask_loss") == flagged and record["thresholds"]["max_mask_loss"] == threshold


def test_a_region_dropped_for_a_few_frames_is_one_mask_loss():
    # held on frame 19 and on frame 25, dropped on 20-24: the region every one of those frames
    # drops is 26 x 30 (the body drifts a pixel a frame), a disc of radius 13
    masks, pose_data = clip()
    block(masks, range(20, 25), (150, 180), (40, 70))
    passed, flags, report = run(masks, pose_data)
    assert passed and flags == {"mask_loss": list(range(20, 25))}, report
    _, _, metrics, _ = guard_run(masks, pose_data)
    frames = json.loads(metrics)["frames"]
    assert [frames[i]["mask_loss_run"] for i in range(19, 26)] == [0.0] + [pytest.approx(13 / 240)] * 5 + [0.0]
    assert all(frames[i]["mask_loss"] == 0.0 for i in range(19, 26))    # no single-frame dropout in it


def test_a_run_of_frames_needs_twice_the_thickness():
    # a 14 x 14 hole beside the skeleton: on one frame it holds a disc of radius 7 (0.029, over
    # max_mask_loss 0.0185); over frames 20-22 the part every frame drops holds one of radius 6
    # (0.025), under twice the threshold
    for frames, flagged in (([20], [20]), (range(20, 23), None)):
        masks, pose_data = clip()
        block(masks, frames, (160, 174), (76, 90))
        _, flags, report = run(masks, pose_data)
        assert flags.get("mask_loss") == flagged, report


def test_a_dropout_longer_than_the_window_is_not_mask_loss():
    masks, pose_data = clip()
    block(masks, range(20, 29), (150, 180), (40, 70))     # nine frames
    _, flags, report = run(masks, pose_data)
    assert "mask_loss" not in flags, report


def arm(masks, frames, rows):
    """A 30 px wide arm held out from the body's right side at `rows` (relative to its top)."""
    for i in frames:
        x1, y1 = origin(i)
        masks[i, y1 + rows[0]:y1 + rows[1], x1 + 100:x1 + 130] = 1.0


def test_a_limb_that_moves_away_and_back_is_not_mask_loss():
    # the arm is lifted 50 px on frames 20-24 and back on frame 25: the spot it left is held on
    # both ends and on none of the frames between, as a dropout would be, but the arm is right
    # there the whole time
    masks, pose_data = clip()
    arm(masks, [*range(0, 20), *range(25, N)], (60, 100))
    arm(masks, range(20, 25), (10, 50))
    _, flags, report = run(masks, pose_data)
    assert "mask_loss" not in flags, report
    masks, pose_data = clip()
    arm(masks, [*range(0, 20), *range(25, N)], (60, 100))   # the same arm gone from the mask instead
    _, flags, report = run(masks, pose_data)
    assert flags["mask_loss"] == list(range(20, 25)), report


def test_a_thin_dropout_counts_where_the_skeleton_crosses_it():
    # a 6 x 6 hole holds a disc of radius 3, 0.0125 of the shorter side: under max_mask_loss
    # (0.0185) but over half of it; on the left thigh the pose sees the body there, beside the
    # body it does not
    for cols, with_pose in (((62, 68), [20]), ((80, 86), None)):
        masks, pose_data = clip()
        block(masks, [20], (160, 166), cols)
        _, flags, report = run(masks, pose_data)
        assert flags.get("mask_loss") == with_pose, report
        _, _, metrics, _ = guard.check_mask(masks, None, MASK, stop_on_fail=False)
        assert "mask_loss" not in json.loads(metrics)["flags"]


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
                                          "min_mask_iou", "max_mask_loss"]
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
