"""The guard on synthetic clips: a clean clip passes, each injected fault fires its own check
on the injected frames only. The tests run both groups, then `combine_guards`, on the raw mask;
the WanAnimate Preprocess Guard does the same on the final mask (`final_record` measures a mask
as the final); tests/pipelines/test_guard_split.py runs each group alone.
Runs without ComfyUI or any model:

    python -m pytest tests/pipelines/test_guard.py
"""
import dataclasses
import json

import numpy as np
import pytest
import torch

from bcvideonodes.pipelines import guard
from guard_fakes import (DRAW_THRESHOLD, LEGS, MASK, N, POSE, H, W, arm, clip, drop_keypoints, hand, origin,
                         place_keypoints)


def move_keypoint(pose_data, i, index, dx=0.0, dy=0.0):
    """Shift one body keypoint on frame `i` by (dx, dy) of the frame size."""
    pts = pose_data["pose_metas_original"][i]["keypoints_body"].copy()
    pts[index, 0] += dx
    pts[index, 1] += dy
    pose_data["pose_metas_original"][i]["keypoints_body"] = pts


def guard_run(masks, pose_data, mask_guard=True):
    """Both groups and the combined result, without stopping: (report, passed, metrics, timeline)."""
    _, _, pose_metrics, _ = guard.check_pose(pose_data, POSE, stop_on_fail=False)
    _, _, mask_metrics, _ = guard.check_mask(masks, pose_data, MASK, mask_guard, stop_on_fail=False)
    report, metrics, timeline = guard.combine_guards(pose_metrics, mask_metrics, stop_on_fail=False)
    return report, report.startswith("Preprocess guard: passed"), metrics, timeline


def run(masks, pose_data, mask_guard=True):
    report, passed, metrics, timeline = guard_run(masks, pose_data, mask_guard)
    flags = json.loads(metrics)["flags"]
    assert timeline.shape[0] == 1 and timeline.shape[-1] == 3
    return passed, flags, report


def fails_only(flags, name, frames):
    failing = {k: v for k, v in flags.items() if k not in guard.WARNINGS}
    assert list(failing) == [name], failing
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
    report, passed, metrics, _ = guard_run(masks, pose_data)
    flags = json.loads(metrics)["flags"]
    assert passed and "mask_empty" not in flags, report


def test_mask_dying_mid_clip():
    masks, pose_data = clip()
    masks[25:] = 0
    passed, flags, _ = run(masks, pose_data)
    assert not passed and flags["mask_empty"] == list(range(25, N))


def test_lower_legs_outside_the_mask_fail_and_name_the_limbs():
    # the mask stops 20 px above the knees on frames 20-24 (her rectangle's rows 0-165): each knee,
    # ankle and foot lies outside it, 20 px or more from it (0.07 of the 240 px shorter side is
    # 16.8), the whole lower leg - prompt_pose's rule for a limb the mask lost
    masks, pose_data = clip()
    for i in range(20, 25):
        x1, y1 = origin(i)
        masks[i, y1 + 165:, :] = 0
    passed, flags, report = run(masks, pose_data)
    assert not passed and flags["mask_limb_out"] == list(range(20, 25)), report
    assert "mask_limb_out (fail)" in report and "right lower leg x5" in report and "left lower leg x5" in report


def test_a_foot_past_the_mask_is_no_limb_out():
    # only the feet (8 px below the ankles) are cut off: one keypoint of each lower leg outside,
    # a foot tip past the edge, not a whole limb (3 keypoints)
    masks, pose_data = clip()
    for i in range(20, 25):
        x1, y1 = origin(i)
        masks[i, y1 + 228:, :] = 0
    passed, flags, report = run(masks, pose_data)
    assert passed and "mask_limb_out" not in flags, report


def test_a_forearm_and_hand_outside_the_mask_is_limb_out():
    # the right forearm held out beside her (elbow 30 px left of her side, wrist 30 px further,
    # a drawn hand around it) with none of it in the mask: a whole forearm-and-hand lost
    masks, pose_data = clip()
    for i in range(10, 15):
        x1, y1 = origin(i)
        place_keypoints(pose_data, [i], 3, x1 - 30, y1 + 60)
        place_keypoints(pose_data, [i], 4, x1 - 55, y1 + 60)
        pose_data["pose_metas_original"][i]["keypoints_right_hand"] = hand(x1 - 58, y1 + 58)
    passed, flags, report = run(masks, pose_data)
    assert not passed and flags["mask_limb_out"] == list(range(10, 15)), report
    assert "right forearm and hand" in report


def test_a_split_off_piece_holding_keypoints_is_not_a_second_object():
    masks, pose_data = clip()
    masks[30, 200:210, :] = 0     # a gap across the body; the lower piece keeps its keypoints
    passed, flags, report = run(masks, pose_data)
    assert "mask_fragmented" not in flags, report


def test_a_hand_the_frame_edge_cut_off_is_not_a_second_object():
    masks, pose_data = clip()
    masks[30, 40:, 90:190] = 1.0      # the body runs off the bottom edge
    masks[30, 0:20, W - 30:] = 1.0    # her hand comes back in at the top right corner
    pts = pose_data["pose_metas_original"][30]["keypoints_body"].copy()
    pts[4] = ((W - 15) / W, 10 / H, 0.9)   # the wrist is in that corner, and confident
    pose_data["pose_metas_original"][30]["keypoints_body"] = pts
    passed, flags, report = run(masks, pose_data)
    assert "mask_fragmented" not in flags, report


def test_an_object_at_the_frame_edge_is_still_a_second_object():
    # the body running off an edge must not excuse everything else that touches one: this is
    # the annexed-object case, and the guard has to keep reporting it
    masks, pose_data = clip()
    masks[30, 40:, 90:190] = 1.0      # the body runs off the bottom edge
    masks[30, 170:250, 0:30] = 1.0    # an object beside her, against the left edge
    passed, flags, report = run(masks, pose_data)
    assert not passed and 30 in flags["mask_fragmented"], report


def pose_free_record(masks):
    """The Mask Guard's metrics record on `masks` without pose_data."""
    return json.loads(guard.check_mask(masks, None, MASK, stop_on_fail=False)[2])


def test_a_speck_is_data_not_a_check():
    # a 30 px arm held out on frame 20 only, a 4 px cut at her side splitting it off: a 26 x 30 px
    # piece, 3.25% of her rectangle - measured, never flagged. The same piece held out on every
    # frame is hers (her part holds it on the frames around) and is not even measured
    masks, _ = clip()
    arm(masks, [20], (60, 90), 30)
    block(masks, [20], (60, 90), (100, 104))
    record = pose_free_record(masks)
    assert record["flags"] == {} and record["frames"][20]["fragments"] == [round(26 * 30 / 24000, 4)]
    masks, _ = clip()
    arm(masks, range(N), (60, 90), 30)
    block(masks, [20], (60, 90), (100, 104))
    assert pose_free_record(masks)["frames"][20]["fragments"] == []


def test_a_big_piece_of_her_the_mask_tore_off_is_still_a_second_object():
    # the arm of the first test 60 px tall: the 26 x 60 px piece is 6.5% of her rectangle, the
    # mask torn in two however much of it her part holds on the frames around
    masks, _ = clip()
    arm(masks, range(N), (60, 120), 30)
    block(masks, [20], (60, 120), (100, 104))
    assert pose_free_flags(masks) == {"mask_fragmented": [20]}


def head_off(masks, frames):
    """The mask leaves out her head (the top 30 px of her rectangle: nose, eyes, ears) on `frames`."""
    for i in frames:
        x1, y1 = origin(i)
        masks[i, y1:y1 + 30, x1:x1 + 100] = 0


def test_the_head_outside_the_mask_fails():
    # from frame 20 on the mask leaves out her head for good (no dropout: it never comes back)
    masks, pose_data = clip()
    head_off(masks, range(20, N))
    passed, flags, report = run(masks, pose_data)
    assert not passed and flags == {"mask_head_out": list(range(20, N))}, report
    assert "mask_head_out (fail)" in report and "missed: nose x20" in report


def test_the_nose_or_two_of_the_eyes_and_ears_outside_the_mask_are_the_head_out():
    # keypoints drawn 15 px left of her rectangle from frame 20 on, the mask as it is. One ear alone
    # is the mask's outline at the hair, not the head, and raises nothing
    for out, flags in (([0], {"mask_head_out": list(range(20, N))}), ([14, 16], {"mask_head_out": list(range(20, N))}),
                       ([16], {})):
        masks, pose_data = clip()
        for i in range(20, N):
            x1, y1 = origin(i)
            for j in out:
                place_keypoints(pose_data, [i], j, x1 - 15, y1 + 18)
        passed, got, report = run(masks, pose_data)
        assert got == flags and passed == ("mask_head_out" not in flags), (out, report)


def test_how_many_eyes_and_ears_outside_the_mask_fail_is_the_config_s():
    # keypoints drawn 15 px left of her rectangle from frame 20 on, the mask as it is: one ear alone
    # fails at head_out_eyes_ears 1; an eye and an ear fail at the default 2 and at 3 are nothing;
    # the nose fails at any count, 4 included
    head_out = {"mask_head_out": list(range(20, N))}
    for out, count, flags in (([16], 1, head_out), ([16], 2, {}), ([14, 16], 2, head_out), ([14, 16], 3, {}),
                              ([0], 4, head_out)):
        masks, pose_data = clip()
        for i in range(20, N):
            x1, y1 = origin(i)
            for j in out:
                place_keypoints(pose_data, [i], j, x1 - 15, y1 + 18)
        config = guard.MaskGuardConfig(**{**dataclasses.asdict(MASK), "head_out_eyes_ears": count})
        record = json.loads(guard.check_mask(masks, pose_data, config, stop_on_fail=False)[2])
        assert record["flags"] == flags and record["thresholds"]["head_out_eyes_ears"] == count, (out, count, record)


def test_a_face_drawn_on_the_back_of_her_head_is_inside_the_mask():
    # her back to the camera: the pose model mirrors her, right side on the image's right, and still
    # draws a face - on the back of her head, which the mask covers. Only a mask that leaves the
    # head out fails
    masks, pose_data = clip()
    for i in range(N):
        pts = pose_data["pose_metas_original"][i]["keypoints_body"].copy()
        x1, _ = origin(i)
        pts[:, 0] = (2 * x1 + 100) / W - pts[:, 0]
        pose_data["pose_metas_original"][i]["keypoints_body"] = pts
    passed, flags, report = run(masks, pose_data)
    assert passed and flags == {}, report
    head_off(masks, range(20, N))
    passed, flags, report = run(masks, pose_data)
    assert not passed and flags == {"mask_head_out": list(range(20, N))}, report


def test_leak_outside_box():
    # a stripe far from the person: the leak is a warning, the torn-off piece fails
    masks, pose_data = clip()
    masks[30, :, 200:] = 1.0
    passed, flags, report = run(masks, pose_data)
    assert not passed and 30 in flags["mask_leak"] and 30 in flags["mask_fragmented"]
    assert "mask_leak (warning)" in report and "mask_fragmented (fail)" in report


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


def test_subject_switch_is_a_warning():
    masks, pose_data = clip()
    pose_data["detections"][15]["bbox"] = [0.0, 0.0, 40.0, 40.0]
    passed, flags, report = run(masks, pose_data)
    assert passed and 15 in flags["subject_switch"] and "subject_switch (warning)" in report, report


def test_torso_jump_is_a_warning():
    masks, pose_data = clip()
    pts = pose_data["pose_metas_original"][12]["keypoints_body"].copy()
    pts[guard.TORSO, 0] += 0.5
    pose_data["pose_metas_original"][12]["keypoints_body"] = pts
    passed, flags, report = run(masks, pose_data)
    assert passed and 12 in flags["pose_jump"] and "pose_jump (warning)" in report, report
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


def block(masks, frames, rows, cols):
    """Zero `rows` x `cols` (ranges relative to each frame's rectangle) of the mask on `frames`."""
    for i in frames:
        x1, y1 = origin(i)
        masks[i, y1 + rows[0]:y1 + rows[1], x1 + cols[0]:x1 + cols[1]] = 0


def gone_on(*frames):
    """Every frame but `frames`: the arm is held out on those and gone from the others' mask."""
    return [i for i in range(N) if i not in frames]


# An 84 x 84 hole in her legs (rows, columns of her rectangle): the grow leaves its 64 x 64 core.
# BlockifyMask cuts the grown box (rows 30-289, columns x1 - 10 to x1 + 110) into 8 x 3 blocks of 32 x
# 40 px; on frame 20 (x1 = 80) the block at rows 222-253, columns 110-149 lies inside the core, holds
# no grown pixel and stays off: 1280 px of her 24000 px rectangle, 5.3%, and both her shins cross it.
HOLE = ((150, 234), (8, 92))


def legs_aside(pose_data, frames):
    """Her hips, knees, ankles and feet drawn down her rectangle's right edge on `frames`: no limb
    of hers crosses HOLE."""
    for i in frames:
        x1, y1 = origin(i)
        for j, y in ((8, 140), (11, 140), (9, 185), (12, 185), (10, 222), (13, 222), (18, 234), (19, 234)):
            place_keypoints(pose_data, [i], j, x1 + 97, y1 + y)


def test_a_hand_sized_part_of_her_the_final_mask_leaves_out_is_mask_loss():
    masks, pose_data = clip()
    block(masks, [20], *HOLE)
    passed, flags, report = run(masks, pose_data)
    assert not passed and flags == {"mask_loss": [20]}, report
    assert "mask_loss (fail)" in report
    frames = json.loads(guard_run(masks, pose_data)[2])["frames"]
    assert frames[20]["mask_loss"] == pytest.approx(1280 / 24000) and frames[19]["mask_loss"] == 0.0
    assert pose_free_flags(masks) == {"mask_loss": [20]}


def pose_free_flags(masks):
    """The Mask Guard's flags on `masks` without pose_data."""
    return json.loads(guard.check_mask(masks, None, MASK, stop_on_fail=False)[2])["flags"]


def test_a_hole_the_final_mask_fills_is_not_mask_loss():
    # a 30 x 30 hole between the knees on one frame: the grow fills all but its 10 x 10 core, and
    # every block of the final's grid over that core holds grown mask, so the models never see it
    masks, pose_data = clip()
    block(masks, [20], (150, 180), (40, 70))
    _, flags, report = run(masks, pose_data)
    assert "mask_loss" not in flags, report
    assert json.loads(guard_run(masks, pose_data)[2])["frames"][20]["mask_loss"] == 0.0
    assert "mask_loss" not in pose_free_flags(masks)


def test_a_region_the_final_leaves_out_without_a_whole_block_is_not_mask_loss():
    # a 60 x 84 hole: the grow leaves a 40 x 64 core (rows 110-149 of the frame), which no 32 px
    # row of blocks (94-125, 126-157) fits in: the final's blocks there all hold grown mask
    masks, pose_data = clip()
    block(masks, [20], (60, 120), (8, 92))
    assert "mask_loss" not in run(masks, pose_data)[1]
    assert "mask_loss" not in pose_free_flags(masks)


def test_only_a_hand_sized_loss_counts():
    # a 100 x 100 person read on a grid of 4 px cells: a 12 x 12 region dropped on frame 5 (1.44% of
    # her, under LOSS_HAND) is nothing, a 14 x 14 one (1.96%) is a loss
    grid = (np.arange(0, 121, 4), np.arange(0, 121, 4))
    for side, share in ((12, 0.0), (14, 196 / 10000)):
        masks = np.zeros((10, 120, 120), bool)
        masks[:, 10:110, 10:110] = True
        masks[5, 20:20 + side, 20:20 + side] = False
        loss = guard.mask.dropouts(masks, lambda f: (None, grid))
        assert loss[5] == pytest.approx(share) and loss[4] == loss[6] == 0.0, side


def arm_clip(frames, place=None, N=12):
    """A 100 x 80 person with a 20 x 20 hand held out at her right on `frames` (at column `place`,
    beside her, when given), read on a grid of 4 px cells; returns (masks, dropouts)."""
    masks = np.zeros((N, 160, 160), bool)
    masks[:, 20:120, 20:100] = True
    for i in frames:
        masks[i, 50:70, 100:120] = True
    for i, x in (place or {}).items():
        masks[i, 50:70, x:x + 20] = True
    grid = (np.arange(0, 161, 4), np.arange(0, 161, 4))
    return masks, guard.mask.dropouts(masks, lambda f: (None, grid))


def test_a_hand_that_moves_away_and_back_is_not_mask_loss():
    # the hand is 22 px further out on frame 5 and back on frame 6: the spot it left is held on
    # both ends and on none of the frames between, as a dropout would be, but the hand is right
    # there (a third of its area within reach). Gone from frame 5 it is 400 px of her 8400: a loss
    _, loss = arm_clip([i for i in range(12) if i != 5], {5: 122})
    assert loss[5] == 0.0
    _, loss = arm_clip([i for i in range(12) if i != 5])
    assert loss[5] == pytest.approx(400 / 8400)


def test_a_leak_that_blinks_off_is_not_mask_loss():
    # the hand held on frames 4 and 6 only: the region frame 5 drops is held next to it and nowhere
    # beyond, background blinking off. Held on frames 2-4 and 6-8 it is a part of her the mask dropped
    assert arm_clip([4, 6])[1][5] == 0.0
    assert arm_clip([2, 3, 4, 6, 7, 8])[1][5] == pytest.approx(400 / 8400)


def test_a_dropout_longer_than_the_window_is_not_mask_loss_without_a_pose():
    # nine frames gone between two that hold it: longer than LOSS_WINDOW (8); eight frames are one loss
    assert arm_clip([0, 10, 11], N=12)[1] == [0.0] * 12
    assert arm_clip([0, 1, 10, 11], N=12)[1][2:10] == [pytest.approx(400 / 8400)] * 8


def test_with_a_pose_the_body_has_to_be_in_the_region():
    # the same hole with her legs drawn down her right edge: the pose puts no limb in it on frame
    # 20, so it is the background between her legs, not her; without pose_data it is a loss
    masks, pose_data = clip()
    block(masks, [20], *HOLE)
    legs_aside(pose_data, range(N))
    assert "mask_loss" not in run(masks, pose_data)[1]
    assert pose_free_flags(masks) == {"mask_loss": [20]}


def test_a_limb_the_pose_loses_with_the_mask_still_counts():
    # the pose loses her legs on frame 20 too (the shins that cross the hole on frames 19 and 21 are
    # not drawn on it, and they are in the shot): the pose cannot say where they went, so the frames
    # around it stand
    masks, pose_data = clip()
    block(masks, [20], *HOLE)
    drop_keypoints(pose_data, [20], LEGS)
    assert run(masks, pose_data)[1] == {"mask_loss": [20]}


def test_a_loss_from_the_clip_s_start_or_to_its_end_counts_with_a_pose():
    # the hole on the last frame (no frame after it to hold the region again) or the first: an
    # open run, judged with the pose only - her shins cross it; drawn aside, nothing
    for frame in (N - 1, 0):
        masks, pose_data = clip()
        block(masks, [frame], *HOLE)
        assert run(masks, pose_data)[1] == {"mask_loss": [frame]}, frame
        assert pose_free_flags(masks) == {}
        legs_aside(pose_data, range(N))
        assert "mask_loss" not in run(masks, pose_data)[1]


# --- the final mask: the WanAnimate Preprocess Guard's, grown and blockified -------------------

def final_record(masks, pose_data, final=True):
    """The mask group's metrics record on `masks` as the final mask (or, with `final` False, as a
    raw one)."""
    return json.loads(guard.check_mask(masks, pose_data, MASK, stop_on_fail=False, final=final)[2])


def test_on_the_final_mask_a_block_the_grid_moves_is_no_loss():
    # her left forearm reaches out of the body to the wrist 150 px right of her rectangle's left
    # edge; the final mask's moving block grid turns a 32 px block above it on and off (on frames
    # 18-19 and 21-22, off on 20). No keypoint lies in it: the grid, not a lost part
    masks, pose_data = clip()
    for i in range(N):
        x1, y1 = origin(i)
        masks[i, y1 + 100:y1 + 124, x1 + 100:x1 + 160] = 1.0
        place_keypoints(pose_data, [i], 7, x1 + 150, y1 + 112)
    for i in (18, 19, 21, 22):
        x1, y1 = origin(i)
        masks[i, y1 + 68:y1 + 100, x1 + 110:x1 + 142] = 1.0
    assert "mask_loss" not in final_record(masks, pose_data)["flags"]


def test_on_the_final_mask_a_hand_it_drops_is_mask_loss():
    # a 60 x 60 hand held out at her right, the wrist keypoint in it, dropped from frame 20's final:
    # it holds the keypoint on the frame and whole blocks of the grid, 4800 px of her 37400 on frame 19
    masks, pose_data = clip()
    for i in range(N):
        x1, y1 = origin(i)
        masks[i, y1 + 80:y1 + 140, x1 + 100:x1 + 100 + min(60, W - x1 - 100)] = 1.0
        place_keypoints(pose_data, [i], 7, x1 + 130, y1 + 110)
    x1, y1 = origin(20)
    masks[20, y1 + 80:y1 + 140, x1 + 100:] = 0
    record = final_record(masks, pose_data)
    assert record["flags"] == {"mask_loss": [20]}, record["flags"]
    assert record["frames"][20]["mask_loss"] == pytest.approx(0.1283, abs=1e-4)


def test_on_the_final_mask_a_block_the_grid_adds_on_one_frame_is_no_attached_leak():
    # the final's outline moves by up to a block from frame to frame with no change in the raw
    # mask: a 32 px column beside her on one frame is that, 60 px is background taken in
    masks, pose_data = clip()
    x2 = origin(20)[0] + 100
    masks[20, 200:280, x2:x2 + 32] = 1.0
    assert final_record(masks, pose_data, final=False)["flags"]["mask_attached_leak"] == [20]
    assert "mask_attached_leak" not in final_record(masks, pose_data)["flags"]
    masks[20, 200:280, x2:x2 + 60] = 1.0
    assert final_record(masks, pose_data)["flags"]["mask_attached_leak"] == [20]


def test_on_the_final_mask_a_piece_is_measured_without_its_padding():
    # a detached 32 px block is what the final makes of a speck of a few pixels; a 64 px piece
    # keeps 12 x 12 px once the padding is off, 1.6% of her rectangle's 48 x 188 core: a speck
    masks, pose_data = clip()
    masks[20, 20:52, 4:36] = 1.0
    assert final_record(masks, pose_data, final=False)["frames"][20]["fragments"] == [round(1024 / 24000, 4)]
    assert final_record(masks, pose_data)["frames"][20]["fragments"] == []
    masks[20, 20:84, 4:68] = 1.0
    assert final_record(masks, pose_data, final=False)["flags"]["mask_fragmented"] == [20]
    record = final_record(masks, pose_data)
    assert "mask_fragmented" not in record["flags"] and record["frames"][20]["fragments"] == [round(144 / (48 * 188), 4)]


def test_the_final_mask_needs_its_pose():
    masks, _ = clip()
    with pytest.raises(ValueError, match="connect pose_data"):
        guard.check_mask(masks, None, MASK, final=True)


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
    assert json.loads(guard_run(masks, pose_data)[2])["frames"][30]["fragments"]
    hand = np.tile([(W - 15) / W, 10 / H, 0.9], (21, 1))
    pose_data["pose_metas_original"][30]["keypoints_right_hand"] = hand
    assert json.loads(guard_run(masks, pose_data)[2])["frames"][30]["fragments"] == []


def test_guards_off_never_raise_but_still_report():
    masks, pose_data = clip()
    masks[:10] = 0
    passed, flags, report = run(masks, pose_data, mask_guard=False)
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
    assert list(record["thresholds"]) == ["draw_threshold", "max_torso_jump", "max_limb_spike", "min_mask_to_box",
                                          "max_mask_outside_box", "max_attached_leak", "head_out_eyes_ears"]
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
    with pytest.raises(ValueError, match="straight from the tracker, before any resize"):
        guard.check_mask(small, pose_data, MASK)
    with pytest.raises(ValueError, match="the final mask of the same frames"):
        guard.check_mask(small, pose_data, MASK, final=True)


def test_single_frame_2d_mask_is_accepted():
    masks, pose_data = clip()
    pose_data = {"pose_metas_original": pose_data["pose_metas_original"][:1], "detections": pose_data["detections"][:1],
                 "pose_config": pose_data["pose_config"], "draw_threshold": DRAW_THRESHOLD}
    passed, flags, _ = run(masks[0], pose_data)
    assert passed
