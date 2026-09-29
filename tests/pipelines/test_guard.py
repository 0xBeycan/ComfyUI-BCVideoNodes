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
    assert flags["mask_missing_keypoints"] == list(range(20, 25)), report
    assert "mask_missing_keypoints (warning)" in report and "missed:" in report
    # the legs are back on frame 25: a dropout of her lower 40 px (27% of her), which fails
    assert not passed and flags["mask_loss_large"] == list(range(20, 25)), report


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


def test_a_piece_of_her_the_mask_split_off_is_not_a_speck():
    # a 30 px arm held out on every frame; on frame 20 a 4 px cut at her side splits it off: a
    # 26 x 30 px piece (3.25% of her rectangle, speck-sized) that her part holds on every frame
    # around it. The same piece on frame 20 alone is an object beside her
    masks, pose_data = clip()
    arm(masks, range(N), (60, 90), 30)
    block(masks, [20], (60, 90), (100, 104))
    assert pose_free_flags(masks) == {}
    _, flags, report = run(masks, pose_data)
    assert "mask_specks" not in flags and "mask_fragmented" not in flags, report
    masks, _ = clip()
    arm(masks, [20], (60, 90), 30)
    block(masks, [20], (60, 90), (100, 104))
    assert pose_free_flags(masks) == {"mask_specks": [20]}


def test_an_object_that_joins_her_for_a_frame_is_still_a_speck():
    # a toy beside her joins her side on frame 19 and is detached on frame 20: her part holds it
    # on one frame around frame 20, not two
    masks, _ = clip()
    arm(masks, [19, 20], (60, 90), 30)
    block(masks, [20], (60, 90), (100, 104))
    assert pose_free_flags(masks) == {"mask_specks": [20]}


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
    # from frame 20 on the mask leaves out her head for good (no dropout: it never comes back).
    # Five of her twenty drawn keypoints are out, a recall of 0.75, which fails as the head out
    # in place of the mask_missing_keypoints warning
    masks, pose_data = clip()
    head_off(masks, range(20, N))
    passed, flags, report = run(masks, pose_data)
    assert not passed and flags == {"mask_head_out": list(range(20, N))}, report
    assert "mask_head_out (fail)" in report and "missed: nose x20" in report


def test_the_nose_or_two_of_the_eyes_and_ears_outside_the_mask_are_the_head_out():
    # keypoints drawn 15 px left of her rectangle from frame 20 on, the mask as it is. One ear alone
    # is the mask's outline at the hair, not the head: with the legs cut off too (recall 0.65) the
    # frame stays a warning
    for out, legs, flags in (([0], False, {"mask_head_out": list(range(20, N))}),
                             ([14, 16], False, {"mask_head_out": list(range(20, N))}),
                             ([16], False, {}),
                             ([16], True, {"mask_missing_keypoints": list(range(20, N)),
                                           "mask_missed_limb": list(range(20, N))})):
        masks, pose_data = clip()
        for i in range(20, N):
            x1, y1 = origin(i)
            for j in out:
                place_keypoints(pose_data, [i], j, x1 - 15, y1 + 18)
            if legs:
                masks[i, y1 + 200:, :] = 0
        passed, got, report = run(masks, pose_data)
        assert got == flags and passed == ("mask_head_out" not in flags), (out, legs, report)


def test_how_many_eyes_and_ears_outside_the_mask_fail_is_the_config_s():
    # keypoints drawn 15 px left of her rectangle from frame 20 on, the mask as it is: one ear alone
    # fails at head_out_eyes_ears 1; an eye and an ear fail at the default 2 and at 3 are no check
    # at all (18 of 20 drawn keypoints inside, the recall of 0.9 is no warning); the nose fails at
    # any count, 4 included
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


def test_a_limb_end_on_the_frame_border_leaves_the_shot():
    # her right forearm reaches out to the left edge of the frame; the mask of it stops 20 px
    # short of the edge on every frame. A wrist 12 px inside the frame is a hand in the shot the
    # mask lost; a wrist the pose model puts on the outermost column is where the forearm leaves
    # the shot, its hand beyond the edge
    for wrist, missed in ((12, list(range(10, 30))), (0.5, None)):
        masks, pose_data = clip()
        for i in range(10, 30):
            x1, y1 = origin(i)
            masks[i, y1 + 100:y1 + 124, 20:x1] = 1.0
            place_keypoints(pose_data, [i], 3, 40, y1 + 112)
            place_keypoints(pose_data, [i], 4, wrist, y1 + 112)
        _, flags, report = run(masks, pose_data)
        assert flags.get("mask_missed_limb") == missed, report


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


def gone_on(*frames):
    """Every frame but `frames`: the arm is held out on those and gone from the others' mask."""
    return [i for i in range(N) if i not in frames]


def test_a_part_the_final_mask_leaves_out_for_one_frame_is_mask_loss():
    # her arm, held out 40 px from her side, is gone from frame 20's mask. The final mask the
    # workflow feeds the models reaches FINAL_GROW (10 px) beyond her side; the 29 px beyond that
    # which the frames around it hold (the body drifts a pixel a frame) are dropped: 29 x 40 px,
    # a disc of radius 15
    masks, pose_data = clip()
    arm(masks, gone_on(20), (60, 100), 40)
    passed, flags, report = run(masks, pose_data)
    assert passed and flags == {"mask_loss": [20]}, report
    _, _, metrics, _ = guard_run(masks, pose_data)
    frames = json.loads(metrics)["frames"]
    assert frames[20]["mask_loss"] == pytest.approx(15 / 240)
    assert frames[0]["mask_loss"] is None and frames[N - 1]["mask_loss"] is None
    assert "mask_loss (warning)" in report


def pose_free_flags(masks):
    """The Mask Guard's flags on `masks` without pose_data."""
    return json.loads(guard.check_mask(masks, None, MASK, stop_on_fail=False)[2])["flags"]


def test_a_dropout_of_a_twentieth_of_her_or_more_fails():
    # the arm of the test above, 40 rows tall, drops 29 x 40 = 1160 px of her 25600 px mask on
    # frame 19 (4.5%): a warning. 50 rows tall it drops 29 x 50 = 1450 px of 26000 (5.6%): a part of
    # her the models cannot restore, which fails in place of the warning
    for rows, area, flags in (((60, 100), 1160 / 25600, {"mask_loss": [20]}),
                              ((60, 110), 1450 / 26000, {"mask_loss_large": [20]})):
        masks, pose_data = clip()
        arm(masks, gone_on(20), rows, 40)
        passed, got, report = run(masks, pose_data)
        assert got == flags and passed == ("mask_loss" in flags), report
        _, _, metrics, _ = guard_run(masks, pose_data)
        frames = json.loads(metrics)["frames"]
        assert frames[20]["mask_loss_area"] == pytest.approx(area)
        assert frames[19]["mask_loss_area"] == 0.0 and frames[0]["mask_loss_area"] is None
        assert pose_free_flags(masks) == flags


def test_the_share_of_her_a_large_loss_drops_is_the_config_s():
    # the two dropouts of the test above, 4.5% and 5.6% of her: the 5.6% one fails at the default
    # large_loss_area 0.05 and is a mask_loss warning at 0.10; at 0 every mask_loss dropout fails,
    # the 4.5% one too
    for rows, share, flags in (((60, 110), 0.05, {"mask_loss_large": [20]}), ((60, 110), 0.10, {"mask_loss": [20]}),
                               ((60, 100), 0.05, {"mask_loss": [20]}), ((60, 100), 0.0, {"mask_loss_large": [20]})):
        masks, pose_data = clip()
        arm(masks, gone_on(20), rows, 40)
        config = guard.MaskGuardConfig(**{**dataclasses.asdict(MASK), "large_loss_area": share})
        record = json.loads(guard.check_mask(masks, pose_data, config, stop_on_fail=False)[2])
        assert record["flags"] == flags and record["thresholds"]["large_loss_area"] == share, (rows, share, record)


def test_only_a_dropout_that_reaches_the_warning_is_measured_for_a_large_loss():
    # a 15 px arm down most of her side, gone for a frame, leaves a 4 x 200 px strip beyond the
    # final's reach (test_mask_loss_is_the_thickness_not_the_area): too thin to warn, so its area
    # counts for nothing however long it is
    masks, _ = clip()
    arm(masks, gone_on(12), (20, 220), 15)
    record = json.loads(guard.check_mask(masks, None, MASK, stop_on_fail=False)[2])
    assert record["flags"] == {} and record["frames"][12]["mask_loss_area"] == 0.0


def test_a_hole_the_final_mask_fills_is_not_mask_loss():
    # a 30 x 30 hole between the knees on one frame: the grow fills all but its 10 x 10 core (a
    # disc of radius 5, over max_mask_loss), and every block of the final's grid over that core
    # holds grown mask, so the models never see the hole
    masks, pose_data = clip()
    block(masks, [20], (150, 180), (40, 70))
    _, flags, report = run(masks, pose_data)
    assert "mask_loss" not in flags, report
    assert json.loads(guard_run(masks, pose_data)[2])["frames"][20]["mask_loss"] == 0.0
    assert "mask_loss" not in pose_free_flags(masks)


def test_a_hole_that_holds_a_whole_block_of_the_final_s_grid_is_mask_loss():
    # an 84 x 84 hole on frame 20; the grow leaves its 64 x 64 core. BlockifyMask cuts the grown
    # box (rows 30-289, columns 70-189) into 8 x 3 blocks of 32 x 40 px: the block at rows
    # 126-157, columns 110-149 lies inside the core, holds no grown pixel and stays off - a disc
    # of radius 16 the models lose. The block is 1280 px of her 24000 px rectangle, 5.3%: a
    # large loss, which fails
    masks, _ = clip()
    block(masks, [20], (60, 144), (8, 92))
    record = json.loads(guard.check_mask(masks, None, MASK, stop_on_fail=False)[2])
    assert record["flags"] == {"mask_loss_large": [20]} and record["frames"][20]["mask_loss"] == pytest.approx(16 / 240)
    assert record["frames"][20]["mask_loss_area"] == pytest.approx(1280 / 24000)


def test_mask_loss_is_the_thickness_not_the_area():
    # beyond the final's reach, an arm 15 px wide down most of her side leaves a 4 x 200 px strip
    # (800 px) when it is gone for a frame, one 25 px wide a 14 x 14 block (196 px): the block
    # holds a disc of radius 7, over max_mask_loss, the strip one of radius 2, under half of it
    strip, _ = clip()
    arm(strip, gone_on(12), (20, 220), 15)
    assert "mask_loss" not in pose_free_flags(strip)
    blob, _ = clip()
    arm(blob, gone_on(28), (60, 74), 25)
    assert pose_free_flags(blob) == {"mask_loss": [28]}


def test_a_region_lost_for_good_is_not_mask_loss():
    masks, pose_data = clip()
    arm(masks, range(0, 20), (60, 100), 40)
    _, flags, report = run(masks, pose_data)
    assert "mask_loss" not in flags, report


def test_the_mask_loss_threshold_is_the_config_s():
    # the arm gone from frame 20 leaves a disc of radius 15 (0.0625 of the shorter side) beyond the
    # final's reach, and no limb crosses it: 0.06 flags it, 0.07 does not
    masks, pose_data = clip()
    arm(masks, gone_on(20), (60, 100), 40)
    for threshold, flagged in ((0.06, [20]), (0.07, None)):
        loose = guard.MaskGuardConfig(**{**dataclasses.asdict(MASK), "max_mask_loss": threshold})
        _, _, metrics, _ = guard.check_mask(masks, pose_data, loose, stop_on_fail=False)
        record = json.loads(metrics)
        assert record["flags"].get("mask_loss") == flagged and record["thresholds"]["max_mask_loss"] == threshold


def test_a_region_dropped_for_a_few_frames_is_one_mask_loss():
    # the arm, held on frame 19 and on frame 25, is gone on 20-24: beyond the reach of the finals
    # of those frames (her side drifts a pixel a frame) the region all of them drop is 25 x 40 px,
    # a disc of radius 13
    masks, pose_data = clip()
    arm(masks, gone_on(*range(20, 25)), (60, 100), 40)
    passed, flags, report = run(masks, pose_data)
    assert passed and flags == {"mask_loss": list(range(20, 25))}, report
    _, _, metrics, _ = guard_run(masks, pose_data)
    frames = json.loads(metrics)["frames"]
    assert [frames[i]["mask_loss_run"] for i in range(19, 26)] == [0.0] + [pytest.approx(13 / 240)] * 5 + [0.0]
    assert all(frames[i]["mask_loss"] == 0.0 for i in range(19, 26))    # no single-frame dropout in it


def test_a_run_of_frames_needs_twice_the_thickness():
    # a 25 px arm gone for one frame leaves 14 x 14 px beyond the final's reach, a disc of radius 7
    # (0.029, over max_mask_loss 0.0185); gone on frames 20-22, the part all three finals leave out
    # holds one of radius 6 (0.025), under twice the threshold
    for frames, flagged in (([20], [20]), (range(20, 23), None)):
        masks, _ = clip()
        arm(masks, gone_on(*frames), (60, 74), 25)
        assert pose_free_flags(masks).get("mask_loss") == flagged


def test_a_dropout_longer_than_the_window_is_not_mask_loss():
    masks, pose_data = clip()
    arm(masks, gone_on(*range(20, 29)), (60, 100), 40)     # nine frames
    _, flags, report = run(masks, pose_data)
    assert "mask_loss" not in flags, report


def test_a_limb_that_moves_away_and_back_is_not_mask_loss():
    # the arm is lifted 50 px on frames 20-24 and back on frame 25: the spot it left is held on
    # both ends and on none of the frames between, as a dropout would be, but the arm is right
    # there the whole time
    masks, _ = clip()
    arm(masks, gone_on(*range(20, 25)), (60, 100), 40)
    arm(masks, range(20, 25), (10, 50), 40)
    assert "mask_loss" not in pose_free_flags(masks)
    masks, _ = clip()
    arm(masks, gone_on(*range(20, 25)), (60, 100), 40)   # the same arm gone from the mask instead
    assert pose_free_flags(masks)["mask_loss"] == list(range(20, 25))


def test_a_leak_that_blinks_off_is_not_mask_loss():
    # the arm of the one-frame dropout above, taken in on frames 19 and 21 only: the region frame
    # 20 drops is held on the frames next to it and nowhere beyond them, background blinking off.
    # Held on frames 17-19 and 21-23 it is a part of her the mask dropped
    masks, _ = clip()
    arm(masks, [19, 21], (60, 100), 40)
    assert "mask_loss" not in pose_free_flags(masks)
    masks, _ = clip()
    arm(masks, [17, 18, 19, 21, 22, 23], (60, 100), 40)
    assert pose_free_flags(masks) == {"mask_loss": [20]}


def test_a_part_gone_beyond_the_run_counts_where_the_pose_has_it_next_to_the_run():
    # the arm is held out on frames 10-19, dropped on 20, back on 21 and gone after (her hand
    # leaves the shot): beyond frame 21 nothing holds the region, as with background blinking
    # off. With her left forearm reaching into it on the frames around the dropout, the pose has
    # the body there
    masks, pose_data = clip()
    arm(masks, [*range(10, 20), 21], (60, 100), 40)
    assert "mask_loss" not in pose_free_flags(masks)
    for i in range(N):
        x1, y1 = origin(i)
        place_keypoints(pose_data, [i], 7, x1 + 130, y1 + 80)
    _, flags, report = run(masks, pose_data)
    assert flags["mask_loss"] == [20], report


def test_a_thin_dropout_counts_where_the_skeleton_crosses_it():
    # a 17 px arm gone from frame 20 leaves 6 x 24 px beyond the final's reach: a disc of radius
    # 3, 0.0125 of the shorter side, under max_mask_loss (0.0185) but over half of it. With her left
    # forearm reaching into it the pose sees the body there; lower down, beside her hip, no limb
    # crosses it
    for rows, wrist, with_pose in (((100, 124), True, [20]), ((150, 174), False, None)):
        masks, pose_data = clip()
        arm(masks, gone_on(20), rows, 17)
        if wrist:
            for i in range(N):
                x1, y1 = origin(i)
                place_keypoints(pose_data, [i], 7, x1 + 113, y1 + 112)
        _, flags, report = run(masks, pose_data)
        assert flags.get("mask_loss") == with_pose, report
        assert "mask_loss" not in pose_free_flags(masks)


# --- the final mask: the WanAnimate Preprocess Guard's, grown and blockified -------------------

def final_record(masks, pose_data, final=True):
    """The mask group's metrics record on `masks` as the final mask (or, with `final` False, as a
    raw one)."""
    return json.loads(guard.check_mask(masks, pose_data, MASK, stop_on_fail=False, final=final)[2])


def test_on_the_final_mask_a_dropped_block_counts_only_where_it_holds_a_keypoint():
    # her left forearm reaches out of the body to the wrist 150 px right of her rectangle's left
    # edge; the final mask's moving block grid turns a 32 px block above it on and off (on frames
    # 18-19 and 21-22, off on 20). The forearm's line crosses that block, a raw mask's dropout; on
    # the final no keypoint lies in it. A block the final drops over her wrist is the hand the raw
    # mask lost: with the block above it that frames 19 and 21 both hold, 30 x 32 + 23 x 24 = 1512
    # px of her 26464 px mask on frame 19 (5.7%), a large loss
    masks, pose_data = clip()
    for i in range(N):
        x1, y1 = origin(i)
        masks[i, y1 + 100:y1 + 124, x1 + 100:x1 + 160] = 1.0
        place_keypoints(pose_data, [i], 7, x1 + 150, y1 + 112)
    for i in (18, 19, 21, 22):
        x1, y1 = origin(i)
        masks[i, y1 + 68:y1 + 100, x1 + 110:x1 + 142] = 1.0
    assert final_record(masks, pose_data, final=False)["flags"].get("mask_loss") == [20]
    assert "mask_loss" not in final_record(masks, pose_data)["flags"]
    x1, y1 = origin(20)
    masks[20, y1 + 96:y1 + 128, x1 + 136:x1 + 168] = 0
    record = final_record(masks, pose_data)
    assert record["flags"]["mask_loss_large"] == [20] and "mask_loss" not in record["flags"]
    assert record["frames"][20]["mask_loss_area"] == pytest.approx(1512 / 26464)


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


def test_on_the_final_mask_the_padding_is_body_the_skeleton_accounts_for():
    # the final reaches FINAL_PAD (26 px) beyond the raw mask on average: her rectangle grown by
    # that much is the body the pose draws, grown as the mask was
    masks, pose_data = clip()
    for i in range(N):
        x1, y1 = origin(i)
        masks[i, y1 - 26:y1 + 266, x1 - 26:x1 + 126] = 1.0
    assert final_record(masks, pose_data, final=False)["flags"]["body_not_drawn"] == list(range(N))
    assert "body_not_drawn" not in final_record(masks, pose_data)["flags"]


def test_on_the_final_mask_a_piece_is_measured_without_its_padding():
    # a detached 32 px block is what the final makes of a speck of a few pixels; a 64 px piece
    # keeps 12 x 12 px once the padding is off, 1.6% of her rectangle's 48 x 188 core: a speck
    masks, pose_data = clip()
    masks[20, 20:52, 4:36] = 1.0
    assert final_record(masks, pose_data, final=False)["flags"]["mask_specks"] == [20]
    assert final_record(masks, pose_data)["frames"][20]["fragments"] == []
    masks[20, 20:84, 4:68] = 1.0
    assert final_record(masks, pose_data, final=False)["flags"]["mask_fragmented"] == [20]
    record = final_record(masks, pose_data)
    assert record["flags"]["mask_specks"] == [20] and record["frames"][20]["fragments"] == [round(144 / (48 * 188), 4)]


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
                                          "min_mask_iou", "max_mask_loss", "head_out_eyes_ears", "large_loss_area"]
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
