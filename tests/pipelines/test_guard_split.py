"""Each guard group on its own: the Pose Guard needs nothing but pose_data, the Mask Guard the
mask and, for its pose-based checks, pose_data, and each reports, measures, plots and stops by
itself.

    python -m pytest tests/pipelines/test_guard_split.py
"""
import json

import pytest

from bcvideonodes.pipelines import guard
from guard_fakes import LEGS, MASK, N, POSE, H, W, arm, clip, drop_keypoints, origin


def jump_torso(pose_data, i):
    """The torso keypoints jump half the frame width on frame `i` (pose_jump)."""
    pts = pose_data["pose_metas_original"][i]["keypoints_body"].copy()
    pts[guard.TORSO, 0] += 0.5
    pose_data["pose_metas_original"][i]["keypoints_body"] = pts


def test_pose_guard_alone_passes_a_clean_clip_and_passes_pose_data_through():
    _, pose_data = clip()
    out, report, metrics, timeline = guard.check_pose(pose_data, POSE)
    assert out is pose_data
    assert report.startswith("Pose guard: passed"), report
    record = json.loads(metrics)
    assert record["guard"] == "pose" and record["enabled"] == sorted(guard.POSE_CHECKS)
    assert all(tuple(row) == guard.POSE_ROW for row in record["frames"])
    assert timeline.dim() == 4 and timeline.shape[0] == 1 and timeline.shape[-1] == 3


def test_pose_guard_alone_fires_only_pose_checks():
    masks, pose_data = clip()
    masks[:10] = 0          # a mask fault the pose guard cannot see
    jump_torso(pose_data, 12)
    with pytest.raises(guard.GuardFailed, match="Pose guard: FAILED") as failure:
        guard.check_pose(pose_data, POSE)
    assert "pose_jump" in str(failure.value) and "mask_" not in str(failure.value)


def test_pose_guard_warnings_never_stop():
    _, pose_data = clip()
    drop_keypoints(pose_data, range(18, 24), LEGS)
    _, report, metrics, _ = guard.check_pose(pose_data, POSE)
    assert report.startswith("Pose guard: passed") and "pose_incomplete (warning)" in report
    assert json.loads(metrics)["flags"]["pose_incomplete"] == list(range(18, 24))


def test_pose_guard_off_measures_but_never_stops():
    _, pose_data = clip()
    jump_torso(pose_data, 12)
    _, report, metrics, _ = guard.check_pose(pose_data, POSE, enabled=False)
    assert report.startswith("Pose guard: passed") and "pose_jump (off)" in report
    assert json.loads(metrics)["flags"]["pose_jump"] == [12, 13]   # out and back


def test_only_damage_diffusion_cannot_absorb_stops():
    assert set(guard.POSE_CHECKS + guard.MASK_CHECKS) - guard.WARNINGS == {
        "pose_jump", "subject_switch", "mask_empty", "mask_leak", "mask_fragmented", "mask_head_out", "mask_loss_large"}


def test_pose_guard_defaults_are_the_measured_ones():
    assert guard.PoseGuardConfig() == POSE
    assert guard.MaskGuardConfig() == MASK


def test_mask_guard_alone_passes_a_clean_clip_and_passes_the_mask_through():
    masks, pose_data = clip()
    out, report, metrics, timeline = guard.check_mask(masks, pose_data, MASK)
    assert out is masks
    assert report.startswith("Mask guard: passed"), report
    record = json.loads(metrics)
    assert record["guard"] == "mask" and record["enabled"] == sorted(guard.MASK_CHECKS)
    assert all(tuple(row) == guard.MASK_ROW for row in record["frames"])
    assert timeline.dim() == 4 and timeline.shape[0] == 1 and timeline.shape[-1] == 3


def test_mask_guard_alone_fires_only_mask_checks():
    masks, pose_data = clip()
    masks[:10] = 0
    pose_data["detections"][15]["bbox"] = [0.0, 0.0, 40.0, 40.0]   # a pose fault it does not own
    with pytest.raises(guard.GuardFailed, match="Mask guard: FAILED") as failure:
        guard.check_mask(masks, pose_data, MASK)
    assert "mask_empty" in str(failure.value) and "subject_switch" not in str(failure.value)


def test_mask_guard_takes_the_keypoints_from_pose_data():
    # the keypoint clause of the fragment check: the same detached piece is her hand when
    # pose_data puts a confident wrist in it, and a detached piece when it does not
    masks, pose_data = clip()
    masks[30, 40:, 90:190] = 1.0
    masks[30, 0:20, W - 30:] = 1.0
    _, _, metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    assert json.loads(metrics)["frames"][30]["fragments"]
    pts = pose_data["pose_metas_original"][30]["keypoints_body"].copy()
    pts[4] = ((W - 15) / W, 10 / H, 0.9)
    pose_data["pose_metas_original"][30]["keypoints_body"] = pts
    _, _, metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    assert not json.loads(metrics)["frames"][30]["fragments"]


def test_mask_guard_frame_edge_does_not_excuse_an_object():
    masks, pose_data = clip()
    masks[30, 40:, 90:190] = 1.0      # the body runs off the bottom edge
    masks[30, 170:250, 0:30] = 1.0    # an object beside her, against the left edge
    _, _, metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    assert 30 in json.loads(metrics)["flags"]["mask_fragmented"]


def test_mask_guard_off_measures_but_never_stops():
    masks, pose_data = clip()
    masks[:10] = 0
    _, report, _, _ = guard.check_mask(masks, pose_data, MASK, enabled=False)
    assert report.startswith("Mask guard: passed") and "mask_empty (off)" in report


def test_the_wrong_config_type_is_an_error():
    masks, pose_data = clip()
    with pytest.raises(TypeError):
        guard.check_pose(pose_data, MASK)
    with pytest.raises(TypeError):
        guard.check_mask(masks, pose_data, {"min_mask_iou": 0.6})


def test_combine_needs_one_of_each_group_of_the_same_clip():
    masks, pose_data = clip()
    _, _, pose_metrics, _ = guard.check_pose(pose_data, POSE, stop_on_fail=False)
    _, _, mask_metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    with pytest.raises(ValueError, match="check_pose and check_mask"):
        guard.combine_guards(mask_metrics, pose_metrics)
    short = {"pose_metas_original": pose_data["pose_metas_original"][:5], "detections": pose_data["detections"][:5],
             "pose_config": pose_data["pose_config"], "draw_threshold": pose_data["draw_threshold"]}
    _, _, short_metrics, _ = guard.check_pose(short, POSE, stop_on_fail=False)
    with pytest.raises(ValueError, match="same clip"):
        guard.combine_guards(short_metrics, mask_metrics)


def test_combine_names_the_first_shared_key_the_groups_disagree_on():
    masks, pose_data = clip()
    _, _, pose_metrics, _ = guard.check_pose(pose_data, POSE, stop_on_fail=False)
    _, _, mask_metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    record = json.loads(mask_metrics)
    record["frames"][5].update(frame=99, box_iou_prev=0.0)    # both keys the two rows share
    with pytest.raises(ValueError) as failure:
        guard.combine_guards(pose_metrics, json.dumps(record))
    # the pose row's order decides, not the hash seed: frame comes before box_iou_prev
    assert str(failure.value) == ("frame 5: the pose and the mask checks disagree on frame (5 and 99); "
                                  "check the same clip with the same pose_data")


def test_the_keypoint_threshold_is_the_draw_threshold_from_pose_data():
    masks, pose_data = clip()
    pose_data["draw_threshold"] = 0.95   # above every keypoint's 0.9: nothing is drawn
    _, _, pose_metrics, _ = guard.check_pose(pose_data, POSE, stop_on_fail=False)
    _, _, mask_metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    for metrics in (pose_metrics, mask_metrics):
        assert json.loads(metrics)["thresholds"]["draw_threshold"] == 0.95
    assert all(row["drawn_keypoints"] == 0 for row in json.loads(pose_metrics)["frames"])
    assert all(not row["box_reliable"] for row in json.loads(mask_metrics)["frames"])
    _, metrics, _ = guard.combine_guards(pose_metrics, mask_metrics, stop_on_fail=False)
    assert json.loads(metrics)["thresholds"]["draw_threshold"] == 0.95


def test_pose_data_without_the_draw_threshold_is_an_error():
    masks, pose_data = clip()
    del pose_data["draw_threshold"]
    with pytest.raises(ValueError, match="draw_threshold"):
        guard.check_pose(pose_data, POSE)
    with pytest.raises(ValueError, match="draw_threshold"):
        guard.check_mask(masks, pose_data, MASK)


def test_combined_equals_both_groups_side_by_side():
    masks, pose_data = clip()
    masks[20:25, 200:, :] = 0
    drop_keypoints(pose_data, range(18, 24), LEGS)
    _, _, pose_metrics, _ = guard.check_pose(pose_data, POSE, stop_on_fail=False)
    _, _, mask_metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    _, metrics, _ = guard.combine_guards(pose_metrics, mask_metrics, stop_on_fail=False)
    pose, mask, both = (json.loads(m) for m in (pose_metrics, mask_metrics, metrics))
    assert both["flags"] == {**pose["flags"], **mask["flags"]}
    for p, m, row in zip(pose["frames"], mask["frames"], both["frames"]):
        assert row == {**p, **m}



def test_mask_guard_without_pose_data_runs_the_pose_free_checks():
    masks, pose_data = clip()
    masks[30, 170:250, 0:30] = 1.0                        # an object beside her
    arm(masks, [i for i in range(N) if i != 20], (60, 100), 40)   # her arm held out, dropped on one frame
    _, y1 = origin(20)
    masks[25:35, y1 + 200:, :] = 0                        # her legs cut off for good: only a pose sees that
    out, report, metrics, timeline = guard.check_mask(masks, None, MASK, stop_on_fail=False)
    record = json.loads(metrics)
    assert out is masks and record["flags"] == {"mask_loss": [20], "mask_fragmented": [30]}, report
    assert report.startswith("Mask guard: FAILED") and report.splitlines()[-1] == (
        "- without pose_data, not checked: mask_empty, mask_leak, mask_attached_leak, mask_missing_keypoints, "
        "mask_head_out, mask_missed_limb, body_not_drawn, mask_unstable")
    assert record["thresholds"]["draw_threshold"] is None
    row = record["frames"][25]
    assert (row["keypoint_recall"], row["missed_keypoints"], row["body_not_drawn"], row["box_reliable"]) == (None, [], None, False)
    assert all(tuple(row) == guard.MASK_ROW for row in record["frames"])
    assert timeline.shape[0] == 1 and timeline.shape[-1] == 3


def test_mask_guard_without_pose_data_cannot_tell_her_hand_from_an_object():
    # the piece her wrist sits in is her hand with pose_data, a detached object without it
    masks, pose_data = clip()
    masks[30, 40:, 90:190] = 1.0
    masks[30, 0:20, W - 30:] = 1.0
    pts = pose_data["pose_metas_original"][30]["keypoints_body"].copy()
    pts[4] = ((W - 15) / W, 10 / H, 0.9)
    pose_data["pose_metas_original"][30]["keypoints_body"] = pts
    _, _, metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    assert not json.loads(metrics)["frames"][30]["fragments"]
    _, _, metrics, _ = guard.check_mask(masks, None, MASK, stop_on_fail=False)
    assert json.loads(metrics)["frames"][30]["fragments"]


def test_mask_guard_without_pose_data_takes_a_piece_the_frame_edge_cut_off_for_her():
    # her arm leaves the shot at the right edge and comes back in further up it: the piece runs
    # off the side of the frame her body runs off, so the frame edge is what cut it from her
    masks, _ = clip()
    x1, y1 = origin(30)
    masks[30, y1 + 110:y1 + 140, x1 + 100:] = 1.0     # the arm runs off the right edge
    masks[30, y1 + 10:y1 + 40, W - 20:] = 1.0          # and comes back in further up that edge
    _, _, metrics, _ = guard.check_mask(masks, None, MASK, stop_on_fail=False)
    record = json.loads(metrics)
    assert record["frames"][30]["fragments"] == [] and not record["flags"], record["flags"]


@pytest.mark.parametrize("rows", [(5, 30), (0, 25)], ids=["inside the frame", "at a side she does not reach"])
def test_mask_guard_without_pose_data_still_reports_a_piece_off_her_edges(rows):
    # a hand above her head, cut from the arm by a burned-in caption: inside the frame, or running
    # off the top while her body reaches no side of the frame - nothing says it is hers
    masks, _ = clip()
    x1, _ = origin(30)
    masks[30, rows[0]:rows[1], x1 + 30:x1 + 60] = 1.0
    _, _, metrics, _ = guard.check_mask(masks, None, MASK, stop_on_fail=False)
    assert json.loads(metrics)["flags"] == {"mask_specks": [30]}


def test_both_guards_take_a_zero_frame_clip():
    masks, pose_data = clip()
    pose_data["pose_metas_original"], pose_data["detections"] = [], []
    guard.check_pose(pose_data, POSE)
    out, report, _, _ = guard.check_mask(masks[:0], pose_data, MASK)
    assert out.shape[0] == 0 and report.startswith("Mask guard: passed"), report
