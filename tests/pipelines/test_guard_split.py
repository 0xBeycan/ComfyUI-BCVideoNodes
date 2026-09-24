"""Each guard group on its own: the Pose Guard needs nothing but pose_data, the Mask Guard the
mask and pose_data, and each reports, measures, plots and stops by itself.

    python -m pytest tests/pipelines/test_guard_split.py
"""
import json

import pytest

from bcvideonodes.pipelines import guard
from guard_fakes import LEGS, MASK, POSE, H, W, clip, drop_keypoints


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
    drop_keypoints(pose_data, range(18, 24), LEGS)
    with pytest.raises(guard.GuardFailed, match="Pose guard: FAILED") as failure:
        guard.check_pose(pose_data, POSE)
    assert "pose_incomplete" in str(failure.value) and "mask_" not in str(failure.value)


def test_pose_guard_off_measures_but_never_stops():
    _, pose_data = clip()
    drop_keypoints(pose_data, range(18, 24), LEGS)
    _, report, metrics, _ = guard.check_pose(pose_data, POSE, enabled=False)
    assert report.startswith("Pose guard: passed") and "pose_incomplete (off)" in report
    assert json.loads(metrics)["flags"]["pose_incomplete"] == list(range(18, 24))


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



def test_both_guards_take_a_zero_frame_clip():
    masks, pose_data = clip()
    pose_data["pose_metas_original"], pose_data["detections"] = [], []
    guard.check_pose(pose_data, POSE)
    out, report, _, _ = guard.check_mask(masks[:0], pose_data, MASK)
    assert out.shape[0] == 0 and report.startswith("Mask guard: passed"), report
