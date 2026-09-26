"""SAM 3.1 Multiplex's config: the measured defaults, the mode-tagged tooltips and ranges, and the
A/B switches' values. No model is loaded.

The config module imports without ComfyUI, but this file imports comfy.cli_args first, so it runs
where ComfyUI is importable (the pod, with the ComfyUI root on PYTHONPATH) and is skipped
elsewhere."""
import dataclasses

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import sam3  # noqa: E402


# --- config --------------------------------------------------------------------------------

def test_config_prompt_defaults_are_easy_sam3_s_values():
    c = sam3.SAM3Config()
    assert (c.birth_threshold, c.detection_threshold, c.nms_iou, c.match_iou) == (0.70, 0.50, 0.10, 0.50)
    assert (c.hotstart_frames, c.hotstart_unmatched) == (15, 8)
    assert (c.recondition_every, c.recondition_score, c.recondition_iou) == (16, 0.80, 0.80)
    assert (c.fill_hole_area, c.memory_gap) == (16, 0)


# The run that validated easy-sam3's set on SAM 3.1 ("S3"): the config node at the earlier defaults
# with these eleven fields set, and nothing else.
S3 = dict(input_range="-1..1", clear_on_anchor=False, memory_gap=0, anchor_mask="propagated", max_conditioning_frames=4,
          keep_birth_frame=False, obj_ptr_token="best_iou", anchor_track_score=0.8, memory_selection=True,
          detection_threshold=0.50, birth_threshold=0.70)
EARLIER = dict(input_range="0..1", clear_on_anchor=True, memory_gap=7, anchor_mask="detection", max_conditioning_frames=2,
               keep_birth_frame=True, obj_ptr_token="token_0", anchor_track_score=0.0, memory_selection=False,
               detection_threshold=0.30, birth_threshold=0.50)


def test_the_defaults_are_the_validated_s3_config_field_by_field():
    now = sam3.SAM3Config()
    before = dataclasses.replace(now, **EARLIER)
    for f in dataclasses.fields(now):
        if f.name in S3:
            assert getattr(now, f.name) == S3[f.name], f.name
            assert getattr(before, f.name) != S3[f.name], f.name
        else:
            assert getattr(now, f.name) == getattr(before, f.name), f.name
    assert sorted(S3) == sorted(EARLIER)


def test_config_box_keypoint_defaults_are_the_source_values():
    c = sam3.SAM3Config()
    assert not hasattr(c, "min_keypoint_conf")   # it is the pose's, read from pose_data
    assert (c.reseed_interval, c.max_propagate) == (24, 72)
    assert (c.min_anchor_keypoints, c.min_anchor_conf, c.min_anchor_completeness) == (8, 0.5, 0.9)
    assert (c.min_tracked_recall, c.negative_points, c.negative_margin) == (0.9, 8, 0.04)
    assert (c.min_annexed_fraction, c.annexed_points, c.mask_threshold) == (0.03, 8, -1.0)
    assert (c.min_island_fraction, c.max_hole_fraction, c.refine, c.temporal) == (0.01, 0.01, True, True)


def test_config_multi_object_defaults_are_their_own_fields():
    c = sam3.SAM3Config()
    assert (c.new_object_threshold, c.assoc_iou, c.duplicate_frames, c.occlusion_iou, c.shrink_keep) == \
        (0.50, 0.10, 8, 0.70, 0.30)


def test_config_fields_carry_a_mode_tagged_tooltip_and_a_range():
    tags = {}
    for f in dataclasses.fields(sam3.SAM3Config):
        tip = f.metadata["tooltip"]
        tag = tip[:tip.index("]") + 1]
        assert tag in ("[prompt]", "[prompt, max_objects 1]", "[prompt, max_objects > 1]", "[box_keypoint]"), f.name
        tags.setdefault(tag, []).append(f.name)
        if isinstance(f.default, str):
            assert f.default in f.metadata["choices"], f.name
        elif not isinstance(f.default, bool):
            assert f.metadata["min"] <= f.default <= f.metadata["max"], f.name
    assert tags["[prompt]"] == ["birth_threshold", "detection_threshold", "nms_iou", "match_iou", "hotstart_frames",
                                "hotstart_unmatched", "recondition_every", "recondition_score", "recondition_iou",
                                "fill_hole_area", "memory_gap", "anchor_matching", "unmatched_counting",
                                "input_range", "obj_ptr_token", "memory_mask"]
    assert tags["[prompt, max_objects > 1]"] == ["new_object_threshold", "assoc_iou", "duplicate_frames",
                                                 "occlusion_iou", "shrink_keep"]
    assert tags["[prompt, max_objects 1]"] == ["anchor_output", "clear_on_anchor", "anchor_mask", "max_conditioning_frames",
                                               "keep_birth_frame", "anchor_track_score", "memory_selection"]


def test_config_ab_switches_default_to_ours_and_reject_other_values():
    c = sam3.SAM3Config()
    for name in ("anchor_matching", "unmatched_counting"):
        assert getattr(c, name) == sam3.OURS, name
        sam3.SAM3Config(**{name: sam3.META})
        with pytest.raises(ValueError, match=name):
            sam3.SAM3Config(**{name: "theirs"})
    assert c.mask_threshold == -1.0


def test_config_anchor_output_defaults_to_the_propagated_mask_and_rejects_other_values():
    assert sam3.SAM3Config().anchor_output == sam3.PROPAGATED
    sam3.SAM3Config(anchor_output=sam3.DETECTION)
    with pytest.raises(ValueError, match="anchor_output"):
        sam3.SAM3Config(anchor_output="meta")


def test_config_run_switches_default_to_meta_s_side_and_reject_other_values():
    c = sam3.SAM3Config()
    assert (c.input_range, c.obj_ptr_token, c.memory_mask) == ("-1..1", "best_iou", "cleaned")
    for name, other in (("input_range", sam3.UNIT_RANGE), ("obj_ptr_token", sam3.TOKEN_0), ("memory_mask", sam3.RAW)):
        assert getattr(sam3.SAM3Config(**{name: other}), name) == other
        with pytest.raises(ValueError, match=name):
            sam3.SAM3Config(**{name: "meta"})
    # after anchor_output: the widgets of saved workflows keep their positions
    names = [f.name for f in dataclasses.fields(sam3.SAM3Config)]
    assert names[-10:-6] == ["anchor_output", "input_range", "obj_ptr_token", "memory_mask"]


def test_config_anchor_policy_switches_default_to_easy_sam3_s():
    c = sam3.SAM3Config()
    assert (c.clear_on_anchor, c.anchor_mask, c.max_conditioning_frames, c.keep_birth_frame, c.anchor_track_score,
            c.memory_selection) == (False, "propagated", 4, False, 0.8, True)
    assert sam3.SAM3Config(anchor_mask=sam3.DETECTION).anchor_mask == "detection"
    with pytest.raises(ValueError, match="anchor_mask"):
        sam3.SAM3Config(anchor_mask="tracker")
    # last: the widgets of saved workflows keep their positions
    names = [f.name for f in dataclasses.fields(sam3.SAM3Config)]
    assert names[-6:] == ["clear_on_anchor", "anchor_mask", "max_conditioning_frames", "keep_birth_frame",
                          "anchor_track_score", "memory_selection"]
