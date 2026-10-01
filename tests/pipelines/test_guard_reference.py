"""The Wan Animate guards' reference check on synthetic masks: the character on the reference
(here given as its mask, as SAM 3.1 Multiplex would find it) placed as core places the reference
and measured against mask frame 0 - raw against raw in the Mask Guard, the final against the final
in the WanAnimate Preprocess Guard - and the metrics and report without a reference exactly as they
were. guard_fakes.clip()'s person is a 100 x 240 rectangle at columns 60-159, rows 40-279 of a
240 x 320 frame on frame 0. Runs without ComfyUI or any model:

    python -m pytest tests/pipelines/test_guard_reference.py
"""
import dataclasses
import json

import pytest
import torch

from bcvideonodes.pipelines import guard
from guard_fakes import MASK, POSE, H, W, clip, origin
from scail2_fakes import scail2

KEYS = ["guard", "thresholds", "enabled", "flags", "frames"]


def person(x1=60, y1=40, width=100, height=240, size=(H, W)):
    """A reference MASK [1, H', W'] holding a `width` x `height` character at (x1, y1)."""
    mask = torch.zeros(1, *size)
    mask[0, y1:y1 + height, x1:x1 + width] = 1.0
    return mask


def mask_run(masks, pose_data, reference=None, final=False, config=None, **kwargs):
    """The Mask Guard's (report, metrics record), without stopping unless asked."""
    out = guard.check_mask(masks, pose_data, MASK, final=final, reference=reference, reference_config=config,
                           **{"stop_on_fail": False, **kwargs})
    assert out[0] is masks
    return out[1], json.loads(out[2])


def final_clip():
    """clip() as the final mask the sampler gets: each frame's rectangle grown by 10 px, whose
    BlockifyMask blocks all hold grown mask (tests/pipelines/test_guard.py, HOLE), so the final is
    the grown box, 120 x 260 px at columns x1 - 10 to x1 + 110, rows 30-289."""
    masks, pose_data = clip()
    final = torch.zeros_like(masks)
    for i in range(masks.shape[0]):
        x1, y1 = origin(i)
        final[i, y1 - 10:y1 + 250, x1 - 10:x1 + 110] = 1.0
    return final, pose_data


# --- raw against raw: the Mask Guard ------------------------------------------------------------

def test_a_reference_placed_like_frame_0_is_no_warning():
    masks, pose_data = clip()
    report, record = mask_run(masks, pose_data, person())
    assert record["reference"] == {"area": 24000 / (H * W), "cropped": 0.0, "iou_first_frame": 1.0,
                                   "scale_first_frame": 1.0, "flags": []}
    assert report.startswith("Mask guard: passed") and "reference_misaligned" not in report
    assert "reference image: area 0.312, cropped 0.000, IoU with mask frame 0 1.000, scale vs mask frame 0 1.000" in report


def test_a_shifted_reference_is_misaligned_and_never_stops():
    # the character 80 px right of frame 0's person (columns 140-239 against 60-159): 20 of 180 columns shared
    masks, pose_data = clip()
    report, record = mask_run(masks, pose_data, person(x1=140), stop_on_fail=True)
    assert record["reference"]["iou_first_frame"] == pytest.approx(20 / 180)
    assert record["reference"]["flags"] == ["reference_misaligned"] and record["flags"] == {}
    assert report.startswith("Mask guard: passed")
    assert ("- reference_misaligned (warning): IoU 0.111 with mask frame 0; replacement expects the reference posed "
            "and placed like the first frame") in report


def test_a_scaled_reference_is_misaligned():
    # the character at half her height and width, top-left at frame 0's: a quarter of her area
    masks, pose_data = clip()
    _, record = mask_run(masks, pose_data, person(width=50, height=120))
    assert record["reference"]["iou_first_frame"] == pytest.approx(0.25)
    assert record["reference"]["scale_first_frame"] == pytest.approx(0.5)
    assert record["reference"]["flags"] == ["reference_misaligned"]


def test_the_threshold_is_the_config_s():
    # an IoU of 0.25 is misaligned at the default 0.4 and not at 0.2
    masks, pose_data = clip()
    reference = person(width=50, height=120)
    for iou, flags in ((0.4, ["reference_misaligned"]), (0.2, [])):
        _, record = mask_run(masks, pose_data, reference, config=guard.ReferenceGuardConfig(min_reference_iou=iou))
        assert record["reference"]["flags"] == flags and record["thresholds"]["min_reference_iou"] == iou


def test_each_guard_has_its_own_default_on_one_field():
    # 0.4 on the raw mask, 0.5 on the final, whose grow and blockify raise every IoU; the same
    # field otherwise, its tooltip saying both
    (raw,), (final,) = (dataclasses.fields(cls) for cls in (guard.ReferenceGuardConfig, guard.FinalReferenceGuardConfig))
    assert (raw.name, raw.default, final.name, final.default) == ("min_reference_iou", 0.4, "min_reference_iou", 0.5)
    assert raw.metadata == final.metadata
    assert "First values, set on a small set of clips: 0.4 for the Mask Guard on the raw mask, 0.5 for the WanAnimate " \
           "Preprocess Guard on the final mask." in raw.metadata["tooltip"]


def test_the_reference_is_placed_as_core_places_it():
    # at twice the generation size, the character twice as big: resized to the mask's size it is frame 0's
    masks, pose_data = clip()
    _, record = mask_run(masks, pose_data, person(120, 80, 200, 480, size=(2 * H, 2 * W)))
    assert record["reference"]["iou_first_frame"] == 1.0 and record["reference"]["cropped"] == 0.0
    # landscape for the portrait generation: core keeps the middle 240 of its 480 columns
    x, y = scail2.center_crop(2 * W, H, W, H)
    assert (x, y) == (W // 2, 0)
    _, record = mask_run(masks, pose_data, person(x1=W // 2 + 60, size=(H, 2 * W)))
    assert record["reference"]["iou_first_frame"] == 1.0 and record["reference"]["cropped"] == 0.0
    _, record = mask_run(masks, pose_data, person(x1=W // 2 - 40, size=(H, 2 * W)))
    assert record["reference"]["cropped"] == pytest.approx(0.4) and record["reference"]["flags"] == ["reference_misaligned"]


def test_an_empty_reference_or_frame_0_is_not_measured():
    masks, pose_data = clip()
    report, record = mask_run(masks, pose_data, torch.zeros(1, H, W))
    assert record["reference"] == {"area": 0.0, "cropped": 0.0, "iou_first_frame": None, "scale_first_frame": None,
                                   "flags": []}
    assert "IoU with mask frame 0 n/a" in report and "SAM 3.1 Multiplex found no person on it" in report
    masks[0] = 0
    _, record = mask_run(masks, None, person())
    assert record["reference"]["iou_first_frame"] is None and record["reference"]["flags"] == []


def test_a_zero_frame_clip_has_nothing_to_compare():
    masks, pose_data = clip()
    _, record = mask_run(masks[:0], None, person())
    assert record["frames"] == [] and record["reference"]["iou_first_frame"] is None


def test_the_reference_mask_must_be_a_mask():
    masks, pose_data = clip()
    for bad in (torch.zeros(1, 1, H, W), torch.zeros(0, H, W)):
        with pytest.raises(ValueError, match="the reference mask must be a MASK"):
            mask_run(masks, pose_data, bad)


# --- the final against the final: the WanAnimate Preprocess Guard --------------------------------

def test_the_final_mask_is_compared_with_the_reference_grown_and_blockified():
    # frame 0's rectangle as the reference: grown and blockified it is the final's frame 0 exactly;
    # the raw character against that final covers only 24000 of its 31200 px
    final, pose_data = final_clip()
    _, record = mask_run(final, pose_data, person(), final=True)
    assert record["reference"]["iou_first_frame"] == 1.0 and record["reference"]["flags"] == []
    _, raw = mask_run(final, pose_data, person())
    assert raw["reference"]["iou_first_frame"] == pytest.approx(24000 / 31200)


def test_the_final_mask_s_default_is_its_own():
    # the character 45 px right of frame 0's person: its final (columns 95-214) shares 75 of the 165
    # columns the two finals cover, an IoU of 0.455, misaligned at the final's 0.5 and not at 0.4
    final, pose_data = final_clip()
    _, record = mask_run(final, pose_data, person(x1=105), final=True)
    assert record["reference"]["iou_first_frame"] == pytest.approx(75 / 165)
    assert record["reference"]["flags"] == ["reference_misaligned"] and record["thresholds"]["min_reference_iou"] == 0.5
    _, record = mask_run(final, pose_data, person(x1=105), final=True,
                         config=guard.FinalReferenceGuardConfig(min_reference_iou=0.4))
    assert record["reference"]["flags"] == []
    with pytest.raises(TypeError, match="expected a FinalReferenceGuardConfig"):
        mask_run(final, pose_data, person(), final=True, config=guard.ReferenceGuardConfig())
    with pytest.raises(TypeError, match="expected a ReferenceGuardConfig"):
        mask_run(final, pose_data, person(), config=guard.FinalReferenceGuardConfig())


def test_a_shifted_reference_is_misaligned_on_the_final_mask():
    # the character at columns 140-239: its final is columns 130-239 (the frame edge stops the grow),
    # 40 of the 190 columns the two finals cover between them
    final, pose_data = final_clip()
    _, record = mask_run(final, pose_data, person(x1=140), final=True)
    assert record["reference"]["iou_first_frame"] == pytest.approx(40 / 190)
    assert record["reference"]["flags"] == ["reference_misaligned"]


# --- the metrics and the report ------------------------------------------------------------------

@pytest.mark.parametrize("with_pose", [True, False])
def test_without_a_reference_the_outputs_are_as_they_were(with_pose):
    masks, pose_data = clip()
    pose_data = pose_data if with_pose else None
    plain = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    configured = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False,
                                  reference_config=guard.ReferenceGuardConfig(min_reference_iou=0.9))
    assert plain[1:3] == configured[1:3] and torch.equal(plain[3], configured[3])
    record = json.loads(plain[2])
    assert list(record) == KEYS and "min_reference_iou" not in record["thresholds"]
    assert "reference image:" not in plain[1]


def test_with_a_reference_the_metrics_only_gain_keys():
    masks, pose_data = clip()
    plain_report, plain = mask_run(masks, pose_data)
    report, record = mask_run(masks, pose_data, person(x1=140))
    assert list(record) == ["guard", "thresholds", "enabled", "flags", "reference", "frames"]
    assert tuple(record["reference"]) == guard.MASK_REFERENCE
    assert record["thresholds"] == {**plain["thresholds"], "min_reference_iou": 0.4}
    assert list(record["thresholds"])[:-1] == list(plain["thresholds"])
    assert {k: v for k, v in record.items() if k != "reference"} == {**plain, "thresholds": record["thresholds"]}
    assert report.startswith(plain_report + "\nreference image: ")


def test_the_preprocess_guard_s_metrics_carry_the_reference():
    final, pose_data = final_clip()
    pose_metrics = guard.check_pose(pose_data, POSE, stop_on_fail=False)[2]
    plain_metrics = guard.check_mask(final, pose_data, MASK, stop_on_fail=False, final=True)[2]
    mask_metrics = guard.check_mask(final, pose_data, MASK, stop_on_fail=False, final=True, reference=person(x1=140))[2]
    plain_report, plain, _ = guard.combine_guards(pose_metrics, plain_metrics, stop_on_fail=False)
    report, metrics, _ = guard.combine_guards(pose_metrics, mask_metrics, stop_on_fail=False)
    record, plain = json.loads(metrics), json.loads(plain)
    assert list(plain) == KEYS[1:]
    assert list(record) == ["thresholds", "enabled", "flags", "reference", "frames"]
    assert record["reference"] == json.loads(mask_metrics)["reference"]
    assert record["reference"]["flags"] == ["reference_misaligned"]
    assert record["thresholds"] == {**plain["thresholds"], "min_reference_iou": 0.5}
    assert report.startswith(plain_report + "\nreference image: ") and "- reference_misaligned (warning)" in report
    assert report.startswith("Preprocess guard: passed")


def test_both_guards_place_and_measure_a_reference_alike():
    # the SCAIL-2 guard's reference measurements and the Mask Guard's are one function: the same
    # character against the same first frame measures the same, crop included
    masks, pose_data = clip()
    for reference in (person(x1=140), person(width=50, height=120), person(x1=W // 2 - 40, size=(H, 2 * W))):
        _, record = mask_run(masks, pose_data, reference)
        colored = scail2.colored_masks(masks, True, reference)
        scail2_record = json.loads(scail2.check_scail2(*colored, stop_on_fail=False)[3])["reference"]
        assert {key: scail2_record[key] for key in record["reference"] if key != "flags"} == \
               {key: value for key, value in record["reference"].items() if key != "flags"}
        assert scail2_record["flags"] == record["reference"]["flags"] == ["reference_misaligned"]
