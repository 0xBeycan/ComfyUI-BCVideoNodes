"""The SCAIL-2 guard on synthetic clips: guard_fakes.clip()'s drifting person rendered through the
SCAIL-2 colored masks. A clean clip passes in both modes, each injected fault fires its own check
on its own frames only, the mode comes from the reference border, and the crop geometry is core's.
The crop test reads comfy.utils, so this runs where ComfyUI is importable:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/pipelines/test_guard_scail2.py
"""
import json

import pytest
import torch

from guard_fakes import H, N, W, clip
from scail2_fakes import scail2


def rendered(replacement_mode, driving=None, reference=None):
    """(pose_video_mask, reference_image_mask) as SCAIL-2 Colored Mask renders them: the clip's
    person, and her first frame as the reference, unless given."""
    masks = clip()[0]
    driving = masks if driving is None else driving
    reference = masks[:1] if reference is None else reference
    return scail2.colored_masks(driving, replacement_mode, reference)


def reference_only(mask, replacement_mode):
    """A colored reference mask of `mask` [1, H', W'] on the mode's background, empty masks too."""
    return scail2.render_identity(mask, scail2.PALETTE[0], scail2.backgrounds(replacement_mode)[1])


def guard_run(pose_video_mask, reference_image_mask, enabled=True, config=None):
    """(passed, flags, reference record, report, metrics record), without stopping."""
    out = scail2.check_scail2(pose_video_mask, reference_image_mask, config, enabled=enabled, stop_on_fail=False)
    assert out[0] is pose_video_mask and out[1] is reference_image_mask
    report, metrics, timeline = out[2:]
    assert timeline.shape[0] == 1 and timeline.shape[-1] == 3
    record = json.loads(metrics)
    return report.startswith("SCAIL-2 guard: passed"), record["flags"], record["reference"], report, record


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_a_clean_clip_passes(replacement_mode):
    passed, flags, reference, report, _ = guard_run(*rendered(replacement_mode))
    assert passed and flags == {} and reference["flags"] == [], report
    assert reference["mode"] == (scail2.REPLACEMENT if replacement_mode else scail2.ANIMATION)
    assert reference["iou_first_frame"] == 1.0 and reference["scale_first_frame"] == 1.0 and reference["cropped"] == 0.0


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_blank_driving_frames_are_driving_empty(replacement_mode):
    masks = clip()[0]
    masks[[5, 6, 20]] = 0
    passed, flags, _, report, _ = guard_run(*rendered(replacement_mode, driving=masks))
    assert passed and flags == {"driving_empty": [5, 6, 20]}, report
    assert "- driving_empty (warning): 3 frame(s), longest run 2: 5-6, 20" in report


def test_a_detached_region_is_driving_fragmented_and_a_speck_is_data():
    masks = clip()[0]
    masks[12, 280:320, 200:240] = 1.0   # 1600 px, 6.7% of the person: a second object
    masks[14, 300:316, 220:236] = 1.0   # 256 px, 1.1%: a speck
    passed, flags, _, report, record = guard_run(*rendered(False, driving=masks))
    assert passed and flags == {"driving_fragmented": [12]}, report
    assert record["frames"][14]["fragments"] == [round(256 / 24000, 4)]


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_no_person_on_any_driving_frame_fails(replacement_mode):
    driving, reference = rendered(replacement_mode, driving=torch.zeros(N, H, W))
    passed, flags, _, report, _ = guard_run(driving, reference)
    assert not passed and flags == {"no_driving_person": list(range(N))}, report
    with pytest.raises(scail2.GuardFailed, match="no_driving_person"):
        scail2.check_scail2(driving, reference)


def test_an_empty_reference_mask_fails():
    driving, _ = rendered(False)
    passed, flags, reference, report, _ = guard_run(driving, reference_only(torch.zeros(1, H, W), False))
    assert not passed and flags == {} and reference["flags"] == ["reference_empty"], report
    assert reference["mode"] == scail2.ANIMATION and reference["iou_first_frame"] is None
    assert "- reference_empty (fail): the reference mask has no character" in report


def test_a_split_reference_mask_is_reference_fragmented():
    reference = clip()[0][:1]
    reference[0, 280:320, 200:240] = 1.0
    passed, flags, record, report, _ = guard_run(*rendered(False, reference=reference))
    assert passed and flags == {} and record["flags"] == ["reference_fragmented"], report


def test_a_reference_the_center_crop_cuts_is_reference_cropped():
    # a landscape reference for the portrait generation: core keeps its middle half
    reference = torch.zeros(1, H, 2 * W)
    reference[0, 40:280, 60:160] = 1.0
    x, y = scail2.center_crop(2 * W, H, W, H)
    assert (x, y) == (W // 2, 0)
    passed, _, record, report, _ = guard_run(*rendered(False, reference=reference))
    assert record["cropped"] == pytest.approx((W // 2 - 60) / 100) and record["flags"] == ["reference_cropped"], report
    assert passed


def test_the_cropped_threshold_is_the_config_s():
    reference = torch.zeros(1, H, 2 * W)
    reference[0, 40:280, 60:160] = 1.0
    config = scail2.SCAIL2GuardConfig(max_reference_cropped=0.7)
    _, _, record, _, metrics = guard_run(*rendered(False, reference=reference), config=config)
    assert record["flags"] == [] and metrics["thresholds"]["max_reference_cropped"] == 0.7


@pytest.mark.parametrize("replacement_mode, flagged", [(False, []), (True, ["reference_misaligned"])])
def test_a_reference_not_placed_like_the_first_frame_is_misaligned_in_replacement_mode_only(replacement_mode, flagged):
    reference = torch.zeros(1, H, W)
    reference[0, 40:280, 140:240] = 1.0   # frame 0's person is at 60:160
    passed, _, record, report, _ = guard_run(*rendered(replacement_mode, reference=reference))
    assert record["iou_first_frame"] == pytest.approx(20 / 180) and record["flags"] == flagged, report
    assert passed


def test_the_reference_scale_is_the_height_ratio_to_the_first_frame():
    reference = torch.zeros(1, H, W)
    reference[0, 100:220, 60:160] = 1.0
    _, _, record, _, _ = guard_run(*rendered(False, reference=reference))
    assert record["scale_first_frame"] == pytest.approx(120 / 240)


def test_the_latent_cut_loses_a_thin_limb_on_black():
    masks = torch.zeros(3, H, W)
    masks[:, 40:280, 60:160] = 1.0
    masks[1] = 0
    masks[1, 40:280, 100] = 1.0           # one pixel wide
    _, _, _, _, record = guard_run(*rendered(False, driving=masks))
    kept = [row["latent_kept"] for row in record["frames"]]
    assert kept == [1.0, 0.0, 1.0]


def test_an_empty_frame_has_no_latent_kept_and_no_iou_change_from_empty():
    masks = clip()[0]
    masks[[3, 4]] = 0
    _, _, _, _, record = guard_run(*rendered(False, driving=masks))
    frames = record["frames"]
    assert frames[3]["latent_kept"] is None and frames[3]["mask_iou_prev"] == 0.0 and frames[4]["mask_iou_prev"] == 1.0
    assert frames[0]["mask_iou_prev"] is None


def test_switched_off_it_measures_and_never_stops():
    driving, reference = rendered(False, driving=torch.zeros(N, H, W))
    out = scail2.check_scail2(driving, reference_only(torch.zeros(1, H, W), False), enabled=False)
    report, record = out[2], json.loads(out[3])
    assert report.startswith("SCAIL-2 guard: passed") and record["enabled"] == []
    assert "- no_driving_person (off)" in report and "- reference_empty (off)" in report


def test_stop_on_fail_off_returns_the_failed_report():
    driving, reference = rendered(False, driving=torch.zeros(N, H, W))
    out = scail2.check_scail2(driving, reference, stop_on_fail=False)
    assert out[2].startswith("SCAIL-2 guard: FAILED")


def test_a_zero_frame_clip():
    driving, reference = rendered(False)
    passed, flags, record, report, metrics = guard_run(driving[:0], reference)
    assert passed and flags == {} and metrics["frames"] == [] and record["iou_first_frame"] is None, report


def test_the_metrics_record():
    _, _, _, _, record = guard_run(*rendered(True))
    assert list(record) == ["guard", "thresholds", "enabled", "flags", "reference", "frames"]
    assert record["guard"] == "scail2"
    assert record["thresholds"] == {"max_reference_cropped": 0.02, "min_reference_iou": 0.4}
    assert record["enabled"] == sorted(scail2.SCAIL2_CHECKS)
    assert tuple(record["reference"]) == scail2.SCAIL2_REFERENCE
    assert all(tuple(row) == scail2.SCAIL2_ROW for row in record["frames"]) and len(record["frames"]) == N


def test_the_warnings_and_the_failures():
    warnings = {"driving_empty", "driving_fragmented", "reference_fragmented", "reference_cropped", "reference_misaligned"}
    assert set(scail2.SCAIL2_CHECKS) - scail2.WARNINGS == {"no_driving_person", "reference_empty"}
    assert set(scail2.SCAIL2_CHECKS) & scail2.WARNINGS == warnings


def test_combine_guards_does_not_take_the_scail2_metrics():
    metrics = scail2.check_scail2(*rendered(False))[3]
    with pytest.raises(ValueError, match="expected the metrics of check_pose and check_mask"):
        scail2.combine_guards(metrics, metrics)


@pytest.mark.parametrize("name, shape", [("pose_video_mask", (N, H, W)), ("reference_image_mask", (1, H, W, 1))])
def test_a_tensor_that_is_not_a_colored_mask_is_an_error(name, shape):
    driving, reference = rendered(False)
    inputs = {"pose_video_mask": driving, "reference_image_mask": reference, name: torch.zeros(shape)}
    with pytest.raises(ValueError, match=f"{name} must be a colored mask IMAGE"):
        scail2.check_scail2(**inputs)


def test_a_reference_without_a_frame_is_an_error():
    driving, reference = rendered(False)
    with pytest.raises(ValueError, match="reference_image_mask has no frame"):
        scail2.check_scail2(driving, reference[:0])


@pytest.mark.parametrize("size, target", [((320, 240), (240, 320)), ((240, 320), (320, 240)), ((333, 200), (96, 160)),
                                          ((64, 64), (64, 32)), ((100, 300), (100, 300))])
def test_the_crop_geometry_is_common_upscale_s(size, target):
    cli_args = pytest.importorskip("comfy.cli_args")
    cli_args.args.disable_xformers = True
    import comfy.utils

    (height, width), (new_height, new_width) = size, target
    image = torch.rand(1, 3, height, width, generator=torch.Generator().manual_seed(0))
    x, y = scail2.center_crop(width, height, new_width, new_height)
    ours = torch.nn.functional.interpolate(image[..., y:height - y, x:width - x], size=(new_height, new_width),
                                           mode="nearest-exact")
    assert torch.equal(ours, comfy.utils.common_upscale(image, new_width, new_height, "nearest-exact", "center"))
