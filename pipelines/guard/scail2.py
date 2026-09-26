"""The SCAIL-2 checks: the colored driving and reference masks the SCAIL-2 sampler reads, and
check_scail2. End-to-end SCAIL-2 draws no pose, so pose_data is optional: without it the driving
mask gets the checks that need no pose; with it (Pose Detection on the driving frames at the
generation size) it also gets the Mask Guard's pose-based mask checks.

The person is what core's WanSCAILToVideo reads as identity 0: the pixels whose blue channel is
above 225/255 and whose red and green are not. The mode is read from the reference mask's border,
as the sampler reads it. The driving mask is taken to be at the generation size, as the sampler
wants it; the reference is center-cropped to the generation's aspect ratio by core, which the
reference measurements repeat.

On SCAIL-2's own examples blank frames and a split-up mask are normal (the person leaves the
shot, a passer-by occludes her), so only two things stop:

  no_driving_person     no driving frame has the person
  reference_empty       the reference mask has no character

and the rest are warnings:

  driving_empty         a driving frame without the person
  driving_fragmented    a detached region at least 5% of the largest one on a driving frame
                        (without a pose a piece of the person cannot be told from another object;
                        with one, a piece holding her own drawn keypoints is her)
  mask_loss             a region the driving mask drops for a run of up to 8 frames between two
                        frames that hold it, not a limb that moved away (see the Mask Guard)
  reference_fragmented  the same on the reference mask
  reference_misaligned  replacement mode only: the character on the reference, cropped and
                        resized as core does, overlaps the person on the first driving frame by
                        an IoU below `min_reference_iou` (the authors expect the reference posed
                        like the first driving frame)

With pose_data, the driving mask also gets mask_empty and mask_leak (these two stop, as in the
Mask Guard), mask_attached_leak, mask_missing_keypoints, mask_missed_limb, body_not_drawn and
mask_unstable (warnings), measured and judged as the Mask Guard does.

Measured as data, never judged: the mask area, the share of the mask the latent cut keeps
(`latent_kept`: core reads the driving mask at half size, area-resized, cut at 225/255, so a
thin limb can vanish), the mask IoU with the previous frame, the share of the reference character
core's center crop cuts off (`cropped`), and the reference's IoU and scale against the first
driving frame in both modes.
"""
import json
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from ...libs import log
from ...models.scail2.adapter import ON, REPLACEMENT, mask_convention
from .common import (FRAGMENT_FRACTION, MASK_CHECKS, POSE_FREE_MASK_CHECKS, SCAIL2_CHECKS, SCAIL2_DRIVING_CHECKS,
                     SCAIL2_ROW, WARNINGS, Scail2Reference, Scail2Row, _flag, _thresholds)
from .config import MaskGuardConfig, SCAIL2GuardConfig, _config
from .mask import NO_KEYPOINTS, _iou, mask_flags, mask_frame_metrics, mask_regions, pose_of
from .report import _kind, _stop, write_report
from .timeline import SCAIL2_PANELS, timeline_image


def _person(rgb):
    """The identity-0 (blue) pixels of a colored mask [..., 3], as core's extraction reads them."""
    return (rgb[..., 2] > ON) & (rgb[..., 0] <= ON) & (rgb[..., 1] <= ON)


def _colored(name, image):
    """`image` checked as a colored mask IMAGE [frames, height, width, 3+]."""
    if image.dim() != 4 or image.shape[-1] < 3:
        raise ValueError(f"{name} must be a colored mask IMAGE [frames, height, width, 3], got a tensor of shape "
                         f"{tuple(image.shape)}; connect the {name} output of SCAIL-2 Preprocess or SCAIL-2 Colored Mask")
    return image


def center_crop(width, height, new_width, new_height):
    """(x, y): the columns and the rows comfy.utils.common_upscale's center crop cuts off each
    side of a `width` x `height` image to reach the aspect ratio of `new_width` x `new_height`."""
    old_aspect, new_aspect = width / height, new_width / new_height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((width - width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((height - height * (old_aspect / new_aspect)) / 2)
    return x, y


def _latent_kept(frame, person, area):
    """The share of the person's `area` pixels on `frame` [H, W, 3] that core's driving-mask
    path keeps: the frame area-resized to half size, then cut at 225/255. None on an empty frame."""
    if not area:
        return None
    H, W = person.shape
    half = F.interpolate(frame.movedim(-1, 0)[None], size=(H // 2, W // 2), mode="area")[0].movedim(0, -1)
    kept = F.interpolate(_person(half)[None, None].float(), size=(H, W), mode="nearest")[0, 0] > 0.5
    return float((kept & person).sum()) / area


def driving_person(pose_video_mask):
    """The person on every driving frame as [T, H, W] booleans, and the share of her each frame
    keeps through the latent cut (`_latent_kept`)."""
    T, H, W = pose_video_mask.shape[:3]
    masks = np.zeros((T, H, W), dtype=bool)
    kept = []
    for i in range(T):
        frame = pose_video_mask[i, ..., :3].float().cpu()
        person = _person(frame)
        masks[i] = person.numpy()
        kept.append(_latent_kept(frame, person, int(masks[i].sum())))
    return masks, kept


def scail2_frame_metrics(masks, kept, max_mask_loss, pose_metas=None, detections=None,
                         draw_threshold=None) -> list[Scail2Row]:
    """One dict of raw measurements per driving frame (keys SCAIL2_ROW): the Mask Guard's
    measurements of the person (`mask_frame_metrics`, the pose-based ones None without a pose)
    and the share the latent cut keeps."""
    rows = mask_frame_metrics(masks, pose_metas, detections, draw_threshold, max_mask_loss)
    return [{key: kept[m["frame"]] if key == "latent_kept" else m[key] for key in SCAIL2_ROW} for m in rows]


def scail2_flags(rows: list[Scail2Row], t):
    """The driving-frame checks: check name -> frames it fired on, in the order the checks first
    fired (SCAIL2_DRIVING_CHECKS breaks a tie). A clip without the person on any frame is
    no_driving_person on every frame, and nothing else. `t` has the mask thresholds."""
    empty = [m["frame"] for m in rows if m["mask_area"] == 0]
    if rows and len(empty) == len(rows):
        return {"no_driving_person": empty}
    flags = {}
    for m in rows:
        if m["mask_area"] == 0:
            _flag(flags, "driving_empty", m["frame"])
        if any(f >= FRAGMENT_FRACTION for f in m["fragments"]):
            _flag(flags, "driving_fragmented", m["frame"])
    borrowed = {name: frames for name, frames in mask_flags(rows, t).items() if name in SCAIL2_DRIVING_CHECKS}
    both = {**flags, **borrowed}
    return {name: both[name] for name in sorted(both, key=lambda name: (both[name][0], SCAIL2_DRIVING_CHECKS.index(name)))}


def _height(mask):
    """The number of rows from the top to the bottom of `mask`'s pixels."""
    rows = np.flatnonzero(mask.any(axis=1))
    return int(rows[-1] - rows[0] + 1)


def reference_record(reference_image_mask, width, height, first_frame, t) -> Scail2Reference:
    """The measurements and the flags of the reference mask's first frame against a `width` x
    `height` generation and the person on the first driving frame (`first_frame`, [H, W]
    booleans, None without driving frames). `t` has the thresholds."""
    mode = mask_convention(reference_image_mask)
    person = _person(reference_image_mask[0, ..., :3].float().cpu()).numpy()
    Hr, Wr = person.shape
    area = int(person.sum())
    fragments = mask_regions(person, NO_KEYPOINTS)[1] if area else []
    # what core makes of it: the center crop to the generation's aspect, resized nearest-exact
    x, y = center_crop(Wr, Hr, width, height)
    kept = person[y:Hr - y, x:Wr - x]
    cropped = 1.0 - float(kept.sum()) / area if area else 0.0
    placed = F.interpolate(torch.from_numpy(kept)[None, None].float(), size=(height, width), mode="nearest-exact")[0, 0].numpy() > 0.5
    against = area and first_frame is not None and first_frame.any() and placed.any()
    record = {"mode": mode, "area": area / (Hr * Wr), "fragments": fragments, "cropped": cropped,
              "iou_first_frame": _iou(placed, first_frame) if against else None,
              "scale_first_frame": _height(placed) / _height(first_frame) if against else None}
    flags = []
    if not area:
        flags.append("reference_empty")
    if any(f >= FRAGMENT_FRACTION for f in fragments):
        flags.append("reference_fragmented")
    if mode == REPLACEMENT and record["iou_first_frame"] is not None and record["iou_first_frame"] < t["min_reference_iou"]:
        flags.append("reference_misaligned")
    record["flags"] = flags
    return record


def _value(value):
    return "n/a" if value is None else f"{value:.3f}"


# What each reference check's report line says, from the reference record.
REFERENCE_LINES = {
    "reference_empty": "the reference mask has no character (blue) pixel; connect a MASK of the character on the "
                       "reference image (SCAIL-2 Preprocess: a prompt that finds it, or reference_mask)",
    "reference_fragmented": "a detached region of {largest:.3f} of the character's largest one",
    "reference_misaligned": "IoU {iou} with the person on the first driving frame; replacement mode expects the "
                            "reference posed and placed like the first driving frame",
}


def scail2_report(rows: list[Scail2Row], flags, reference: Scail2Reference, enabled, posed):
    """The report text and whether the enabled checks all passed: the driving-frame checks as
    the other guards report them, then the reference measurements and checks."""
    driving, driving_passed = write_report("driving mask", rows, flags, enabled)
    if not posed:
        driving += "\n- without pose_data, not checked: " + ", ".join(
            name for name in SCAIL2_DRIVING_CHECKS if name in MASK_CHECKS and name not in POSE_FREE_MASK_CHECKS)
    failed = [name for name in reference["flags"] if name in enabled and name not in WARNINGS]
    passed = driving_passed and not failed
    mode = reference["mode"] or "unclear"
    lines = [f"SCAIL-2 guard: {'passed' if passed else 'FAILED'} - {mode} mode (the reference mask's border)", driving,
             f"reference mask: area {_value(reference['area'])}, fragments {reference['fragments']}, "
             f"cropped {_value(reference['cropped'])}, IoU with driving frame 0 {_value(reference['iou_first_frame'])}, "
             f"scale vs driving frame 0 {_value(reference['scale_first_frame'])}"]
    values = {"largest": max(reference["fragments"], default=0.0), "iou": _value(reference["iou_first_frame"])}
    for name in reference["flags"]:
        lines.append(f"- {name} ({_kind(name, enabled)}): " + REFERENCE_LINES[name].format(**values))
    return "\n".join(lines), passed


def check_scail2(pose_video_mask, reference_image_mask, config=None, enabled=True, stop_on_fail=True, pose_data=None,
                 mask_config=None):
    """The SCAIL-2 checks on the colored driving mask [frames, H, W, 3] (at the generation size)
    and the colored reference mask [N, H', W', 3], as SCAIL-2 Preprocess or SCAIL-2 Colored Mask
    render them, and with `pose_data` (Pose Detection on the same driving frames at the same
    size) the Mask Guard's pose-based checks on the driving mask.

    `config` is a SCAIL2GuardConfig and `mask_config` a MaskGuardConfig (None = defaults).
    `enabled` False still measures and reports every check but marks them off, so none can fail.
    With `stop_on_fail` a failed enabled check raises GuardFailed with the report.

    Returns (pose_video_mask unchanged, reference_image_mask unchanged, report, metrics JSON,
    timeline IMAGE). The metrics are {"guard": "scail2", "thresholds", "enabled", "flags" (the
    driving-frame checks, name -> frames), "reference" (the reference record, its checks in its
    "flags"), "frames"}."""
    config = _config(config, SCAIL2GuardConfig)
    mask_config = _config(mask_config, MaskGuardConfig)
    _colored("pose_video_mask", pose_video_mask)
    _colored("reference_image_mask", reference_image_mask)
    if reference_image_mask.shape[0] == 0:
        raise ValueError("reference_image_mask has no frame; connect the reference_image_mask output of SCAIL-2 "
                         "Preprocess or SCAIL-2 Colored Mask")
    thresholds = {**asdict(config), **_thresholds(pose_data, mask_config)}
    checks = set(SCAIL2_CHECKS if enabled else ())
    T, H, W = pose_video_mask.shape[:3]
    with log.step(f"SCAIL-2 guard: checking {T} frames ({'on' if enabled else 'off'})"):
        masks, kept = driving_person(pose_video_mask)
        pose_metas, detections = pose_of(masks, pose_data, "run Pose Detection on the driving frames at the "
                                                           "generation size, the frames SCAIL-2 Preprocess got")
        rows = scail2_frame_metrics(masks, kept, mask_config.max_mask_loss, pose_metas, detections,
                                    thresholds["draw_threshold"])
        flags = scail2_flags(rows, thresholds)
        reference = reference_record(reference_image_mask, W, H, masks[0] if T else None, thresholds)
    report, passed = scail2_report(rows, flags, reference, checks, pose_data is not None)
    metrics = json.dumps({"guard": "scail2", "thresholds": thresholds, "enabled": sorted(checks), "flags": flags,
                          "reference": reference, "frames": rows})
    timeline = timeline_image(rows, flags, SCAIL2_PANELS)
    _stop(report, passed, stop_on_fail)
    return pose_video_mask, reference_image_mask, report, metrics, timeline
