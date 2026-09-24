"""The SCAIL-2 checks: the colored driving and reference masks the SCAIL-2 sampler reads, judged
without a pose (end-to-end SCAIL-2 draws none), and check_scail2.

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
                        (without a pose a piece of the person cannot be told from another object)
  reference_fragmented  the same on the reference mask
  reference_cropped     more than `max_reference_cropped` of the character falls outside the
                        center crop to the generation's aspect ratio (a portrait reference in a
                        landscape generation loses the head or the feet)
  reference_misaligned  replacement mode only: the character on the reference, cropped and
                        resized as core does, overlaps the person on the first driving frame by
                        an IoU below `min_reference_iou` (the authors expect the reference posed
                        like the first driving frame)

Measured as data, never judged: the mask area, the share of the mask the latent cut keeps
(`latent_kept`: core reads the driving mask at half size, area-resized, cut at 225/255, so a
thin limb can vanish), the mask IoU with the previous frame, and the reference's IoU and scale
against the first driving frame in both modes. The thresholds are first values, not calibrated.
"""
import json
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from ...libs import log
from ...models.scail2.adapter import ON, REPLACEMENT, mask_convention
from .common import FRAGMENT_FRACTION, SCAIL2_CHECKS, WARNINGS, Scail2Reference, Scail2Row, _flag
from .config import SCAIL2GuardConfig, _config
from .mask import _iou, mask_regions
from .report import _kind, _stop, write_report
from .timeline import SCAIL2_PANELS, timeline_image

NO_KEYPOINTS = np.empty((0, 2))  # without a pose, the largest region is the person


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


def scail2_frame_metrics(pose_video_mask) -> list[Scail2Row]:
    """One dict of raw measurements per frame of the colored driving mask (keys SCAIL2_ROW)."""
    T, H, W = pose_video_mask.shape[:3]
    rows = []
    prev = None
    for i in range(T):
        frame = pose_video_mask[i, ..., :3].float().cpu()
        person = _person(frame)
        mask = person.numpy()
        area = int(mask.sum())
        rows.append({"frame": i, "mask_area": area / (H * W),
                     "fragments": mask_regions(mask, NO_KEYPOINTS)[1] if area else [],
                     "latent_kept": _latent_kept(frame, person, area),
                     "mask_iou_prev": _iou(mask, prev) if prev is not None else None})
        prev = mask
    return rows


def scail2_flags(rows: list[Scail2Row]):
    """The driving-frame checks: check name -> frames it fired on, in the order the checks first
    fired. A clip without the person on any frame is no_driving_person on every frame, and
    nothing else."""
    empty = [m["frame"] for m in rows if m["mask_area"] == 0]
    if rows and len(empty) == len(rows):
        return {"no_driving_person": empty}
    flags = {}
    for m in rows:
        if m["mask_area"] == 0:
            _flag(flags, "driving_empty", m["frame"])
        if any(f >= FRAGMENT_FRACTION for f in m["fragments"]):
            _flag(flags, "driving_fragmented", m["frame"])
    return flags


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
    if cropped > t["max_reference_cropped"]:
        flags.append("reference_cropped")
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
    "reference_cropped": "{cropped:.3f} of the character falls outside the center crop to the generation's aspect "
                         "ratio; give the reference the aspect ratio of the generation",
    "reference_misaligned": "IoU {iou} with the person on the first driving frame; replacement mode expects the "
                            "reference posed and placed like the first driving frame",
}


def scail2_report(rows: list[Scail2Row], flags, reference: Scail2Reference, enabled):
    """The report text and whether the enabled checks all passed: the driving-frame checks as
    the other guards report them, then the reference measurements and checks."""
    driving, driving_passed = write_report("driving mask", rows, flags, enabled)
    failed = [name for name in reference["flags"] if name in enabled and name not in WARNINGS]
    passed = driving_passed and not failed
    mode = reference["mode"] or "unclear"
    lines = [f"SCAIL-2 guard: {'passed' if passed else 'FAILED'} - {mode} mode (the reference mask's border)", driving,
             f"reference mask: area {_value(reference['area'])}, fragments {reference['fragments']}, "
             f"cropped {_value(reference['cropped'])}, IoU with driving frame 0 {_value(reference['iou_first_frame'])}, "
             f"scale vs driving frame 0 {_value(reference['scale_first_frame'])}"]
    values = {"largest": max(reference["fragments"], default=0.0), "cropped": reference["cropped"],
              "iou": _value(reference["iou_first_frame"])}
    for name in reference["flags"]:
        lines.append(f"- {name} ({_kind(name, enabled)}): " + REFERENCE_LINES[name].format(**values))
    return "\n".join(lines), passed


def check_scail2(pose_video_mask, reference_image_mask, config=None, enabled=True, stop_on_fail=True):
    """The SCAIL-2 checks on the colored driving mask [frames, H, W, 3] (at the generation size)
    and the colored reference mask [N, H', W', 3], as SCAIL-2 Preprocess or SCAIL-2 Colored Mask
    render them.

    `config` is a SCAIL2GuardConfig (None = defaults). `enabled` False still measures and
    reports every check but marks them off, so none can fail. With `stop_on_fail` a failed
    enabled check raises GuardFailed with the report.

    Returns (pose_video_mask unchanged, reference_image_mask unchanged, report, metrics JSON,
    timeline IMAGE). The metrics are {"guard": "scail2", "thresholds", "enabled", "flags" (the
    driving-frame checks, name -> frames), "reference" (the reference record, its checks in its
    "flags"), "frames"}."""
    config = _config(config, SCAIL2GuardConfig)
    _colored("pose_video_mask", pose_video_mask)
    _colored("reference_image_mask", reference_image_mask)
    if reference_image_mask.shape[0] == 0:
        raise ValueError("reference_image_mask has no frame; connect the reference_image_mask output of SCAIL-2 "
                         "Preprocess or SCAIL-2 Colored Mask")
    thresholds = asdict(config)
    checks = set(SCAIL2_CHECKS if enabled else ())
    T, H, W = pose_video_mask.shape[:3]
    with log.step(f"SCAIL-2 guard: checking {T} frames ({'on' if enabled else 'off'})"):
        rows = scail2_frame_metrics(pose_video_mask)
        flags = scail2_flags(rows)
        first = _person(pose_video_mask[0, ..., :3].float().cpu()).numpy() if T else None
        reference = reference_record(reference_image_mask, W, H, first, thresholds)
    report, passed = scail2_report(rows, flags, reference, checks)
    metrics = json.dumps({"guard": "scail2", "thresholds": thresholds, "enabled": sorted(checks), "flags": flags,
                          "reference": reference, "frames": rows})
    timeline = timeline_image(rows, flags, SCAIL2_PANELS)
    _stop(report, passed, stop_on_fail)
    return pose_video_mask, reference_image_mask, report, metrics, timeline
