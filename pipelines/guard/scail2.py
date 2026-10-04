"""The SCAIL-2 checks: the colored driving and reference masks the SCAIL-2 sampler reads, and
check_scail2. End-to-end SCAIL-2 draws no pose, so pose_data is optional: without it the driving
mask gets the checks that need no pose; with it (Pose Detection on the driving frames at the
generation size) it also gets the Mask Guard's pose-based mask checks.

The person is what core's WanSCAILToVideo reads as identity 0: the pixels whose blue channel is
above 225/255 and whose red and green are not. The mode is read from the reference mask's border,
as the sampler reads it. The driving mask is taken to be at the generation size, as the sampler
wants it; the reference is center-cropped to the generation's aspect ratio by core, which the
reference measurements repeat.

The checks that stop the workflow:

  no_driving_person     no driving frame has the person
  driving_fragmented    a detached region at least 5% of the largest one on a driving frame
                        (without a pose only a piece the frame edge cut from her - one that runs
                        off a side of the frame she runs off - is told from another object; with
                        one, a piece holding her own drawn keypoints is her)
  reference_empty       the reference mask has no character (the primary, or an extra reference)

and with pose_data the Mask Guard's fails on the driving mask: mask_empty, mask_head_out,
mask_limb_out and mask_loss, judged by what the latent grid reads of her (below). The warnings:

  driving_empty         without pose_data, a driving frame without the person: normal where the
                        person leaves the shot, and only a pose tells (with it, mask_empty)
  reference_fragmented  a detached region of a reference mask (the primary, or an extra reference)
  reference_misaligned  replacement mode only: the character on the reference, cropped and
                        resized as core does, overlaps the person on the first driving frame by
                        an IoU below `min_reference_iou` (the authors expect the reference posed
                        like the first driving frame)

and with pose_data mask_leak and mask_attached_leak.

SCAIL-2 reads the driving mask coarsely and grows nothing (common.LATENT_READ): core area-resizes
it to half size and cuts it at 225/255, then area-pools that to the latent grid, one cell per
16 x 16 px of the generation. So a region of her the model loses is a whole cell of that grid
(mask_loss; `latent_reading`), and a drawn keypoint in a cell the grid reads as her is inside the
mask (mask_head_out, mask_limb_out). mask_loss needs pose_data: on that grid a limb in motion
empties whole cells of a correct mask, which only the pose tells from a part of her dropped.

The reference mask's first frame is the primary reference; its other frames are extra references
(SCAIL-2 multi-reference, SCAIL-2 Preprocess's face close-up), the character in blue on black in
both modes, each checked for reference_empty and reference_fragmented and named by its index in the
report. The mode is read from the primary's border alone, and reference_misaligned judges the
primary alone: an extra reference is another view, not placed like the first driving frame.

Measured as data, never judged: the mask area, the share of the mask the half-size cut keeps
(`latent_kept`: core reads the driving mask at half size, area-resized, cut at 225/255, so a
thin limb can vanish), the share of the reference character core's center crop cuts off
(`cropped`), and the reference's IoU and scale against the first driving frame in both modes.
"""
import json
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from ...libs import log
from ...libs.video import requantized
from ...models.scail2.adapter import ON, REPLACEMENT, mask_convention
from .common import (FRAGMENT_FRACTION, LATENT_READ, SCAIL2_CHECKS, SCAIL2_DRIVING_CHECKS, SCAIL2_POSE_FREE_CHECKS,
                     SCAIL2_ROW, WARNINGS, Scail2ExtraReference, Scail2Reference, Scail2Row, _flag, _thresholds)
from .config import MaskGuardConfig, SCAIL2GuardConfig, _config
from .mask import mask_flags, mask_frame_metrics, mask_regions, pose_of
from .reference import reference_fit
from .report import _kind, _stop, _value, write_report
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


def _half(frame):
    """The person on `frame` [H, W, 3] as core's WanSCAILToVideo takes the driving mask in: the
    frame area-resized to half size (its center crop cuts nothing at the generation size), then
    each colour channel cut at 225/255 (`_person`), [H // 2, W // 2] booleans."""
    H, W = frame.shape[:2]
    return _person(F.interpolate(frame.movedim(-1, 0)[None], size=(H // 2, W // 2), mode="area")[0].movedim(0, -1))


def _latent_kept(half, person, area):
    """The share of the person's `area` pixels (`person`, [H, W]) that the half-size cut `half`
    keeps. None on an empty frame."""
    if not area:
        return None
    kept = F.interpolate(half[None, None].float(), size=person.shape, mode="nearest")[0, 0] > 0.5
    return float((kept & person).sum()) / area


def _latent_cells(half):
    """The cells of the latent grid that read as the person, from the half-size cut `half`: core's
    _extract_mask_to_28ch area-pools it 8x (each side halved three times, rounding up), and a
    cell reads as her when she fills at least LATENT_READ of it."""
    h, w = half.shape
    for _ in range(3):
        h, w = (h + 1) // 2, (w + 1) // 2
    return F.interpolate(half[None, None].float(), size=(h, w), mode="area")[0, 0] >= LATENT_READ


def driving_person(pose_video_mask):
    """The person on every driving frame as [T, H, W] booleans, the share of her each frame
    keeps through the half-size cut (`_latent_kept`), and the cells of the latent grid each frame
    reads as her ([T, h, w] booleans, `_latent_cells`)."""
    T, H, W = pose_video_mask.shape[:3]
    masks = np.zeros((T, H, W), dtype=bool)
    kept, cells = [], []
    for i in range(T):
        frame = requantized(pose_video_mask[i, ..., :3].cpu()).float()
        person, half = _person(frame), _half(frame)
        masks[i] = person.numpy()
        kept.append(_latent_kept(half, person, int(masks[i].sum())))
        cells.append(_latent_cells(half).numpy())
    return masks, kept, np.array(cells, dtype=bool) if T else np.zeros((0, 0, 0), bool)


def _cell_edges(cells, size):
    """The edges of the cells of one side of the latent grid, `cells` of them, on a side of `size`
    px, as the nearest upsampling of `latent_reading` places them: from 0 to `size`."""
    source = np.floor(np.arange(size) * (cells / size)).astype(int)
    return np.concatenate([[0], np.flatnonzero(np.diff(source)) + 1, [size]])


def latent_reading(cells, shape):
    """What SCAIL-2 reads of each driving frame, as `read(f)` (mask.dropouts): the cells of frame
    f's latent grid that read as the person (`cells`, `driving_person`) at the frame size `shape`
    (H, W), and that grid's cells as (row edges, column edges)."""
    H, W = shape
    grid = rows = cols = None
    if len(cells):
        grid = (_cell_edges(cells.shape[1], H), _cell_edges(cells.shape[2], W))
        # the cell each pixel row and each pixel column reads, as the nearest upsampling picks it
        rows, cols = (F.interpolate(torch.arange(n, dtype=torch.float32)[None, None, :, None], size=(size, 1),
                                    mode="nearest")[0, 0, :, 0].long().numpy()
                      for n, size in ((cells.shape[1], H), (cells.shape[2], W)))

    def read(f):
        return cells[f].take(cols, axis=1).take(rows, axis=0), grid
    return read


def scail2_frame_metrics(masks, kept, cells, pose_metas=None, detections=None, draw_threshold=None) -> list[Scail2Row]:
    """One dict of raw measurements per driving frame (keys SCAIL2_ROW): the Mask Guard's
    measurements of the person (`mask_frame_metrics`, the pose-based ones None without a pose),
    mask_loss and the keypoint tests judged by what the latent grid reads of her
    (`latent_reading`), and the share the half-size cut keeps. mask_loss needs a pose."""
    read = latent_reading(cells, masks.shape[1:])
    rows = mask_frame_metrics(masks, pose_metas, detections, draw_threshold, read,
                              read_keypoints=lambda f: read(f)[0], loss_without_pose=False)
    return [{key: kept[m["frame"]] if key == "latent_kept" else m[key] for key in SCAIL2_ROW} for m in rows]


def scail2_flags(rows: list[Scail2Row], t, posed):
    """The driving-frame checks: check name -> frames it fired on, in the order the checks first
    fired (SCAIL2_DRIVING_CHECKS breaks a tie). A clip without the person on any frame is
    no_driving_person on every frame, and nothing else. `t` has the mask thresholds; `posed` says
    pose_data was connected: with it an empty frame is mask_empty's, without it driving_empty."""
    empty = [m["frame"] for m in rows if m["mask_area"] == 0]
    if rows and len(empty) == len(rows):
        return {"no_driving_person": empty}
    flags = {}
    for m in rows:
        if m["mask_area"] == 0 and not posed:
            _flag(flags, "driving_empty", m["frame"])
        if any(f >= FRAGMENT_FRACTION for f in m["fragments"]):
            _flag(flags, "driving_fragmented", m["frame"])
    borrowed = mask_flags(rows, t, checks=SCAIL2_DRIVING_CHECKS)
    both = {**flags, **borrowed}
    return {name: both[name] for name in sorted(both, key=lambda name: (both[name][0], SCAIL2_DRIVING_CHECKS.index(name)))}


def reference_record(reference_image_mask, width, height, first_frame, t) -> Scail2Reference:
    """The measurements and the flags of the reference mask's first frame against a `width` x
    `height` generation and the person on the first driving frame (`first_frame`, [H, W]
    booleans, None without driving frames), placed as core places it (reference.reference_fit).
    `t` has the thresholds."""
    mode = mask_convention(reference_image_mask)
    person = _reference_person(reference_image_mask, 0)
    fit = reference_fit(person, width, height, first_frame)
    fragments = mask_regions(person, None)[1] if fit["area"] else []
    record = {"mode": mode, "area": fit["area"], "fragments": fragments, "cropped": fit["cropped"],
              "iou_first_frame": fit["iou_first_frame"], "scale_first_frame": fit["scale_first_frame"]}
    flags = _character_flags(fit["area"], fragments)
    if mode == REPLACEMENT and record["iou_first_frame"] is not None and record["iou_first_frame"] < t["min_reference_iou"]:
        flags.append("reference_misaligned")
    record["flags"] = flags
    return record


def _reference_person(reference_image_mask, index):
    """The character (blue) on frame `index` of the colored reference mask, [H', W'] booleans."""
    return _person(requantized(reference_image_mask[index, ..., :3].cpu()).float()).numpy()


def _character_flags(area, fragments):
    """The checks every reference mask gets: reference_empty without a character pixel,
    reference_fragmented with a detached region of FRAGMENT_FRACTION of the largest or more."""
    flags = ["reference_empty"] if not area else []
    if any(f >= FRAGMENT_FRACTION for f in fragments):
        flags.append("reference_fragmented")
    return flags


def extra_reference_records(reference_image_mask) -> list[Scail2ExtraReference]:
    """One record per extra reference, the reference mask's frames after the first (SCAIL-2
    multi-reference): its index in the batch, the character's share of the frame, its detached
    regions and its checks (reference_empty, reference_fragmented)."""
    records = []
    for index in range(1, reference_image_mask.shape[0]):
        person = _reference_person(reference_image_mask, index)
        area = int(person.sum())
        fragments = mask_regions(person, None)[1] if area else []
        records.append({"index": index, "area": area / person.size, "fragments": fragments,
                        "flags": _character_flags(area, fragments)})
    return records


# What each reference check's report line says, from the reference record.
REFERENCE_LINES = {
    "reference_empty": "the reference mask has no character (blue) pixel; connect a MASK of the character on the "
                       "reference image (SCAIL-2 Preprocess: a prompt that finds it, or reference_mask)",
    "extra_reference_empty": "the extra reference's mask has no character (blue) pixel, so it binds to no identity; "
                             "SCAIL-2 Preprocess's prompt found no character on its face close-up: check the prompt, "
                             "or turn face_crop off",
    "reference_fragmented": "a detached region of {largest:.3f} of the character's largest one",
    "reference_misaligned": "IoU {iou} with the person on the first driving frame; replacement mode expects the "
                            "reference posed and placed like the first driving frame",
}


def scail2_report(rows: list[Scail2Row], flags, reference: Scail2Reference, enabled, posed,
                  extras: list[Scail2ExtraReference] = ()):
    """The report text and whether the enabled checks all passed: the driving-frame checks as
    the other guards report them, then the reference measurements and checks, the primary's
    (reference 0) and each extra reference's, by its index."""
    driving, driving_passed = write_report("driving mask", rows, flags, enabled)
    if not posed:
        driving += "\n- without pose_data, not checked: " + ", ".join(
            name for name in SCAIL2_DRIVING_CHECKS if name not in SCAIL2_POSE_FREE_CHECKS)
    failed = [name for record in (reference, *extras) for name in record["flags"] if name in enabled and name not in WARNINGS]
    passed = driving_passed and not failed
    mode = reference["mode"] or "unclear"
    lines = [f"SCAIL-2 guard: {'passed' if passed else 'FAILED'} - {mode} mode (the primary reference mask's border)", driving,
             f"reference 0 mask (primary): area {_value(reference['area'])}, fragments {reference['fragments']}, "
             f"cropped {_value(reference['cropped'])}, IoU with driving frame 0 {_value(reference['iou_first_frame'])}, "
             f"scale vs driving frame 0 {_value(reference['scale_first_frame'])}"]
    values = {"largest": max(reference["fragments"], default=0.0), "iou": _value(reference["iou_first_frame"])}
    for name in reference["flags"]:
        lines.append(f"- {name} ({_kind(name, enabled)}): reference 0: " + REFERENCE_LINES[name].format(**values))
    for extra in extras:
        lines.append(f"reference {extra['index']} mask (extra, on black): area {_value(extra['area'])}, "
                     f"fragments {extra['fragments']}")
        largest = {"largest": max(extra["fragments"], default=0.0)}
        for name in extra["flags"]:
            line = REFERENCE_LINES["extra_" + name if name == "reference_empty" else name]
            lines.append(f"- {name} ({_kind(name, enabled)}): reference {extra['index']}: " + line.format(**largest))
    return "\n".join(lines), passed


def check_scail2(pose_video_mask, reference_image_mask, config=None, enabled=True, stop_on_fail=True, pose_data=None,
                 mask_config=None):
    """The SCAIL-2 checks on the colored driving mask [frames, H, W, 3] (at the generation size)
    and the colored reference mask [N, H', W', 3] (the primary first, then any extra references),
    as SCAIL-2 Preprocess or SCAIL-2 Colored Mask render them, and with `pose_data` (Pose
    Detection on the same driving frames at the same size) the Mask Guard's pose-based checks on
    the driving mask.

    `config` is a SCAIL2GuardConfig and `mask_config` a MaskGuardConfig (None = defaults).
    `enabled` False still measures and reports every check but marks them off, so none can fail.
    With `stop_on_fail` a failed enabled check raises GuardFailed with the report.

    Returns (pose_video_mask unchanged, reference_image_mask unchanged, report, metrics JSON,
    timeline IMAGE). The metrics are {"guard": "scail2", "thresholds", "enabled", "flags" (the
    driving-frame checks, name -> frames), "reference" (the primary reference's record, its checks
    in its "flags"), with extra references "extra_references" (one record each: index, area,
    fragments, flags), "frames"}."""
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
        masks, kept, cells = driving_person(pose_video_mask)
        pose_metas, detections = pose_of(masks, pose_data, "run Pose Detection on the driving frames at the "
                                                           "generation size, the frames SCAIL-2 Preprocess got")
        rows = scail2_frame_metrics(masks, kept, cells, pose_metas, detections, thresholds["draw_threshold"])
        flags = scail2_flags(rows, thresholds, pose_data is not None)
        reference = reference_record(reference_image_mask, W, H, masks[0] if T else None, thresholds)
        extras = extra_reference_records(reference_image_mask)
    report, passed = scail2_report(rows, flags, reference, checks, pose_data is not None, extras)
    record = {"guard": "scail2", "thresholds": thresholds, "enabled": sorted(checks), "flags": flags, "reference": reference}
    if extras:
        record["extra_references"] = extras
    metrics = json.dumps({**record, "frames": rows})
    timeline = timeline_image(rows, flags, SCAIL2_PANELS)
    _stop(report, passed, stop_on_fail)
    return pose_video_mask, reference_image_mask, report, metrics, timeline
