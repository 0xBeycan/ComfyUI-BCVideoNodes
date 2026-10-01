"""The reference check of a replacement run: the character on the reference image, placed as the
core node places the reference, against the person on the mask's first frame.

Core's Wan Animate and SCAIL-2 nodes both place the reference on the generation with
comfy.utils.common_upscale: a center crop to the generation's aspect ratio, then a resize to its
size; a mask is resized nearest-exact, as core resizes the masks it places. `reference_fit` repeats
that on the character's mask and measures it against the first frame: the IoU, the height ratio,
and the share of the character the crop cuts off. Replacement expects the reference posed and placed
like the first frame, so an IoU below `min_reference_iou` is `reference_misaligned`, a warning.

The SCAIL-2 guard reads the character from its colored reference mask (scail2.reference_record).
The Mask Guard and the WanAnimate Preprocess Guard read it from SAM 3.1 Multiplex on the reference
image (`mask_reference`), against the mask they check: the raw mask, or the final mask the sampler
gets, with the placed character grown and blockified the same way first.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ...libs.resize import center_crop
from .common import MaskReference


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _height(mask):
    """The number of rows from the top to the bottom of `mask`'s pixels."""
    rows = np.flatnonzero(mask.any(axis=1))
    return int(rows[-1] - rows[0] + 1)


def reference_fit(person, width, height, first_frame, reading=None):
    """The character `person` ([H', W'] booleans of the reference) placed on a `width` x `height`
    generation as core places the reference - the center crop to its aspect ratio, resized
    nearest-exact - and measured against the person on the first frame (`first_frame`, [height,
    width] booleans, None without frames): {"area": the character's share of the reference,
    "cropped": the share of it the crop cuts off, "iou_first_frame", "scale_first_frame": the IoU
    and the height ratio of the placed character to the first frame's person, None when either is
    empty}. `reading`, when given, turns the placed character into what is compared with the first
    frame (the final mask's grow and blockify)."""
    Hr, Wr = person.shape
    area = int(person.sum())
    x, y = center_crop(Wr, Hr, width, height)
    kept = person[y:Hr - y, x:Wr - x]
    placed = F.interpolate(torch.from_numpy(kept)[None, None].float(), size=(height, width), mode="nearest-exact")[0, 0].numpy() > 0.5
    if reading is not None:
        placed = reading(placed)
    against = area and first_frame is not None and first_frame.any() and placed.any()
    return {"area": area / (Hr * Wr), "cropped": 1.0 - float(kept.sum()) / area if area else 0.0,
            "iou_first_frame": _iou(placed, first_frame) if against else None,
            "scale_first_frame": _height(placed) / _height(first_frame) if against else None}


def mask_reference(reference, shape, first_frame, t, reading=None) -> MaskReference:
    """The Mask Guard's reference record: the character on the reference (`reference`, the MASK
    SAM 3.1 Multiplex found it as, [frames, H', W'] or [H', W'], its first frame read) placed on
    the mask's frames (`shape`, (height, width)) and measured against the mask's first frame
    (`first_frame`, None without frames) by `reference_fit`, `reading` as there. `t` has
    `min_reference_iou`: an IoU below it is `reference_misaligned` in the record's flags."""
    if reference.dim() not in (2, 3) or (reference.dim() == 3 and reference.shape[0] == 0):
        raise ValueError(f"the reference mask must be a MASK [frames, height, width] with a frame, got a tensor of shape "
                         f"{tuple(reference.shape)}")
    person = (reference if reference.dim() == 2 else reference[0]).cpu().numpy() > 0.5
    height, width = shape
    record = reference_fit(person, width, height, first_frame, reading)
    misaligned = record["iou_first_frame"] is not None and record["iou_first_frame"] < t["min_reference_iou"]
    record["flags"] = ["reference_misaligned"] if misaligned else []
    return record
