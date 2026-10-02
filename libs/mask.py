"""Masks at a frame's resolution: mask logits resized to the frame and cut, the largest piece of a
binary mask, a person's binary mask cleaned of small enclosed holes and of islands far smaller
than the body, a mask rendered as a colored image, and the final mask of the Wan Animate workflow
(the raw mask grown and blockified) with the frames painted black under it."""
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F


def to_frame_size(low_res, H, W, threshold=0.0):
    """One frame's [1, 1, h, w] mask logits as an [H, W] float mask of the frame, cut at
    `threshold`."""
    upsampled = F.interpolate(low_res.float(), size=(H, W), mode="bilinear", align_corners=False)
    return (upsampled[0, 0] > threshold).float().cpu()


def masked_frames(masks, chunk=16):
    """[N] bools: which of the [N, H, W] masks have at least one pixel set. Read `chunk` frames at a
    time: any() of a float mask makes a boolean copy of what it reads first, over the whole clip about
    half a GB on a 612-frame 720p clip."""
    present = torch.zeros(len(masks), dtype=torch.bool)
    for s in range(0, len(masks), chunk):
        present[s:s + chunk] = masks[s:s + chunk].flatten(1).any(dim=1)
    return present


def count_masked_frames(masks):
    """How many of the [N, H, W] masks have at least one pixel set (masked_frames)."""
    return int(masked_frames(masks).sum())


def largest_piece(mask):
    """The largest 4-connected piece of the [H, W] bool array `mask` as a bool array (of pieces of
    equal size, the first in scan order), or None when `mask` is empty."""
    from scipy import ndimage
    labels, count = ndimage.label(mask)
    if not count:
        return None
    return labels == int(np.bincount(labels.ravel())[1:].argmax()) + 1


def drop_islands(mask, min_fraction):
    """Remove connected regions of a binary uint8 mask smaller than `min_fraction` of its
    largest one."""
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero(areas >= areas.max() * min_fraction) + 1
    return np.isin(labels, keep).astype(np.uint8)


def fill_holes(mask, max_fraction):
    """Fill the small background regions the mask fully encloses: a person has no holes, so a
    small hole is clothing or hair the decoder dropped. A large one is not - it is the
    background showing between a limb and the body, and it is left alone. Small is up to
    `max_fraction` of the mask's own area."""
    import cv2
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask == 0).astype(np.uint8), connectivity=4)
    if count <= 2:
        return mask
    H, W = mask.shape
    limit = max_fraction * int(mask.sum())
    out = mask.copy()
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if x > 0 and y > 0 and x + w < W and y + h < H and area <= limit:   # encloses, and small
            out[labels == i] = 1
    return out


def clean_mask(mask, config):
    """A person's mask: no small enclosed holes, no islands far smaller than the body."""
    return drop_islands(fill_holes(mask.astype(np.uint8), config.max_hole_fraction), config.min_island_fraction)


def render_identity(mask, color, background, threshold=0.5, chunk=16):
    """A [T, H, W] mask as a [T, H, W, 3] image of the mask's dtype (float32, or float16 from a
    float16 clip: pure colours are exact in it): `color` where the mask is above `threshold`,
    `background` elsewhere. Both colours are RGB in 0..1. Cut and filled `chunk` frames at a time:
    the cut is a boolean copy of what it reads, over the whole clip about half a GB on a 609-frame
    720p clip."""
    color = torch.tensor(color, dtype=mask.dtype, device=mask.device)
    background = torch.tensor(background, dtype=mask.dtype, device=mask.device)
    out = torch.empty(*mask.shape, 3, dtype=mask.dtype, device=mask.device)
    for s in range(0, len(mask), chunk):
        on = (mask[s:s + chunk].float() > threshold).unsqueeze(-1)
        torch.where(on, color, background, out=out[s:s + chunk])
    return out


# --- the final mask ---------------------------------------------------------------------------

# The final mask of the Wan Animate workflow, WanAnimate Preprocess's final_mask and what the guards
# judge: the raw mask grown by `grow` dilations of the 3 x 3 cross, then the box of the grown mask
# cut into blocks of about `block_size` px, every block that holds a grown pixel on and nothing
# outside the box - ComfyUI-BCNodes' MaskGrow (blur 0) then BlockifyMask. GROW and BLOCK_SIZE are
# the defaults of the grow and block_size widgets (FinalMaskConfig), and the final SAM 3.1
# Multiplex's repairs model. A block_size of 0 cuts no blocks (the Mask Guard's, on a raw mask): the
# final is the grown mask itself, read a pixel at a time, as 1 px blocks give it.
GROW, BLOCK_SIZE = 10, 32
CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
_KEEP_EQUAL = ("WanAnimate Preprocess makes final_mask and bg_images with it; the Mask Guard and the WanAnimate "
               "Preprocess Guard judge that final mask with it, so keep theirs equal to the preprocess's.")


@dataclass
class FinalMaskConfig:
    """The final mask's widgets (see GROW), the same on WanAnimate Preprocess and the two guards."""
    grow: int = field(default=GROW, metadata={"min": 0, "max": 999, "step": 1, "tooltip": (
        "The final mask: the raw mask grown by this many steps of the 3 x 3 cross (up to this many px), then cut "
        "into blocks by block_size; ComfyUI-BCNodes' MaskGrow with blur 0. " + _KEEP_EQUAL)})
    block_size: int = field(default=BLOCK_SIZE, metadata={"min": 8, "max": 512, "step": 1, "tooltip": (
        "The final mask: the box of the grown mask cut into blocks of about this many px (the last row and column "
        "take the remainder), every block holding a grown pixel on; ComfyUI-BCNodes' BlockifyMask. " + _KEEP_EQUAL)})


def bounding_box(region):
    """The box of the [H, W] booleans `region` as (row slice, column slice), None when it is empty."""
    rows, cols = np.flatnonzero(region.any(axis=1)), np.flatnonzero(region.any(axis=0))
    if not len(rows):
        return None
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def block_edges(start, end, block_size):
    """BlockifyMask's cuts of one side of the box, start to end: side // block_size blocks (at least
    one) of equal size, the remainder joining the last, as the edges from `start` to `end`. A
    block_size of 0 cuts no blocks: every pixel is one (see GROW)."""
    n = max(1, (end - start) // (block_size or 1))
    size = (end - start) // n
    return np.array([start + i * size for i in range(n)] + [end])


def block_grid(mask, block_size=BLOCK_SIZE):
    """BlockifyMask's grid over the box of `mask` ([H, W] booleans, the grown mask; a final mask has
    the same box): (row edges, column edges), each block from one edge to the next along both sides;
    None on an empty frame. The grid is laid from each frame's own box."""
    box = bounding_box(mask)
    if box is None:
        return None
    rows, cols = box
    return block_edges(rows.start, rows.stop, block_size), block_edges(cols.start, cols.stop, block_size)


def final_blocks(mask, grow=GROW, block_size=BLOCK_SIZE):
    """The final mask of one raw [H, W] boolean frame (see GROW) as its block grid and its blocks:
    `grow` dilations by CROSS (an empty frame stays empty), then block_grid: a block on wherever it
    holds a grown pixel. Returns (grid, [rows, columns] booleans, the blocks that are on), None on
    an empty frame. The grown mask lies within `grow` px of the mask's box, so only that part of
    the frame is grown."""
    import cv2

    box = bounding_box(mask)
    if box is None:
        return None
    H, W = mask.shape
    y0, x0 = max(0, box[0].start - grow), max(0, box[1].start - grow)
    y1, x1 = min(H, box[0].stop + grow), min(W, box[1].stop + grow)
    grown = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), CROSS, iterations=grow).astype(bool)
    rows, cols = block_grid(grown, block_size)
    grown = grown[rows[0]:rows[-1], cols[0]:cols[-1]]
    on = np.logical_or.reduceat(np.logical_or.reduceat(grown, rows[:-1] - rows[0], axis=0), cols[:-1] - cols[0], axis=1)
    return (rows + y0, cols + x0), on


def lay_out(blocks, out):
    """Writes the blocks of final_blocks' (grid, on) into `out`, an [H, W] array of the frame with
    every pixel off: each block's pixels set to whether it is on."""
    (rows, cols), on = blocks
    out[rows[0]:rows[-1], cols[0]:cols[-1]] = np.repeat(np.repeat(on, np.diff(rows), axis=0), np.diff(cols), axis=1)


def final_mask(mask, grow=GROW, block_size=BLOCK_SIZE):
    """The final mask (see GROW) of the raw MASK `mask` [N, H, W], as a 0 / 1 MASK of the raw mask's
    dtype (float32, or float16 from a float16 clip) on the CPU, frame by frame into one output. A
    pixel of the raw mask is set where MaskGrow's 8-bit quantisation keeps a level (255 x value >= 1,
    in float32: a float16 frame is widened first, not requantized, which would round a value under
    1 / 255 up to it; SAM 3.1 Multiplex's mask is 0 or 1)."""
    out = torch.zeros(mask.shape, dtype=mask.dtype)
    for f in range(len(mask)):
        blocks = final_blocks((mask[f].cpu().float() * 255 >= 1).numpy(), grow, block_size)
        if blocks is not None:
            lay_out(blocks, out[f].numpy())
    return out


def painted_black(images, mask):
    """The IMAGE `images` [N, H, W, C] with every pixel `mask` [N, H, W] holds (above 0) black and the
    rest unchanged, frame by frame into one output of the images' dtype on the CPU: ComfyUI-BCNodes'
    Draw Mask On Image with the colour "0, 0, 0" on a 0 / 1 mask. A pixel is selected, never
    computed, so a float16 clip gives the float16 of the float32 result."""
    out = torch.empty(images.shape, dtype=images.dtype)
    black = torch.zeros((), dtype=out.dtype)
    for f in range(len(images)):
        torch.where(mask[f].unsqueeze(-1).cpu() > 0, black, images[f].cpu(), out=out[f])
    return out
