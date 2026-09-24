"""Masks at a frame's resolution: mask logits resized to the frame and cut, a person's binary
mask cleaned of small enclosed holes and of islands far smaller than the body, and a mask
rendered as a colored image."""
import numpy as np
import torch
import torch.nn.functional as F


def to_frame_size(low_res, H, W, threshold=0.0):
    """One frame's [1, 1, h, w] mask logits as an [H, W] float mask of the frame, cut at
    `threshold`."""
    upsampled = F.interpolate(low_res.float(), size=(H, W), mode="bilinear", align_corners=False)
    return (upsampled[0, 0] > threshold).float().cpu()


def count_masked_frames(masks):
    """How many of the [N, H, W] masks have at least one pixel set."""
    return int((masks.flatten(1).any(dim=1)).sum())


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


def render_identity(mask, color, background, threshold=0.5):
    """A [T, H, W] mask as a [T, H, W, 3] float32 image: `color` where the mask is above
    `threshold`, `background` elsewhere. Both colours are RGB in 0..1."""
    on = (mask.float() > threshold).unsqueeze(-1)
    color = torch.tensor(color, dtype=torch.float32, device=mask.device)
    background = torch.tensor(background, dtype=torch.float32, device=mask.device)
    return torch.where(on, color, background)
