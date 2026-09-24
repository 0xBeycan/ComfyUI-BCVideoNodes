"""Wan Animate's concat-mask repair: the rows the reference implementation builds for one window,
written over the video part of the concat mask WanAnimateToVideo returns."""

from ..common.animate import mask_window


def replacement_mask_rows(character_mask, offset, length, seed_frames, lat_h, lat_w):
    """The concat-mask rows for one window, built the way the reference
    implementation (wan/animate.py get_i2v_mask) builds them: one row per
    pixel frame, frame 0 repeated four times, seed frames known (0), frames
    the mask does not cover unknown (1). The pixel mask -> latent grid step
    uses core's filter and crop (nearest-exact, center). Returns None when the mask does not
    reach this window, which is when core does not apply it either."""
    import torch

    frames = mask_window(character_mask, offset, length, lat_h, lat_w)
    if frames is None:
        return None
    frames[:seed_frames] = 0.0
    return torch.cat((frames[:1].expand(4, -1, -1), frames[1:]), dim=0)


def fix_replacement_mask(cond, rows, seen):
    """Write ``rows`` (from replacement_mask_rows) over the video part of
    the concat mask WanAnimateToVideo returns; index 0 stays the reference
    latent.

    Why: the concat mask has 4 rows per latent frame and pixel frame f >= 1
    belongs at row f + 3 (frame 0 fills latent 0). Core's own seed-frame
    zeroing, its other Wan nodes (``mask[:, :, :frames + 3]``) and the
    reference follow that; WanAnimateToVideo writes character_mask at row f,
    three rows early, so the last three seed rows are overwritten and the
    seed latent is flagged "character unknown" over real pixels.
    """
    for entry in cond:
        mask = entry[1].get("concat_mask") if len(entry) > 1 and isinstance(entry[1], dict) else None
        if mask is None or id(mask) in seen:
            continue
        seen.add(id(mask))
        height, width = mask.shape[-2], mask.shape[-1]
        mask[:, :, 1:] = rows.to(mask).view(1, -1, 4, height, width).transpose(1, 2)
