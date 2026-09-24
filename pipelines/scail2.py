"""SCAIL-2's colored masks: a person MASK rendered in an identity colour on the background each
mode was trained with, in the form core's WanSCAILToVideo reads (float 0..1, pure colours, which
its 28-channel extraction thresholds at 225/255); and the driving video on black, the pose video of
animation mode with the background blacked out.

Single identity only: the person is palette colour 0 (blue). Multi-person is phase 2; it renders
each identity with the same render_identity (libs/mask.py) in its own palette colour.
"""
import torch

from ..libs import log
from ..libs.mask import render_identity

# core's palette (comfy_extras/nodes_scail.py DEFAULT_PALETTE): "Model was trained on these exact
# colors". Blue, red, green, magenta, cyan, yellow.
PALETTE = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (1.0, 1.0, 0.0))
WHITE = (1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0)
MASK_THRESHOLD = 0.5


def backgrounds(replacement_mode):
    """(driving mask background, reference mask background): black / white in animation mode,
    white / black in replacement mode (SCAIL-Pose's preprocess and core's SCAIL2ColoredMask)."""
    return (WHITE, BLACK) if replacement_mode else (BLACK, WHITE)


def _frames(mask):
    """A MASK as [T, H, W]."""
    return mask.unsqueeze(0) if mask.ndim == 2 else mask


def colored_masks(driving_mask, replacement_mode, reference_mask=None):
    """(pose_video_mask [T, H, W, 3], reference_image_mask [N, H, W, 3]) for WanSCAILToVideo from
    the person's driving MASK [T, H, W] and reference MASK [N, H, W], both on above 0.5.

    Without a reference mask, or with one that marks no pixel, the reference mask is the
    background alone, as core renders it: in animation mode that is logged (the mode can collapse
    into replacement behaviour, SCAIL-2 README), in replacement mode it raises, since core would
    black out the whole reference."""
    driving_background, reference_background = backgrounds(replacement_mode)
    driving = _frames(driving_mask)
    pose_video_mask = render_identity(driving, PALETTE[0], driving_background, MASK_THRESHOLD)
    reference = None if reference_mask is None else _frames(reference_mask)
    if reference is None or not bool((reference > MASK_THRESHOLD).any()):
        what = "no reference_mask is connected" if reference is None else "reference_mask marks no pixel"
        if replacement_mode:
            raise ValueError(f"{what}: in replacement mode the reference is cut out by its mask, so it would be black. "
                             "Connect a MASK of the character on the reference image (in SCAIL-2 Preprocess: a prompt "
                             "that finds the character on the reference, or reference_mask).")
        log.warning(f"{what}: the reference mask is plain white, with no identity colour. Animation mode without a "
                    "reference identity mask can collapse into replacement behaviour (SCAIL-2 README); connect a MASK "
                    "of the character on the reference image.")
        if reference is None:
            reference = torch.zeros(1, *driving.shape[1:], device=driving.device)
    reference_image_mask = render_identity(reference, PALETTE[0], reference_background, MASK_THRESHOLD)
    return pose_video_mask, reference_image_mask


def check_black_background(black_background, replacement_mode):
    """Raises when the driving video's background is to be blacked out in replacement mode, where
    the result keeps that background. Called before the tracking, so the error costs no SAM run."""
    if black_background and replacement_mode:
        raise ValueError("black_background is for animation mode: in replacement mode the result keeps the driving "
                         "video's background, so it must reach the sampler. Turn black_background off, or turn "
                         "replacement_mode off to animate the reference character instead.")


def driving_on_black(images, driving_mask):
    """The driving video `images` [T, H, W, C] with every pixel outside the person's MASK
    [T, H, W] (on above 0.5, as the colored masks cut it) black, the rest unchanged: the pose
    video SCAIL-2 was trained on in animation mode (black backgrounds, zai-org/SCAIL-2 issue #17),
    as SCAIL-Pose's --crop_e2e_mask writes it (the union of the person silhouettes kept)."""
    keep = (_frames(driving_mask) > MASK_THRESHOLD).unsqueeze(-1).to(images.device)
    return torch.where(keep, images, torch.zeros((), dtype=images.dtype, device=images.device))
