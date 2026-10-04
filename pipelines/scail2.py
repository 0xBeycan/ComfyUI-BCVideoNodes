"""SCAIL-2's colored masks: a person MASK rendered in an identity colour on the background each
mode was trained with, in the form core's WanSCAILToVideo reads (float 0..1, pure colours, which
its 28-channel extraction thresholds at 225/255); the driving video on black, the pose video of
animation mode with the background blacked out; and the face close-up, an extra reference of
SCAIL-2's multi-reference (zai-org/SCAIL-2 README, Experimental Functions: Multi-Reference): the
face cut from the source image in the generation's aspect, its mask the character in the identity
colour on black in both modes.

Single identity only: the person is palette colour 0 (blue). Multi-person is phase 2; it renders
each identity with the same render_identity (libs/mask.py) in its own palette colour.
"""
import torch

from ..libs import log, resize
from ..libs.mask import render_identity
from ..libs.video import requantized
from .face import face_bboxes_from_pose

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


def colored_masks(driving_mask, replacement_mode, reference_mask=None, render_driving=True):
    """(pose_video_mask [T, H, W, 3], reference_image_mask [N, H, W, 3]) for WanSCAILToVideo from
    the person's driving MASK [T, H, W] and reference MASK [N, H, W], both on above 0.5.
    `render_driving` False: pose_video_mask is not rendered, [0, H, W, 3]; the reference mask does
    not read it.

    Without a reference mask, or with one that marks no pixel, the reference mask is the
    background alone, as core renders it: in animation mode that is logged (the mode can collapse
    into replacement behaviour, SCAIL-2 README), in replacement mode it raises, since core would
    black out the whole reference."""
    driving_background, reference_background = backgrounds(replacement_mode)
    driving = _frames(driving_mask)
    pose_video_mask = render_identity(driving if render_driving else driving[:0], PALETTE[0], driving_background, MASK_THRESHOLD)
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


def check_face_crop(face_crop, reference_source, reference_image, reference_mask=None):
    """Raises when face_crop is on without a single reference_source image, or with a reference_mask
    of another size than reference_image (the references and their masks go to the sampler as one
    batch each, and one batch holds one size); logs a connected reference_source face_crop off
    leaves unused. Called before any model runs."""
    if not face_crop:
        if reference_source is not None:
            log.info("face_crop is off; reference_source not used")
        return
    if reference_source is None:
        raise ValueError("face_crop is on but reference_source is not connected: connect Load Reference Image's source_image "
                         "(the reference at its own resolution) to SCAIL-2 Preprocess's reference_source, or turn face_crop off.")
    if reference_source.shape[0] != 1:
        raise ValueError(f"reference_source holds {reference_source.shape[0]} images; face_crop cuts the face from one image: "
                         "connect Load Reference Image's source_image.")
    height, width = reference_image.shape[1:3]
    if reference_mask is not None and tuple(reference_mask.shape[-2:]) != (height, width):
        raise ValueError(f"reference_mask is {reference_mask.shape[-1]}x{reference_mask.shape[-2]} but reference_image "
                         f"{width}x{height}; with face_crop the references and their masks go to the sampler as one batch "
                         "each, so connect a reference_mask of reference_image's size (Load Reference Image's resized_mask), "
                         "or leave it unconnected.")


def driving_on_black(images, driving_mask, chunk=16):
    """The driving video `images` [T, H, W, C] with every pixel outside the person's MASK
    [T, H, W] (on above 0.5, as the colored masks cut it; a single mask is every frame's) black,
    the rest unchanged: the pose video SCAIL-2 was trained on in animation mode (black
    backgrounds, zai-org/SCAIL-2 issue #17), as SCAIL-Pose's --crop_e2e_mask writes it (the union
    of the person silhouettes kept). Cut and filled `chunk` frames at a time, so the cut, a boolean
    copy of the mask, never covers the whole clip."""
    mask = _frames(driving_mask).expand(len(images), -1, -1)
    out = torch.empty_like(images)
    black = torch.zeros((), dtype=images.dtype, device=images.device)
    for s in range(0, len(images), chunk):
        keep = (mask[s:s + chunk] > MASK_THRESHOLD).unsqueeze(-1).to(images.device)
        torch.where(keep, images[s:s + chunk], black, out=out[s:s + chunk])
    return out


# The face close-up's framing: the face box is centred in the crop horizontally and fills its
# width (its height when the generation is so wide that the box would not fit); of the crop's free
# height, half goes above the box, but never more than FACE_HEADROOM box heights: the hair above
# the face keypoints' box stays in, and the rest goes below it (neck, shoulders).
FACE_HEADROOM = 0.4


def face_box(pose_data, image_width, image_height):
    """(x1, y1, x2, y2): the face box of the one image Pose Detection read into `pose_data`, as
    the Face Crop node boxes it (face.face_bboxes_from_pose: the face keypoints grown to
    FACE_CROP_SCALE, Wan Animate's face and hair framing; one image, so no smoothing). A face seen
    from behind still has its box. Raises when Pose Detection found no person on the image."""
    if pose_data["detections"][0]["score"] < 0:  # the detector found no usable person box (pose.detect)
        raise ValueError("face_crop: Pose Detection found no person on reference_source, so there is no face to cut. "
                         "Connect an image of the character with the head in view to reference_source (Load Reference "
                         "Image's source_image), or turn face_crop off.")
    return face_bboxes_from_pose(pose_data, image_width, image_height, smoothing="off")[0]


def face_crop_box(box, image_width, image_height, width, height):
    """(x, y, w, h): the region of an `image_width` x `image_height` image the face close-up is cut
    from, in the `width` x `height` generation's aspect, around the face `box` (x1, y1, x2, y2)
    framed as FACE_HEADROOM says, shifted inside the image. When that region is larger than the
    image, the largest region of the aspect the image holds, placed the same way."""
    x1, y1, x2, y2 = box
    box_width, box_height = x2 - x1, y2 - y1
    aspect = width / height
    w = max(box_width, box_height * aspect)
    h = w / aspect
    if w > image_width or h > image_height:
        w, h = (image_height * aspect, image_height) if image_width / image_height > aspect else (image_width, image_width / aspect)
    w, h = min(round(w), image_width), min(round(h), image_height)
    x = (x1 + x2) / 2 - w / 2
    y = y1 - min(max(h - box_height, 0) / 2, FACE_HEADROOM * box_height)
    return min(max(round(x), 0), image_width - w), min(max(round(y), 0), image_height - h), w, h


def face_reference(source, box, width, height):
    """The face close-up: the face_crop_box region of `source` [1, H, W, 3] (the reference image at
    its own resolution) resized to `width` x `height` by lanczos (libs/resize, Load Reference Image's
    resize), IMAGE [1, height, width, 3] float32."""
    frame = requantized(source[0, ..., :3])
    x, y, w, h = face_crop_box(box, frame.shape[1], frame.shape[0], width, height)
    face = torch.empty((1, height, width, 3), dtype=torch.float32)
    resize.fit(frame[y:y + h, x:x + w], face[0])
    log.info(f"face_crop: the face box {tuple(box)} framed as {w}x{h} at ({x}, {y}) of the {frame.shape[1]}x"
             f"{frame.shape[0]} reference_source, resized to {width}x{height}")
    return face


def extra_reference_mask(mask):
    """The colored mask of an extra reference (SCAIL-2 multi-reference) from the character's MASK
    [N, H, W] on it (on above 0.5): the character in the identity colour on black, in animation and
    replacement mode alike (the official multi-reference example), [N, H, W, 3]. A mask that marks
    no pixel is logged: the extra reference then binds to no identity (the SCAIL-2 Preprocess Guard
    fails it)."""
    frames = _frames(mask)
    if not bool((frames > MASK_THRESHOLD).any()):
        log.warning("the extra reference's mask marks no pixel: the prompt found no character on the face close-up, so "
                    "it binds to no identity. Check the prompt, or turn face_crop off.")
    return render_identity(frames, PALETTE[0], BLACK, MASK_THRESHOLD)


def with_extra_reference(reference_images, reference_image_mask, image, image_mask):
    """(reference_images, reference_image_mask) with `image` [1, H, W, 3] and its colored mask
    `image_mask` [1, H, W, 3] appended as the next reference: core's WanSCAILToVideo takes the
    references as one batch and their masks as another, paired by position."""
    return (torch.cat([reference_images, image.to(reference_images.dtype)]),
            torch.cat([reference_image_mask, image_mask.to(reference_image_mask.dtype)]))
