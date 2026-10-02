"""The one resize of the video nodes: a frame fitted to a target size, by cropping (cut to the
target's aspect, centred, then resized: a cover) or by padding (resized to fit inside, centred
between black bars: a contain), with one of the methods of comfy.utils.common_upscale; or cut
(the centred region of the target size, no resampling: Load Video's resolution source).

A frame is [H, W] or [H, W, C], either uint8 pixels (numpy; what a decoder gives) or a float
tensor in [0, 1] (an IMAGE frame). It is written into `out`, a float tensor of the target size,
so a batch is filled frame by frame into one preallocated tensor.

center_crop is core's own rule for the same cut (comfy.utils.common_upscale, "center"), for code
that has to land where core lands: the guards placing a reference, the pose drawn at a ControlNet's
size."""
import numpy as np
import torch

CROP, PAD = "crop", "pad"
FITS = [CROP, PAD]
# not one of Conform Video's fits: the centred region of the target size, no resampling
CUT = "cut"
LANCZOS = "lanczos"
# comfy.utils.common_upscale's methods; lanczos is PIL's, the others torch's interpolate and bislerp
METHODS = [LANCZOS, "bicubic", "bilinear", "area", "nearest-exact", "bislerp"]


def crop_box(width, height, target_width, target_height):
    """(x, y, w, h): the centred region of a `width` x `height` frame with the target's aspect.
    The cut side is rounded, not truncated (1 px apart from `int` on some sizes)."""
    aspect, target_aspect = width / height, target_width / target_height
    if aspect > target_aspect:
        w, h = round(height * target_aspect), height
    else:
        w, h = width, round(width / target_aspect)
    return (width - w) // 2, (height - h) // 2, w, h


def center_crop(width, height, new_width, new_height):
    """(x, y): the columns and the rows comfy.utils.common_upscale's center crop cuts off each
    side of a `width` x `height` image to reach the aspect ratio of `new_width` x `new_height`: how
    core places a reference and fits a ControlNet hint. It rounds the margin where crop_box rounds
    the side, so the two can keep a side 1 px apart."""
    old_aspect, new_aspect = width / height, new_width / new_height
    x = y = 0
    if old_aspect > new_aspect:
        x = round((width - width * (new_aspect / old_aspect)) / 2)
    elif old_aspect < new_aspect:
        y = round((height - height * (old_aspect / new_aspect)) / 2)
    return x, y


def contain_box(width, height, target_width, target_height):
    """(x, y, w, h): where a `width` x `height` frame, resized to fit inside the target with its
    own aspect, sits centred in it."""
    ratio = min(target_width / width, target_height / height)
    w, h = round(width * ratio), round(height * ratio)
    return (target_width - w) // 2, (target_height - h) // 2, w, h


def fit(frame, out, how=CROP, method=LANCZOS):
    """Writes `frame` fitted to `out`'s size into `out`: `how` is CROP, PAD (black bars) or CUT
    (`out` no larger than the frame; `method` unused), `method` one of METHODS."""
    height, width = frame.shape[:2]
    target_height, target_width = out.shape[:2]
    if how == CROP:
        x, y, w, h = crop_box(width, height, target_width, target_height)
        _resize(frame[y:y + h, x:x + w], out, method)
    elif how == PAD:
        x, y, w, h = contain_box(width, height, target_width, target_height)
        out[:y].zero_()
        out[y + h:].zero_()
        out[y:y + h, :x].zero_()
        out[y:y + h, x + w:].zero_()
        _resize(frame, out[y:y + h, x:x + w], method)
    elif how == CUT:
        x, y = (width - target_width) // 2, (height - target_height) // 2
        _store(frame[y:y + target_height, x:x + target_width], out)
    else:
        raise ValueError(f"fit {how!r} is not one of {', '.join([*FITS, CUT])}.")


def _resize(frame, out, method):
    """`frame` resized to `out`'s size (no resampling at that size) and written into it."""
    height, width = out.shape[:2]
    if frame.shape[:2] == (height, width):
        _store(frame, out)
    elif method == LANCZOS:
        # comfy.utils.lanczos: PIL LANCZOS on 8-bit pixels, then / 255
        from PIL import Image

        image = Image.fromarray(np.ascontiguousarray(_pixels(frame)))
        _store(np.array(image.resize((width, height), resample=Image.Resampling.LANCZOS)), out)
    elif method in METHODS:
        from comfy.utils import common_upscale

        planes = _unit(frame).to(out.device)
        planes = planes[None] if planes.ndim == 2 else planes.movedim(-1, 0)
        resized = common_upscale(planes[None], width, height, method, "disabled")[0]
        out.copy_(resized[0] if out.ndim == 2 else resized.movedim(0, -1))
    else:
        raise ValueError(f"method {method!r} is not one of {', '.join(METHODS)}.")


def _store(frame, out):
    """`frame` written into `out`: uint8 pixels as / 255 (the float32 values comfy.utils.lanczos
    gives; computed in float32 and then stored for an `out` of another dtype, float16 arithmetic on
    the CPU being slow), a float frame as it is."""
    if isinstance(frame, torch.Tensor):
        out.copy_(frame)
    elif out.dtype == torch.float32:
        out.copy_(torch.from_numpy(np.ascontiguousarray(frame))).div_(255)
    else:
        out.copy_(torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255))


def _pixels(frame):
    """`frame` as uint8 pixels: a float frame rounded to the nearest level (comfy.utils.lanczos
    truncates, which darkens every value that is not already a whole level)."""
    if not isinstance(frame, torch.Tensor):
        return frame
    return torch.round(frame * 255).clamp_(0, 255).to(torch.uint8).cpu().numpy()


def _unit(frame):
    """`frame` as a float tensor in [0, 1]."""
    if isinstance(frame, torch.Tensor):
        return frame
    return torch.from_numpy(np.ascontiguousarray(frame)).float().div_(255)
