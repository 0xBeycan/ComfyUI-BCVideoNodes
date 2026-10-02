"""The flows of the video input nodes: Load Video (a file decoded frame by frame, cropped and
resized to the model's generation size, or kept at the source size, straight into one preallocated
IMAGE batch in the dtype of its precision, with its audio and its video_info), Load Reference Image (an image fitted to the video's loaded size) and Conform
Video (a clip fitted to the nearest platform size).

The full-resolution clip never exists: one source frame is decoded at a time, and the batch is
allocated once, at its final frame count."""
from typing import Optional, TypedDict

import torch

from ..libs import log, resize, video_decode
from ..libs import video_sizes as sizes
from ..libs.video import FP32, precision_dtype, requantized
from ..libs.video_info import VideoInfo

# a force_fps this close to the video's rate (as a share of it) is that rate
FORCE_FPS_TOLERANCE = 1e-4


def parse_force_fps(value):
    """force_fps's text as a rate: None when empty (the source's), else a number above 0."""
    text = str(value).strip()
    if not text:
        return None
    try:
        rate = float(text)
    except ValueError:
        rate = None
    if rate is None or not rate > 0 or rate == float("inf"):
        raise ValueError(f"force_fps must be empty (the source frame rate) or a number above 0, such as 24 "
                         f"or 29.97; got {value!r}.")
    return rate


def parse_frame_count(value):
    """frame_count's text as a count: None when empty (every frame from start_frame on), else a
    whole number of at least 1."""
    text = str(value).strip()
    if not text:
        return None
    try:
        count = int(text)
    except ValueError:
        count = 0
    if count < 1:
        raise ValueError(f"frame_count must be empty (every frame from start_frame on) or a whole number of at "
                         f"least 1; got {value!r}.")
    return count


def loaded_frames(kept, start_frame, frame_count, step, force_fps=None):
    """The source indices Load Video loads: `kept` (the frames after force_fps) from the 1-based
    `start_frame`, `frame_count` of them (None: all the rest), cut to the model's step * n + 1
    frames. Raises ValueError, saying what to change, for a range past the frames."""
    at = f" at force_fps {force_fps:g}" if force_fps is not None else ""
    if start_frame < 1:
        raise ValueError(f"start_frame must be 1 or more (the first frame is 1); got {start_frame}.")
    if start_frame > len(kept):
        raise ValueError(f"start_frame {start_frame} is past the end of the video: it has {len(kept)} frames{at}. "
                         f"Set start_frame to {len(kept)} or less.")
    if frame_count is not None and start_frame + frame_count - 1 > len(kept):
        raise ValueError(f"start_frame {start_frame} with frame_count {frame_count} ends at frame "
                         f"{start_frame + frame_count - 1}, past the video's {len(kept)} frames{at}. Set frame_count "
                         f"to {len(kept) - start_frame + 1} or less, or start earlier.")
    chosen = kept[start_frame - 1:] if frame_count is None else kept[start_frame - 1:start_frame - 1 + frame_count]
    return chosen[:(len(chosen) - 1) // step * step + 1]


def kept_rate(rate, fps):
    """The rate Load Video keeps frames at for a force_fps of `rate` (parse_force_fps's) on a video
    of `fps`: None (every frame) when `rate` is None or within FORCE_FPS_TOLERANCE of `fps`, else
    `rate`. Raises ValueError, saying what to set, for a rate above `fps`."""
    if rate is not None and abs(rate - fps) <= fps * FORCE_FPS_TOLERANCE:
        # the video's own rate, as typed (29.97 for 30000/1001): every frame, not a grid a hair
        # slower that drops frame 1
        return None
    if rate is not None and rate > fps:
        # the loader keeps or drops real frames, it never repeats one to raise the rate: a repeat
        # fakes the frame rate, and a typo such as 300 would multiply the batch
        raise ValueError(f"force_fps {rate:g} is above the video's frame rate {fps:.6g}: the loader only "
                         f"lowers the frame rate (it keeps or drops real frames, never repeats them). Leave force_fps "
                         f"empty or set it to {fps:.6g} or less.")
    return rate


def plan(path, model, resolution, orientation, force_fps, frame_count, precision=FP32, probe=None):
    """Load Video's widgets checked against the video at `path`, before a frame is loaded:
    {"source": the probe (video_decode.probe's dict), "force_fps": the rate typed or None, "rate":
    the rate frames are kept at (kept_rate's), "count": frame_count or None, "orientation": portrait
    or landscape, "width", "height": the loaded size, "fit": how a frame gets there (resize.CROP, or
    resize.CUT for source), "dtype": the batch's (libs/video.precision_dtype)}. Raises ValueError,
    saying what to change, in the order load_video meets them: the widgets, then the file. `probe`
    reads the file (video_decode.probe when None)."""
    size = sizes.model_size(model, resolution)
    typed = parse_force_fps(force_fps)
    count = parse_frame_count(frame_count)
    if orientation not in sizes.ORIENTATIONS:
        raise ValueError(f"orientation {orientation!r} is not one of {', '.join(sizes.ORIENTATIONS)}.")
    dtype = precision_dtype(precision)
    source = (probe or video_decode.probe)(path)
    rate = kept_rate(typed, source["fps"])
    own = sizes.orientation_of(source["width"], source["height"])
    turned = own if orientation == sizes.AUTO else orientation
    if size is not None:
        width, height = sizes.oriented(size, turned)
    else:
        # source: the video's own size; in the other orientation the centred crop to the turned
        # aspect, the short side kept (1920x1080 -> 608x1080), never a rotation; then each side
        # cut down to the model's grid (Wan 16, SCAIL 32, None 1). Frames are cut, not resized.
        width, height = source["width"], source["height"]
        if turned != own:
            width, height = resize.crop_box(width, height, height, width)[2:]
        grid = sizes.MODELS[model]["grid"]
        if width < grid or height < grid:
            raise ValueError(f"resolution source gives {width}x{height}, smaller than model {model}'s {grid}-pixel "
                             f"grid; pick model None or one of {model}'s sized resolutions.")
        width, height = width // grid * grid, height // grid * grid
    return {"source": source, "force_fps": typed, "rate": rate, "count": count, "orientation": turned,
            "width": width, "height": height, "fit": resize.CROP if size is not None else resize.CUT, "dtype": dtype}


def frame_indices(source, rate, start_frame, count, model):
    """The source indices Load Video loads from a video probed as `source`: the frames kept at
    `rate`, from `start_frame`, `count` of them (None: all the rest), cut to the model's frame rule
    (4n+1 for Wan and SCAIL, none for None; loaded_frames, which raises for a range past the
    frames)."""
    kept = video_decode.select_frames(source["fps"], source["frames"], rate)
    return loaded_frames(kept, start_frame, count, sizes.MODELS[model]["frames"], rate)


def video_info(model, resolution, planned, frames):
    """The VideoInfo of `frames` frames loaded as `planned` (plan's) says."""
    source = planned["source"]
    loaded_fps = planned["rate"] if planned["rate"] is not None else source["fps"]
    frame_time = 1 / loaded_fps
    return VideoInfo(model=model, resolution=resolution, orientation=planned["orientation"],
                     source_fps=source["fps"], source_frame_count=source["frames"],
                     source_duration=source["frames"] / source["fps"],
                     source_width=source["width"], source_height=source["height"],
                     loaded_fps=loaded_fps, loaded_frame_count=frames, loaded_duration=frames * frame_time,
                     loaded_width=planned["width"], loaded_height=planned["height"])


def load_video(path, model, resolution, orientation, force_fps, start_frame, frame_count, precision=FP32):
    """(IMAGE [N, H, W, 3], AUDIO or None, VideoInfo) of the video file at `path`, as Load Video's
    widgets say; the IMAGE float32, or float16 at precision fp16 (every 8-bit level k / 255 is
    kept: libs/video.requantized gives back the float32 values exactly). N is fixed before the
    batch is allocated: the container's packets give the frame count; should the decoder disagree,
    the load runs once more with the decoder's count."""
    planned = plan(path, model, resolution, orientation, force_fps, frame_count, precision)
    source, rate, width, height = planned["source"], planned["rate"], planned["width"], planned["height"]
    if planned["force_fps"] is not None and rate is None:
        log.info(f"force_fps {planned['force_fps']:g} is the video's frame rate ({source['fps']:.6g}): every frame kept")
    result = {}
    stored = "" if precision == FP32 else f" as {precision}"
    with log.step(f"loading {model} {resolution} {planned['orientation']} ({width}x{height}){stored} from a "
                  f"{source['width']}x{source['height']} video", result):
        try:
            loaded = _decode(path, source, width, height, planned["fit"], rate, start_frame, planned["count"], model,
                             planned["dtype"])
        except video_decode.FrameCountChanged as changed:
            log.warning(f"the container promised {source['frames']} frames, the decoder gave {changed.frames}; "
                        f"loading again with {changed.frames}")
            source["frames"] = changed.frames
            loaded = None
        if loaded is None:  # outside the except block: its traceback would keep the first batch alive
            loaded = _decode(path, source, width, height, planned["fit"], rate, start_frame, planned["count"], model,
                             planned["dtype"])
        images, indices = loaded
        info = video_info(model, resolution, planned, len(indices))
        frame_time = 1 / info["loaded_fps"]
        audio = video_decode.read_audio(path, (start_frame - 1) * frame_time, len(indices) * frame_time,
                                        origin=source["start"])
        result["frames"] = len(indices)
        result["audio"] = "none" if audio is None else f"{audio['waveform'].shape[-1]} samples"
    return images, audio, info


def _decode(path, source, width, height, how, rate, start_frame, count, model, dtype):
    """(the batch, its source indices): the kept frames decoded one at a time and fitted (resize.fit
    `how`) into a batch of `dtype` allocated at its final size."""
    from comfy.utils import ProgressBar

    indices = frame_indices(source, rate, start_frame, count, model)
    images = torch.empty((len(indices), height, width, 3), dtype=dtype)
    bar = ProgressBar(len(indices))
    for pixels, positions in video_decode.kept_frames(path, indices, source["frames"], to_end=count is None):
        resize.fit(pixels, images[positions[0]], how)
        for position in positions[1:]:
            images[position].copy_(images[positions[0]])
        bar.update_absolute(positions[-1] + 1, len(indices))
    return images, indices


class LoadPreview(TypedDict):
    """What Load Video's preview shows, from the file's header and packets (no frame is loaded):
    the probe of the file (None when it cannot be read), the video_info the loader outputs (None on
    an error), the frames from start_frame on at the kept rate (what an empty frame_count stands
    for, before the cut to the model's frame rule; None when force_fps or start_frame is wrong) and the error the
    loader raises, word for word (None when it loads)."""
    source: Optional[dict]
    info: Optional[VideoInfo]
    available: Optional[int]
    error: Optional[str]


def preview(path, model, resolution, orientation, force_fps, start_frame, frame_count, precision=FP32, probe=None):
    """The LoadPreview of Load Video's widget values on the video at `path`, from the loader's own
    checks and frame selection. Its counts are the container's: should the decoder disagree when
    the video loads, the loader goes by the decoder's (load_video). `probe` as plan's."""
    from functools import lru_cache

    from av.error import FFmpegError

    probe = lru_cache(maxsize=1)(probe or video_decode.probe)
    answer = LoadPreview(source=None, info=None, available=None, error=None)
    try:
        planned = plan(path, model, resolution, orientation, force_fps, frame_count, precision, probe)
        indices = frame_indices(planned["source"], planned["rate"], start_frame, planned["count"], model)
        answer["info"] = video_info(model, resolution, planned, len(indices))
    except (ValueError, FFmpegError) as error:
        answer["error"] = str(error)
    # what the file and force_fps alone give, whatever else is wrong: an error met here is one
    # plan or frame_indices raises too, so "error" above already names it or one met before it
    try:
        source = probe(path)
        answer["source"] = dict(source)
        kept = video_decode.select_frames(source["fps"], source["frames"],
                                          kept_rate(parse_force_fps(force_fps), source["fps"]))
        if 1 <= start_frame <= len(kept):
            answer["available"] = len(kept) - start_frame + 1
    except (ValueError, FFmpegError):
        pass
    return answer


def load_reference_image(path, width, height):
    """(IMAGE [1, height, width, 3], MASK) of the image file at `path`, loaded as core's Load
    Image does (EXIF orientation applied, alpha -> MASK as 1 - alpha; its first frame) and fitted
    to `width` x `height` by a centre crop and lanczos. The MASK is fitted the same way, or core's
    64x64 zeros when the image has no alpha."""
    import numpy as np
    import node_helpers
    from PIL import Image, ImageOps

    source = node_helpers.pillow(Image.open, path)
    source = node_helpers.pillow(ImageOps.exif_transpose, source)
    image = torch.empty((1, height, width, 3), dtype=torch.float32)
    resize.fit(np.array(source.convert("RGB")), image[0])
    if "A" not in source.getbands():
        return image, torch.zeros((1, 64, 64), dtype=torch.float32)
    mask = torch.empty((1, height, width), dtype=torch.float32)
    resize.fit(np.array(source.getchannel("A")), mask[0])
    return image, mask.neg_().add_(1)


def conform_video(images, how, method):
    """`images` fitted to the CONFORM_SIZES entry nearest its short edge, oriented as its frames
    (video_sizes.conform_size), frame by frame into one preallocated batch of its dtype (a
    half-precision frame is resized requantized to float32: libs/video.requantized); the input
    itself when it already has that size."""
    frames, height, width, channels = images.shape
    target_width, target_height = sizes.conform_size(width, height)
    if (target_width, target_height) == (width, height):
        log.info(f"conform: {width}x{height} is already a standard size, nothing to do")
        return images
    out = torch.empty((frames, target_height, target_width, channels), dtype=images.dtype, device=images.device)
    with log.step(f"conforming {frames} frames of {width}x{height} to {target_width}x{target_height} "
                  f"({how}, {method})"):
        for index in range(frames):
            resize.fit(requantized(images[index]), out[index], how, method)
    return out
