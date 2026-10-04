"""The flows of the video input nodes: Load Video (a file decoded frame by frame, cropped and
resized to the model's generation size, or kept at the source size, straight into one preallocated
IMAGE batch in the dtype of its precision, with its audio and its video_info), Load Reference Image (an image fitted to the video's loaded size, and as loaded) and Conform
Video (a clip fitted to the nearest platform size).

The full-resolution clip never exists: one source frame is decoded at a time, and the batch is
allocated once, at its final frame count."""
from typing import Optional, TypedDict

import torch

from ..libs import log, resize, video_decode
from ..libs import video_sizes as sizes
from ..libs.video import FP16, FP32, precision_dtype, requantized
from ..libs.video_info import VideoInfo


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


def plan(path, model, resolution, orientation, force_fps, frame_count, precision=FP16, probe=None):
    """Load Video's widgets checked against the video at `path`, before a frame is loaded:
    {"source": the probe (video_decode.probe's dict), "rate": force_fps's rate, the loaded rate
    exactly as typed (None when empty: the source's), "count": frame_count or None, "orientation":
    portrait or landscape, "width", "height": the loaded size, "fit": how a frame gets there
    (resize.CROP, or resize.CUT for source), "dtype": the batch's (libs/video.precision_dtype)}. Raises ValueError,
    saying what to change, in the order load_video meets them: the widgets, then the file. `probe`
    reads the file (video_decode.probe when None)."""
    size = sizes.model_size(model, resolution)
    rate = parse_force_fps(force_fps)
    count = parse_frame_count(frame_count)
    if orientation not in sizes.ORIENTATIONS:
        raise ValueError(f"orientation {orientation!r} is not one of {', '.join(sizes.ORIENTATIONS)}.")
    dtype = precision_dtype(precision)
    source = (probe or video_decode.probe)(path)
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
    return {"source": source, "rate": rate, "count": count, "orientation": turned,
            "width": width, "height": height, "fit": resize.CROP if size is not None else resize.CUT, "dtype": dtype}


def frame_indices(source, rate, start_frame, count, model):
    """(the source indices Load Video loads from a video probed as `source`, how many frames the
    range held before the cut): the frames of `rate`'s time grid (video_decode.select_frames: real
    frames dropped below the video's rate, repeated above it), from `start_frame`, `count` of them
    (None: all the rest), cut to the model's frame rule (4n+1 for Wan and SCAIL, none for None;
    loaded_frames, which raises for a range past the frames). The cut drops the range's last
    frames."""
    kept = video_decode.select_frames(source["fps"], source["frames"], rate)
    indices = loaded_frames(kept, start_frame, count, sizes.MODELS[model]["frames"], rate)
    return indices, len(kept) - start_frame + 1 if count is None else count


def loaded_rate(rate, source):
    """The loaded frame rate: force_fps's `rate`, or the rate of the video probed as `source` when
    it is None."""
    return rate if rate is not None else source["fps"]


def video_info(model, resolution, planned, frames):
    """The VideoInfo of `frames` frames loaded as `planned` (plan's) says, but its audio, which
    load_video puts first once read (the preview reads no samples)."""
    source = planned["source"]
    loaded_fps = loaded_rate(planned["rate"], source)
    frame_time = 1 / loaded_fps
    return VideoInfo(model=model, resolution=resolution, orientation=planned["orientation"],
                     source_fps=source["fps"], source_frame_count=source["frames"],
                     source_duration=source["frames"] / source["fps"],
                     source_width=source["width"], source_height=source["height"],
                     loaded_fps=loaded_fps, loaded_frame_count=frames, loaded_duration=frames * frame_time,
                     loaded_width=planned["width"], loaded_height=planned["height"])


def load_video(path, model, resolution, orientation, force_fps, start_frame, frame_count, precision=FP16):
    """(IMAGE [N, H, W, 3], AUDIO or None, VideoInfo) of the video file at `path`, as Load Video's
    widgets say; the IMAGE float16 at precision fp16, the default (every 8-bit level k / 255 is
    kept: libs/video.requantized gives back the float32 values exactly), float32 at fp32. N is fixed before the
    batch is allocated: the container's packets give the frame count; should the decoder disagree,
    the load runs once more with the decoder's count."""
    planned = plan(path, model, resolution, orientation, force_fps, frame_count, precision)
    source, rate, width, height = planned["source"], planned["rate"], planned["width"], planned["height"]
    result = {}
    stored = "" if precision == FP32 else f" as {precision}"
    named = "" if model == sizes.NO_MODEL else f"{model} "
    with log.step(f"loading {named}{resolution} {planned['orientation']} ({width}x{height}){stored} from a "
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
        images, indices, ranged = loaded
        info = video_info(model, resolution, planned, len(indices))
        frame_time = 1 / info["loaded_fps"]
        # the loaded frames' span on the loaded rate's grid: it ends where the frames end, after the
        # cut to the frame rule too, and frames dropped or repeated by force_fps keep it in sync
        audio = video_decode.read_audio(path, (start_frame - 1) * frame_time, len(indices) * frame_time,
                                        origin=source["start"])
        if ranged > len(indices):
            log.info(f"{ranged} frames -> {len(indices)} for {model}'s {sizes.MODELS[model]['frames']}n+1: the last "
                     f"{ranged - len(indices)} dropped" + (", audio cut to match" if audio is not None else ""))
        info = VideoInfo(audio=audio, **info)  # the audio output itself, first, for Get Video Info
        result["frames"] = len(indices)
        result["audio"] = "none" if audio is None else f"{audio['waveform'].shape[-1]} samples"
    return images, audio, info


def _decode(path, source, width, height, how, rate, start_frame, count, model, dtype):
    """(the batch, its source indices, the frames of the range before the cut: frame_indices'):
    each kept frame decoded once and fitted (resize.fit `how`) into a batch of `dtype` allocated at
    its final size, a repeated frame copied there from its first position."""
    from comfy.utils import ProgressBar

    indices, ranged = frame_indices(source, rate, start_frame, count, model)
    images = torch.empty((len(indices), height, width, 3), dtype=dtype)
    bar = ProgressBar(len(indices))
    for pixels, positions in video_decode.kept_frames(path, indices, source["frames"], to_end=count is None):
        resize.fit(pixels, images[positions[0]], how)
        for position in positions[1:]:
            images[position].copy_(images[positions[0]])
        bar.update_absolute(positions[-1] + 1, len(indices))
    return images, indices, ranged


class FrameRange(TypedDict):
    """The counts frame_count can take for a file, model, force_fps and start_frame, which Load
    Video's seconds slider spans: the loaded frame rate (force_fps, or the source's), the model's
    frame step (counts of the form step * n + 1: 4 for Wan and SCAIL, 1 for None) and the largest
    count, what an empty frame_count loads (on the frame rule already)."""
    fps: float
    step: int
    maximum: int


class LoadPreview(TypedDict):
    """What Load Video's preview shows, from the file's header and packets (no frame is loaded):
    the probe of the file (None when it cannot be read), the video_info the loader outputs without
    its audio (no sample is read; the probe says whether the file has audio; None on an error), the frames from start_frame on at the kept rate (what an empty frame_count stands
    for, before the cut to the model's frame rule; None when force_fps or start_frame is wrong), the
    FrameRange (None when force_fps, start_frame or model is wrong; it does not depend on
    frame_count, so it comes with frame_count's own errors too) and the error the loader raises,
    word for word (None when it loads)."""
    source: Optional[dict]
    info: Optional[VideoInfo]
    available: Optional[int]
    frame_range: Optional[FrameRange]
    error: Optional[str]


def preview(path, model, resolution, orientation, force_fps, start_frame, frame_count, precision=FP16, probe=None):
    """The LoadPreview of Load Video's widget values on the video at `path`, from the loader's own
    checks and frame selection. Its counts are the container's: should the decoder disagree when
    the video loads, the loader goes by the decoder's (load_video). `probe` as plan's."""
    from functools import lru_cache

    from av.error import FFmpegError

    probe = lru_cache(maxsize=1)(probe or video_decode.probe)
    answer = LoadPreview(source=None, info=None, available=None, frame_range=None, error=None)
    try:
        planned = plan(path, model, resolution, orientation, force_fps, frame_count, precision, probe)
        indices, _ = frame_indices(planned["source"], planned["rate"], start_frame, planned["count"], model)
        answer["info"] = video_info(model, resolution, planned, len(indices))
    except (ValueError, FFmpegError) as error:
        answer["error"] = str(error)
    # what the file and force_fps alone give, whatever else is wrong: an error met here is one
    # plan or frame_indices raises too, so "error" above already names it or one met before it
    try:
        source = probe(path)
        answer["source"] = dict(source)
        rate = parse_force_fps(force_fps)
        kept = video_decode.select_frames(source["fps"], source["frames"], rate)
        if 1 <= start_frame <= len(kept):
            answer["available"] = len(kept) - start_frame + 1
            if model in sizes.MODELS:
                step = sizes.MODELS[model]["frames"]
                answer["frame_range"] = FrameRange(fps=loaded_rate(rate, source), step=step,
                                                   maximum=len(loaded_frames(kept, start_frame, None, step)))
    except (ValueError, FFmpegError):
        pass
    return answer


def load_reference_image(path, width, height):
    """(IMAGE [1, height, width, 3], MASK, source IMAGE [1, H, W, 3], source MASK) of the image file
    at `path`, loaded as core's Load Image does (EXIF orientation applied, alpha -> MASK as
    1 - alpha; its first frame). The first two are fitted to `width` x `height` by a centre crop and
    lanczos, the MASK fitted the same way; the source two are the image at its own resolution, not
    resized. Without alpha both MASKs are core's 64x64 zeros."""
    import numpy as np
    import node_helpers
    from PIL import Image, ImageOps

    source = node_helpers.pillow(Image.open, path)
    source = node_helpers.pillow(ImageOps.exif_transpose, source)
    rgb = np.array(source.convert("RGB"))
    image = torch.empty((1, height, width, 3), dtype=torch.float32)
    resize.fit(rgb, image[0])
    source_image = torch.empty((1, *rgb.shape), dtype=torch.float32)
    resize.store(rgb, source_image[0])
    if "A" not in source.getbands():
        return image, torch.zeros((1, 64, 64), dtype=torch.float32), source_image, torch.zeros((1, 64, 64), dtype=torch.float32)
    alpha = np.array(source.getchannel("A"))
    mask = torch.empty((1, height, width), dtype=torch.float32)
    resize.fit(alpha, mask[0])
    source_mask = torch.empty((1, *alpha.shape), dtype=torch.float32)
    resize.store(alpha, source_mask[0])
    return image, mask.neg_().add_(1), source_image, source_mask.neg_().add_(1)


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
