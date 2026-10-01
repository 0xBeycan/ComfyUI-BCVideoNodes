"""The flows of the video input nodes: Load Video (a file decoded frame by frame, cropped and
resized to the model's generation size straight into one preallocated IMAGE batch, with its audio
and its video_info), Load Reference Image (an image fitted to the video's loaded size) and Conform
Video (a clip fitted to the nearest platform size).

The full-resolution clip never exists: one source frame is decoded at a time, and the batch is
allocated once, at its final frame count."""
import torch

from ..libs import log, resize, video_decode
from ..libs import video_sizes as sizes
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


def load_video(path, model, resolution, orientation, force_fps, start_frame, frame_count):
    """(IMAGE [N, H, W, 3] float32, AUDIO or None, VideoInfo) of the video file at `path`, as Load
    Video's widgets say. N is fixed before the batch is allocated: the container's packets give the
    frame count; should the decoder disagree, the load runs once more with the decoder's count."""
    size = sizes.model_size(model, resolution)
    rate = parse_force_fps(force_fps)
    count = parse_frame_count(frame_count)
    if orientation not in sizes.ORIENTATIONS:
        raise ValueError(f"orientation {orientation!r} is not one of {', '.join(sizes.ORIENTATIONS)}.")
    source = video_decode.probe(path)
    if rate is not None and abs(rate - source["fps"]) <= source["fps"] * FORCE_FPS_TOLERANCE:
        # the video's own rate, as typed (29.97 for 30000/1001): every frame, not a grid a hair
        # slower that drops frame 1
        log.info(f"force_fps {rate:g} is the video's frame rate ({source['fps']:.6g}): every frame kept")
        rate = None
    elif rate is not None and rate > source["fps"]:
        # the loader keeps or drops real frames, it never repeats one to raise the rate: a repeat
        # fakes the frame rate, and a typo such as 300 would multiply the batch
        raise ValueError(f"force_fps {rate:g} is above the video's frame rate {source['fps']:.6g}: the loader only "
                         f"lowers the frame rate (it keeps or drops real frames, never repeats them). Leave force_fps "
                         f"empty or set it to {source['fps']:.6g} or less.")
    turned = sizes.orientation_of(source["width"], source["height"]) if orientation == sizes.AUTO else orientation
    width, height = sizes.oriented(size, turned)
    result = {}
    with log.step(f"loading {model} {resolution} {turned} ({width}x{height}) from a "
                  f"{source['width']}x{source['height']} video", result):
        try:
            loaded = _decode(path, source, width, height, rate, start_frame, count, model)
        except video_decode.FrameCountChanged as changed:
            log.warning(f"the container promised {source['frames']} frames, the decoder gave {changed.frames}; "
                        f"loading again with {changed.frames}")
            source["frames"] = changed.frames
            loaded = None
        if loaded is None:  # outside the except block: its traceback would keep the first batch alive
            loaded = _decode(path, source, width, height, rate, start_frame, count, model)
        images, indices = loaded
        loaded_fps = rate if rate is not None else source["fps"]
        frame_time = 1 / loaded_fps
        audio = video_decode.read_audio(path, (start_frame - 1) * frame_time, len(indices) * frame_time,
                                        origin=source["start"])
        result["frames"] = len(indices)
        result["audio"] = "none" if audio is None else f"{audio['waveform'].shape[-1]} samples"
    info = VideoInfo(model=model, resolution=resolution, orientation=turned,
                     source_fps=source["fps"], source_frame_count=source["frames"],
                     source_duration=source["frames"] / source["fps"],
                     source_width=source["width"], source_height=source["height"],
                     loaded_fps=loaded_fps, loaded_frame_count=len(indices),
                     loaded_duration=len(indices) * frame_time, loaded_width=width, loaded_height=height)
    return images, audio, info


def _decode(path, source, width, height, rate, start_frame, count, model):
    """(the batch, its source indices): the kept frames decoded one at a time and fitted into a
    batch allocated at its final size."""
    from comfy.utils import ProgressBar

    kept = video_decode.select_frames(source["fps"], source["frames"], rate)
    indices = loaded_frames(kept, start_frame, count, sizes.MODELS[model]["frames"], rate)
    images = torch.empty((len(indices), height, width, 3), dtype=torch.float32)
    bar = ProgressBar(len(indices))
    for pixels, positions in video_decode.kept_frames(path, indices, source["frames"], to_end=count is None):
        resize.fit(pixels, images[positions[0]])
        for position in positions[1:]:
            images[position].copy_(images[positions[0]])
        bar.update_absolute(positions[-1] + 1, len(indices))
    return images, indices


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
    (video_sizes.conform_size), frame by frame into one preallocated batch; the input itself when
    it already has that size."""
    frames, height, width, channels = images.shape
    target_width, target_height = sizes.conform_size(width, height)
    if (target_width, target_height) == (width, height):
        log.info(f"conform: {width}x{height} is already a standard size, nothing to do")
        return images
    out = torch.empty((frames, target_height, target_width, channels), dtype=images.dtype, device=images.device)
    with log.step(f"conforming {frames} frames of {width}x{height} to {target_width}x{target_height} "
                  f"({how}, {method})"):
        for index in range(frames):
            resize.fit(images[index], out[index], how, method)
    return out
