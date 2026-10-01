"""Video files read with PyAV, one frame at a time: the frame rate and the frame count, the frames
kept on a force_fps time grid, each kept frame as RGB pixels, and the audio of a time range.

Frames are counted and indexed as the decoder outputs them (0-based), which is how the loader's
frame selection and its errors count them. Colour: YUV -> RGB with the stream's own matrix and
range (an untagged stream gets FFmpeg's default, BT.601 limited), the picture turned as the
stream's display matrix says."""
import numpy as np


class FrameCountChanged(Exception):
    """The decoder gave `frames` frames where the container's packets promised another count."""

    def __init__(self, frames):
        super().__init__(frames)
        self.frames = frames


def _video_stream(container, path):
    """The first video stream of `container` that is not a cover picture."""
    import av

    for stream in container.streams.video:
        if not stream.disposition & av.stream.Disposition.attached_pic:
            return stream
    raise ValueError(f"{path} has no video stream; pick a video file.")


def probe(path):
    """{fps, frames, width, height, start} of the video: the frame rate (the stream's average,
    else its base rate), the frame count (the packets that carry a frame: a demux, no decode),
    the size as displayed (turned by the display matrix) and the first frame's time in seconds."""
    import av

    with av.open(path) as container:
        stream = _video_stream(container, path)
        rate = stream.average_rate or stream.base_rate
        if not rate:
            raise ValueError(f"{path} gives no frame rate; re-encode it with a constant frame rate.")
        first = next(container.decode(stream), None)
        if first is None:
            raise ValueError(f"{path} has no decodable video frame; pick another file or re-encode it.")
        width, height = (first.height, first.width) if _turns(first) % 2 else (first.width, first.height)
        start = first.time or 0.0
    with av.open(path) as container:
        stream = _video_stream(container, path)
        frames = sum(1 for packet in container.demux(stream) if packet.size and not packet.is_discard)
    return {"fps": float(rate), "frames": frames, "width": width, "height": height, "start": start}


def select_frames(fps, frames, force_fps=None):
    """The source frame indices kept at `force_fps` (None: every frame), of `frames` frames at
    `fps`: real frames kept or dropped on the target time grid, never blended. Output frame m is
    the first source frame at or after m / force_fps (a frame would repeat were force_fps above
    fps; Load Video rejects that before it gets here). The time is accumulated in floats in this
    order on purpose, so the kept indices are the ones the established loaders keep, float
    boundaries included (30 -> 24 keeps 0, 2, 3, 4, 5, 7)."""
    base = 1 / fps
    target = base if force_fps is None else 1 / force_fps
    offset, index, kept = target, 0, []
    while True:
        if offset < target:
            index += 1
            if index >= frames:
                return kept
            offset += base
        if offset < target:
            continue
        offset -= target
        kept.append(index)


def _turns(frame):
    """Quarter turns (counter-clockwise) that show `frame` upright, from its display matrix."""
    return (frame.rotation // 90) % 4 if frame.rotation else 0


def rgb(frame):
    """`frame` as upright [H, W, 3] uint8 RGB. swscale's full-chroma, accurate-rounding path: its
    default path is ~2 levels dark on frames whose width is not a multiple of 16 (1080-wide)."""
    from av.video.reformatter import Interpolation

    flags = Interpolation.BILINEAR | Interpolation.FULL_CHR_H_INT | Interpolation.ACCURATE_RND
    pixels = frame.reformat(format="rgb24", interpolation=flags).to_ndarray()
    turns = _turns(frame)
    return np.rot90(pixels, turns) if turns else pixels


def kept_frames(path, indices, frames, to_end):
    """Yields (RGB pixels, positions) for each distinct source frame of `indices` (sorted, a
    repeat is one frame at several positions), one decoded frame at a time. With `to_end` it
    decodes to the end of the stream even when every kept frame is out. Raises
    FrameCountChanged when the stream ends and the decoded count is not `frames`."""
    import av

    with av.open(path) as container:
        stream = _video_stream(container, path)
        stream.thread_type = "AUTO"
        position, decoded = 0, 0
        for frame in container.decode(stream):
            if position < len(indices) and indices[position] == decoded:
                end = position
                while end < len(indices) and indices[end] == decoded:
                    end += 1
                yield rgb(frame), range(position, end)
                position = end
            decoded += 1
            if position == len(indices) and not to_end:
                return
    if decoded != frames:
        raise FrameCountChanged(decoded)


def read_audio(path, start, duration, origin=0.0):
    """ComfyUI AUDIO ({"waveform": [1, channels, samples] float32, "sample_rate"}) of the first
    audio stream from `start` seconds for `duration` seconds, times counted from `origin` (the
    first video frame's time), cut to the sample. Shorter when the audio ends first; None when the
    file has no audio stream."""
    import itertools

    import av
    import torch

    with av.open(path) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        rate = stream.rate
        first, count = round(start * rate), round(duration * rate)
        waveform = torch.zeros((1, stream.channels, count), dtype=torch.float32)
        resampler = av.AudioResampler(format="fltp")
        position, filled = None, 0
        for frame in itertools.chain(container.decode(stream), [None]):
            for chunk in resampler.resample(frame):
                if position is None:
                    position = round((chunk.time - origin) * rate) if chunk.time is not None else 0
                low, high = max(first, position), min(first + count, position + chunk.samples)
                if high > low:
                    samples = torch.from_numpy(chunk.to_ndarray()[:, low - position:high - position])
                    waveform[0, :, low - first:high - first] = samples
                    filled = high - first
                position += chunk.samples
            if position is not None and position >= first + count:
                break
    return {"waveform": waveform[..., :filled], "sample_rate": rate}
