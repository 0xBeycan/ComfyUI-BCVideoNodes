"""Video files from IMAGE frames with PyAV, one frame at a time: the codecs the Save Video node
offers and the values each codec's crf, preset and pix_fmt widgets take, the RGB -> YUV conversion
with its colour tags, the audio track cut to the video's length, and the container's metadata.

Colour: the frames are taken as sRGB-encoded RGB and written as BT.709 YUV in limited (tv) range,
tagged BT.709 matrix, primaries and transfer. sRGB and BT.709 share primaries and matrix; the
transfer tag is BT.709 rather than sRGB so that players and video platforms show the pixels as
they are instead of converting them.
- 8 bits: FFmpeg's scaler (swscale), bicubic, as the ffmpeg command line converts: the frames are
  rounded to 8-bit RGB, Y is 16 + 219 * luma, chroma is downsampled by swscale's bicubic filter.
- 10 bits: the BT.709 formula on the float frames, Y = 64 + 876 * luma, chroma 512 + 896 * (B - Y)
  or (R - Y) scaled, averaged over each 2x2 block. swscale writes 10-bit levels as the 8-bit ones
  * 1023 / 255 (white at 942.8, not 940), which standard decoders show 0.3% too bright.

PyAV is imported inside the functions that use it.
"""

import math
from fractions import Fraction

X264_PRESETS = ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow", "placebo"]

# codec -> the encoder, the container, the audio encoder, the values of the crf, preset and pix_fmt
# widgets (preset "option" is the encoder option the preset widget sets), and fixed encoder options.
# The codec names are stored in saved workflows: add codecs, never rename or remove one.
CODECS = {
    "h264-mp4": {
        "encoder": "libx264", "extension": "mp4", "audio": "aac",
        "crf": {"default": 19, "min": 0, "max": 51},
        "preset": {"option": "preset", "values": X264_PRESETS, "default": "medium"},
        "pix_fmt": {"values": ["yuv420p", "yuv420p10le", "yuv444p"], "default": "yuv420p"},
        "options": {},
    },
    "h265-mp4": {
        "encoder": "libx265", "extension": "mp4", "audio": "aac", "tag": "hvc1",
        "crf": {"default": 22, "min": 0, "max": 51},
        "preset": {"option": "preset", "values": X264_PRESETS, "default": "medium"},
        "pix_fmt": {"values": ["yuv420p10le", "yuv420p"], "default": "yuv420p10le"},
        "options": {"x265-params": "log-level=error"},
    },
    "av1-webm": {
        "encoder": "libsvtav1", "extension": "webm", "audio": "libopus",
        "crf": {"default": 23, "min": 1, "max": 63},
        "preset": {"option": "preset", "values": [str(p) for p in range(14)], "default": "8"},
        "pix_fmt": {"values": ["yuv420p10le", "yuv420p"], "default": "yuv420p10le"},
        "options": {},
    },
    "vp9-webm": {
        "encoder": "libvpx-vp9", "extension": "webm", "audio": "libopus",
        "crf": {"default": 20, "min": 0, "max": 63},
        "preset": {"option": "cpu-used", "values": [str(p) for p in range(6)], "default": "1"},
        "pix_fmt": {"values": ["yuv420p"], "default": "yuv420p"},
        "options": {"row-mt": "1"},
    },
}
WIDGETS = ("crf", "preset", "pix_fmt")  # the widgets whose values depend on the codec
AUDIO_BIT_RATE = {"aac": 96000, "libopus": 64000}  # bits per second per channel
SVT_LOG_LEVEL = "2"  # SVT-AV1 prints its info banner on every encode; 2 keeps warnings and errors


def widget_values(name):
    """What the `name` codec's crf, preset and pix_fmt widgets offer: the table the node hands the
    frontend."""
    return {widget: {k: v for k, v in CODECS[name][widget].items() if k != "option"} for widget in WIDGETS}


def union(widget):
    """Every value any codec offers for the combo `widget`, in table order: what the node declares
    so that ComfyUI accepts each codec's values."""
    values = []
    for codec in CODECS.values():
        values += [v for v in codec[widget]["values"] if v not in values]
    return values


def check_settings(codec, crf, preset, pix_fmt):
    """Raises ValueError naming the widget when a value is not one `codec` takes."""
    if codec not in CODECS:
        raise ValueError(f"codec {codec!r} is not one of {', '.join(CODECS)}")
    spec = CODECS[codec]
    if not spec["crf"]["min"] <= crf <= spec["crf"]["max"]:
        raise ValueError(f"crf {crf} is outside {codec}'s range {spec['crf']['min']}-{spec['crf']['max']}")
    for widget, value in (("preset", preset), ("pix_fmt", pix_fmt)):
        if value not in spec[widget]["values"]:
            raise ValueError(f"{widget} {value!r} is not one of {codec}'s: {', '.join(spec[widget]['values'])}")


def check_encoders(codec):
    """Raises RuntimeError when this machine's PyAV lacks `codec`'s video or audio encoder, naming
    it and listing the encoders of the table this PyAV has."""
    import av

    missing = [name for name in (CODECS[codec]["encoder"], CODECS[codec]["audio"]) if name not in av.codecs_available]
    if missing:
        present = sorted({c[key] for c in CODECS.values() for key in ("encoder", "audio")} & set(av.codecs_available))
        raise RuntimeError(f"{codec} needs the encoder {', '.join(missing)}, which this machine's PyAV "
                           f"{av.__version__} does not have. Encoders it has: {', '.join(present) or 'none'}. "
                           f"Choose another codec, or install a PyAV build that includes it.")


def frame_rate(fps):
    """`fps` as the stream's exact rate: 29.97002997 -> 30000/1001, 24.0 -> 24."""
    return Fraction(fps).limit_denominator(1001000)


def even_size(height, width):
    """The size a frame is written at: odd sides padded to even, as 4:2:0 needs."""
    return height + height % 2, width + width % 2


def audio_samples(frames, fps, sample_rate):
    """How many audio samples cover `frames` video frames at `fps`."""
    return round(frames * sample_rate / fps)


KR, KB = 0.2126, 0.0722  # BT.709 luma weights of red and blue
TV_10BIT = {"black": 64, "luma": 876, "zero": 512, "chroma": 896}  # BT.709 10-bit limited range: the 8-bit levels x 4


class _FrameConverter:
    """IMAGE frames [H, W, C] in 0..1 -> YUV frames of `pix_fmt` at even size, through
    preallocated buffers, odd sides padded by repeating the last row / column. 8-bit formats go
    through swscale from RGB rounded to 8 bits (x * 255 + 0.5 truncated); yuv420p10le is computed
    from the float frames with the BT.709 formula."""

    def __init__(self, height, width, pix_fmt):
        import numpy as np
        import torch
        from av.video.reformatter import VideoReformatter

        self.height, self.width = height, width
        self.pix_fmt = pix_fmt
        h, w = even_size(height, width)
        self.rgb = torch.empty((h, w, 3), dtype=torch.float32)
        if pix_fmt == "yuv420p10le":
            self.luma = torch.empty((h, w), dtype=torch.float32)
            self.difference = torch.empty((h, w), dtype=torch.float32)
            self.planes = np.empty((h * 3 // 2, w), dtype=np.uint16)  # Y, then U, then V, as PyAV lays them out
        else:
            self.ints = torch.empty((h, w, 3), dtype=torch.uint8)
            self.reformatter = VideoReformatter()

    def __call__(self, frame):
        h, w = self.height, self.width
        if tuple(frame.shape[:2]) != (h, w):
            raise ValueError(f"frame of size {frame.shape[1]}x{frame.shape[0]} in a video of {w}x{h}: "
                             f"every frame must have the first frame's size")
        rgb = self.rgb
        rgb[:h, :w].copy_(frame[..., :3])
        if rgb.shape[0] > h:
            rgb[h:, :w] = rgb[h - 1:h, :w]
        if rgb.shape[1] > w:
            rgb[:, w:] = rgb[:, w - 1:w]
        return self._ten_bit() if self.pix_fmt == "yuv420p10le" else self._eight_bit()

    def _eight_bit(self):
        import av
        from av.video.reformatter import ColorRange, Colorspace, Interpolation

        # rounded half up: x * 255 + 0.5, truncated by the cast
        self.ints.copy_(self.rgb.mul_(255.0).add_(0.5).clamp_(0.0, 255.0))
        rgb = av.VideoFrame.from_ndarray(self.ints.numpy(), format="rgb24")
        return self.reformatter.reformat(rgb, format=self.pix_fmt, dst_colorspace=Colorspace.ITU709,
                                         dst_color_range=ColorRange.MPEG, interpolation=Interpolation.BICUBIC)

    def _ten_bit(self):
        import av
        import numpy as np
        import torch

        rgb, luma, difference = self.rgb.clamp_(0.0, 1.0), self.luma, self.difference
        h, w = luma.shape
        torch.mul(rgb[..., 0], KR, out=luma).add_(rgb[..., 1], alpha=1 - KR - KB).add_(rgb[..., 2], alpha=KB)
        planes = self.planes.reshape(-1)
        size = h * w
        chroma = [planes[size:size * 5 // 4].reshape(h // 2, w // 2), planes[size * 5 // 4:].reshape(h // 2, w // 2)]
        for plane, channel, weight in zip(chroma, (2, 0), (KB, KR)):
            torch.sub(rgb[..., channel], luma, out=difference)
            block = difference.view(h // 2, 2, w // 2, 2).mean((1, 3))
            _store(block.mul_(TV_10BIT["chroma"] / (2 * (1 - weight))).add_(TV_10BIT["zero"]), plane)
        _store(luma.mul_(TV_10BIT["luma"]).add_(TV_10BIT["black"]), planes[:size].reshape(h, w))
        frame = av.VideoFrame.from_ndarray(self.planes, format="yuv420p10le")
        frame.colorspace = 1  # AVCOL_SPC_BT709
        frame.color_range = 1  # AVCOL_RANGE_MPEG
        return frame


def _store(levels, plane):
    """Float `levels` rounded (half up) into the uint16 `plane`, within 0..1023."""
    import numpy as np

    np.copyto(plane, levels.add_(0.5).clamp_(0.0, 1023.0).numpy(), casting="unsafe")


def _video_stream(container, codec, height, width, fps, crf, preset, pix_fmt, keyframe_interval):
    from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

    spec = CODECS[codec]
    stream = container.add_stream(spec["encoder"], rate=frame_rate(fps))
    stream.height, stream.width = even_size(height, width)
    stream.pix_fmt = pix_fmt
    stream.options = {**spec["options"], "crf": str(crf), spec["preset"]["option"]: str(preset)}
    ctx = stream.codec_context
    # frame and slice threads, as FFmpeg's own default: PyAV's slice-only default makes x264 cut
    # every frame into slices, which costs compression and changes the file
    ctx.thread_type = "AUTO"
    if keyframe_interval:
        ctx.gop_size = keyframe_interval
    if "tag" in spec:
        ctx.codec_tag = spec["tag"]
    ctx.colorspace = 1  # AVCOL_SPC_BT709
    ctx.color_primaries = ColorPrimaries.BT709
    ctx.color_trc = ColorTrc.BT709
    ctx.color_range = ColorRange.MPEG
    return stream


class _AudioTrack:
    """The AUDIO input's first batch item written into the container in pieces as the video
    advances, so the file interleaves; it ends where the video ends (or where the audio does)."""

    def __init__(self, container, codec, audio, fps):
        import av

        self.wave = audio["waveform"][0]  # [channels, samples], a view
        self.sample_rate = int(audio["sample_rate"])
        self.fps = fps
        self.written = 0
        channels = self.wave.shape[0]
        self.layout = av.AudioLayout(f"{channels}c").name  # FFmpeg's default layout for the count
        encoder = CODECS[codec]["audio"]
        rates = av.codec.Codec(encoder, "w").audio_rates
        rate = self.sample_rate if not rates or self.sample_rate in rates else 48000
        self.stream = container.add_stream(encoder, rate=rate, layout=self.layout)
        self.stream.bit_rate = AUDIO_BIT_RATE[encoder] * channels

    def write_until(self, container, frames):
        """Writes the samples that cover the first `frames` video frames."""
        import av
        import numpy as np

        end = min(self.wave.shape[-1], audio_samples(frames, self.fps, self.sample_rate))
        if end <= self.written:
            return
        piece = np.ascontiguousarray(self.wave[:, self.written:end].float().cpu().numpy())
        chunk = av.AudioFrame.from_ndarray(piece, format="fltp", layout=self.layout)
        chunk.sample_rate = self.sample_rate
        chunk.pts = self.written
        chunk.time_base = Fraction(1, self.sample_rate)
        container.mux(self.stream.encode(chunk))
        self.written = end

    def close(self, container):
        container.mux(self.stream.encode(None))


def write_video(path, frames, fps, codec, crf, preset, pix_fmt, audio=None, metadata=None, keyframe_interval=None):
    """Encodes `frames` (an iterable of [H, W, C] float tensors in 0..1, C >= 3, all of the first's
    size) into the file `path` with `codec` (a CODECS key). `audio` (ComfyUI AUDIO) is muxed in and
    cut to the video's length; `metadata` (name -> JSON-able value) is written into the container
    as JSON strings. Returns the number of frames written."""
    import itertools
    import json
    import os

    import av

    check_settings(codec, crf, preset, pix_fmt)
    check_encoders(codec)
    if CODECS[codec]["encoder"] == "libsvtav1":
        os.environ.setdefault("SVT_LOG", SVT_LOG_LEVEL)
    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        raise ValueError("no frames to write: the images input is empty")
    height, width = int(first.shape[0]), int(first.shape[1])
    # mp4: the custom tags (workflow, prompt) kept, the index at the front so a browser plays it
    # while it loads
    options = {"movflags": "use_metadata_tags+faststart"} if CODECS[codec]["extension"] == "mp4" else {}
    count = 0
    try:
        with av.open(path, "w", options=options) as container:
            for key, value in (metadata or {}).items():
                container.metadata[key] = value if isinstance(value, str) else json.dumps(value)
            stream = _video_stream(container, codec, height, width, fps, crf, preset, pix_fmt, keyframe_interval)
            track = _AudioTrack(container, codec, audio, fps) if audio is not None else None
            convert = _FrameConverter(height, width, pix_fmt)
            for frame in itertools.chain((first,), frames):
                yuv = convert(frame)
                yuv.pts = count
                container.mux(stream.encode(yuv))
                count += 1
                if track is not None:
                    track.write_until(container, count)
            container.mux(stream.encode(None))
            if track is not None:
                track.close(container)
    except BaseException:
        # a half-written file would take the next counter value and not play
        if os.path.exists(path):
            os.remove(path)
        raise
    return count


def has_audio(audio):
    """Whether an AUDIO input carries samples."""
    return audio is not None and audio["waveform"].shape[-1] > 0 and math.prod(audio["waveform"].shape[:-1]) > 0
