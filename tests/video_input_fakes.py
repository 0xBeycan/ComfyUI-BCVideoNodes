"""The `video` Names the video input test bodies read pack names through (tests/names.py), and the
clips they load: small lossless files (FFV1 video, PCM audio, Matroska) written with PyAV into a
test's tmp dir, so a decoded pixel is exactly the pixel written."""
from fractions import Fraction

from names import Names, Ref, Seam, refs

video = Names("video", {
    **refs("libs.video_decode", "select_frames", "kept_frames", "read_audio", "rgb", "FrameCountChanged"),
    "probe": Seam(Ref("libs.video_decode", "probe")),
    **refs("libs.resize", "fit", "crop_box", "contain_box", "CROP", "PAD", "FITS", "LANCZOS", "METHODS"),
    **refs("libs.video_sizes", "MODELS", "RESOLUTIONS", "ORIENTATIONS", "CONFORM_SIZES", "conform_size",
           "model_size", "orientation_of", "oriented"),
    **refs("libs.video_info", "VideoInfo", "COMFY_TYPES"),
    **refs("pipelines.video_input", "load_video", "loaded_frames", "parse_force_fps", "parse_frame_count",
           "load_reference_image", "conform_video"),
    **refs("nodes.video_input", "BCVLoadVideo", "BCVGetVideoInfo", "BCVLoadReferenceImage", "BCVConformVideo"),
})

# a grey clip's frame i has luma 16 + LUMA_STEP * i: distinct after the RGB conversion
LUMA_STEP = 2


def yuv_frame(y, u, v):
    """A yuv420p av.VideoFrame from its planes (numpy uint8: y [H, W], u and v [H/2, W/2])."""
    import av
    import numpy as np

    frame = av.VideoFrame(y.shape[1], y.shape[0], "yuv420p")
    for plane, values in zip(frame.planes, (y, u, v)):
        rows = np.zeros((plane.height, plane.line_size), np.uint8)
        rows[:, :plane.width] = values
        plane.update(rows.tobytes())
    return frame


def write_clip(path, frames, rate=30, *, colorspace=None, color_range=None, rotation=None, audio=None):
    """Writes `frames` (yuv420p av.VideoFrames, frame i at time i / rate) as an FFV1 Matroska clip,
    tagged with `colorspace` / `color_range` (FFmpeg's numbers) when given and turned by the display
    rotation `rotation` (degrees counter-clockwise). `audio`: (int16 [channels, samples], sample
    rate), written as PCM from time 0."""
    import av

    rate = Fraction(rate)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=rate)
        stream.width, stream.height, stream.pix_fmt = frames[0].width, frames[0].height, "yuv420p"
        if colorspace is not None:
            stream.codec_context.colorspace = colorspace
        if color_range is not None:
            stream.codec_context.color_range = color_range
        if rotation is not None:
            stream.set_display_rotation(rotation)
        sound = None
        if audio is not None:
            samples, sample_rate = audio
            sound = container.add_stream("pcm_s16le", rate=sample_rate, layout="stereo" if len(samples) == 2 else "mono")
        for index, frame in enumerate(frames):
            if colorspace is not None:
                frame.colorspace = colorspace
            if color_range is not None:
                frame.color_range = color_range
            frame.pts, frame.time_base = index, 1 / rate
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
        if sound is not None:
            packed = samples.T.reshape(1, -1).copy()
            chunk = av.AudioFrame.from_ndarray(packed, format="s16", layout="stereo" if len(samples) == 2 else "mono")
            chunk.sample_rate, chunk.pts = sample_rate, 0
            container.mux(sound.encode(chunk))
            container.mux(sound.encode())
    return str(path)


def grey_clip(path, count, rate=30, width=64, height=32, **kwargs):
    """A clip of `count` flat grey frames, frame i at luma 16 + LUMA_STEP * i (neutral chroma)."""
    import numpy as np

    frames = [yuv_frame(np.full((height, width), 16 + LUMA_STEP * i, np.uint8),
                        np.full((height // 2, width // 2), 128, np.uint8),
                        np.full((height // 2, width // 2), 128, np.uint8)) for i in range(count)]
    return write_clip(path, frames, rate, **kwargs)


def grey_index(value):
    """The frame index of a grey clip's frame from its RGB value in [0, 1] (limited-range BT.601
    or BT.709: on grey both give (Y - 16) * 255 / 219)."""
    return round(value * 219 / LUMA_STEP)


def ramp_audio(seconds, sample_rate=48000):
    """A stereo int16 ramp: sample s is s % 30000 on the left, its negative on the right, so a
    sample's value tells its position."""
    import numpy as np

    ramp = (np.arange(round(seconds * sample_rate)) % 30000).astype(np.int16)
    return np.stack([ramp, -ramp]), sample_rate
