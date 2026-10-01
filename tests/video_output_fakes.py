"""The `video` Names the Save Video / Video Comparer test bodies read pack names through
(tests/names.py), and the synthetic clips and the decoder they check the files with. Standard
library only at module level; torch, numpy and PyAV are imported inside the helpers."""
from names import Names, refs

video = Names("video_output", {
    **refs("libs.video_encode", "CODECS", "WIDGETS", "AUDIO_BIT_RATE", "widget_values", "union", "check_settings",
           "check_encoders", "frame_rate", "even_size", "audio_samples", "write_video", "has_audio",
           "_FrameConverter"),
    **refs("libs.video_compare", "side_by_side_geometry", "fit_into", "side_by_side_frames"),
    **refs("nodes.video_output", "BCVSaveVideo", "BCVVideoComparer", "UI_KEY", "FPS", "VIDEO", "COMPARER_CODEC",
           "COMPARER_CRF"),
})

# colour patches, strong but inside the gamut (decoding them does not clip): where decoding with the
# wrong matrix (BT.601 for BT.709) is off by about 10 levels
PATCHES = ((0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8), (0.8, 0.8, 0.2), (0.2, 0.8, 0.8), (0.8, 0.2, 0.8))
PATCH = 16  # patch side in pixels; a multiple of 2 so 4:2:0 chroma blocks never straddle two patches


def clip(frames=9, height=64, width=96):
    """[frames, height, width, 3] float32 in 0..1: a smooth colour gradient that moves one pixel
    per frame, with the colour PATCHES in a row along the top."""
    import torch

    y = torch.linspace(0, 1, height).view(height, 1)
    x = torch.linspace(0, 1, width + frames).view(1, width + frames)
    out = torch.empty((frames, height, width, 3), dtype=torch.float32)
    for i in range(frames):
        xs = x[:, i:i + width]
        out[i, ..., 0] = 0.15 + 0.7 * xs * torch.ones_like(y)
        out[i, ..., 1] = 0.15 + 0.7 * y * torch.ones_like(xs)
        out[i, ..., 2] = 0.5 - 0.3 * (xs - y)
        for k, colour in enumerate(PATCHES):
            out[i, :PATCH, k * PATCH:(k + 1) * PATCH] = torch.tensor(colour)
    return out


def patch_interiors(frames):
    """[..., 3] pixels of `frames` [N, H, W, 3] inside the PATCHES, 3 pixels in from every edge
    (where 4:2:0 chroma has no neighbour of another colour)."""
    return frames[:, 3:PATCH - 3, :PATCH * len(PATCHES)].unflatten(2, (len(PATCHES), PATCH))[:, :, :, 3:PATCH - 3]


def smooth(frames):
    """`frames` [N, H, W, 3] below the PATCHES row, 4 pixels in from the other edges (where 4:2:0
    chroma meets the replicated border): the gradient only."""
    return frames[:, PATCH + 4:-4, 4:-4]


def tone(seconds, rate=44100, channels=2):
    """ComfyUI AUDIO: a sine per channel, `seconds` long."""
    import math

    import torch

    t = torch.arange(round(seconds * rate), dtype=torch.float32) / rate
    wave = torch.stack([0.3 * torch.sin(2 * math.pi * (440 + 110 * c) * t) for c in range(channels)])
    return {"waveform": wave[None], "sample_rate": rate}


def decode(path, bt601=False):
    """What a file holds: the video stream's codec, pix_fmt, size, rate, colour tags, codec tag and
    frames (float32 [N, H, W, 3] in 0..1, converted to 8-bit RGB the way Load Video converts:
    following the stream's tags, or with BT.601 when `bt601`, through swscale's full-chroma,
    accurate-rounding path), the container metadata, and the audio stream's codec, rate and
    decoded sample count."""
    import av
    import numpy as np
    import torch
    from av.video.reformatter import Colorspace, Interpolation

    flags = Interpolation.BILINEAR | Interpolation.FULL_CHR_H_INT | Interpolation.ACCURATE_RND

    with av.open(path) as container:
        v = container.streams.video[0]
        info = {"codec": v.codec_context.name, "pix_fmt": v.codec_context.pix_fmt, "width": v.codec_context.width,
                "height": v.codec_context.height, "rate": v.average_rate, "tag": v.codec_context.codec_tag,
                "metadata": dict(container.metadata)}
        frames = []
        for frame in container.decode(v):
            info.update(colorspace=int(frame.colorspace), color_range=int(frame.color_range),
                        primaries=int(v.codec_context.color_primaries), trc=int(v.codec_context.color_trc))
            rgb = frame.reformat(format="rgb24", src_colorspace=Colorspace.ITU601 if bt601 else None,
                                 interpolation=flags).to_ndarray()
            frames.append(torch.from_numpy(rgb.astype(np.float32) / 255.0))
        info["frames"] = torch.stack(frames) if frames else None
    with av.open(path) as container:
        if container.streams.audio:
            a = container.streams.audio[0]
            info.update(audio_codec=a.codec_context.name, audio_rate=a.rate, audio_channels=a.channels,
                        audio_samples=sum(f.samples for f in container.decode(a)))
    return info
