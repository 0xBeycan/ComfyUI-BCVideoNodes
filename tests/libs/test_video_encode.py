"""The video encoder of libs/video_encode.py: the codec table against the encoders of this
machine's PyAV, the files it writes decoded back (frame count, rate, pix_fmt, colour tags, the
colours following the tags, the audio cut to the video), the conversion's rounding and 10-bit
precision against the BT.709 formula, the metadata, and the errors. Small synthetic clips.
"""
import json
import os
import re
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from video_output_fakes import clip, decode, patch_interiors, smooth, tone, video  # noqa: E402

FPS = 30000 / 1001
BT709 = 1  # AVCOL_SPC_BT709, AVCOL_PRI_BT709 and AVCOL_TRC_BT709 are all 1
TV_RANGE = 1  # AVCOL_RANGE_MPEG
PIX_FMTS = [(codec, pix_fmt) for codec, spec in video.CODECS.items() for pix_fmt in spec["pix_fmt"]["values"]]
# colour error (8-bit levels) of a round trip at the codec's default crf, on the gradient: the mean
# and the largest error. A wrong matrix or range is caught on the colour patches (test_round_trip).
TOLERANCE = {"mean": 1.5, "max": 24.0}


def colour_error(decoded, frames):
    """(mean error, largest error) on the gradient, in 8-bit levels."""
    error = smooth(decoded - frames).abs() * 255
    return error.mean(), error.max()


def encoded(tmp_path, codec, pix_fmt=None, frames=None, audio=None, name="clip", **settings):
    spec = video.CODECS[codec]
    path = str(tmp_path / f"{name}.{spec['extension']}")
    values = {"crf": spec["crf"]["default"], "preset": spec["preset"]["default"], **settings}
    count = video.write_video(path, clip() if frames is None else frames, FPS, codec, values["crf"], values["preset"],
                              pix_fmt or spec["pix_fmt"]["default"], audio=audio)
    return path, count


# ---- the table -----------------------------------------------------------------------------------

def test_the_codecs_and_their_widgets():
    assert list(video.CODECS) == ["h264-mp4", "h265-mp4", "av1-webm", "vp9-webm"]
    assert video.WIDGETS == ("crf", "preset", "pix_fmt")
    for name, spec in video.CODECS.items():
        values = video.widget_values(name)
        assert set(values) == set(video.WIDGETS) and "option" not in values["preset"]
        assert spec["crf"]["min"] <= spec["crf"]["default"] <= spec["crf"]["max"]
        assert spec["preset"]["default"] in spec["preset"]["values"]
        assert spec["pix_fmt"]["default"] in spec["pix_fmt"]["values"]
    assert video.union("pix_fmt") == ["yuv420p", "yuv420p10le", "yuv444p"]
    assert video.union("preset")[:10] == video.CODECS["h264-mp4"]["preset"]["values"]
    assert set(video.union("preset")) == {v for spec in video.CODECS.values() for v in spec["preset"]["values"]}


@pytest.mark.parametrize("codec", list(video.CODECS))
def test_every_value_is_one_its_encoder_takes(codec):
    spec = video.CODECS[codec]
    encoder = av.codec.Codec(spec["encoder"], "w")
    assert set(spec["pix_fmt"]["values"]) <= {f.name for f in encoder.video_formats}
    assert 48000 in av.codec.Codec(spec["audio"], "w").audio_rates


def x26x_crf(path):
    """The crf x264 / x265 wrote into their settings SEI, what the encoder really ran with (x264
    runs crf 0 as lossless, constant qp 0: 0 then)."""
    data = open(path, "rb").read()
    found = re.search(rb" crf=([0-9.]+)", data) or re.search(rb" rc=cqp .*? qp=(\d+)", data)
    return float(found.group(1)) if found else None


@pytest.mark.parametrize("codec", list(video.CODECS))
def test_every_preset_and_the_crf_range_encode(tmp_path, codec):
    """Each preset and both ends of the crf range open the encoder and write a file (an option
    value the encoder rejects fails the open); x264 and x265 clamp a crf out of their range
    silently, so their SEI must show the crf asked for."""
    spec = video.CODECS[codec]
    frames = clip(3, 64, 64)  # x265 gives the packets of a 2-frame clip garbage timestamps
    for preset in spec["preset"]["values"]:
        path, count = encoded(tmp_path, codec, frames=frames, preset=preset, name=f"preset_{preset}")
        assert count == 3 and decode(path)["frames"].shape[0] == 3
    for crf in (spec["crf"]["min"], spec["crf"]["max"]):
        path, _ = encoded(tmp_path, codec, frames=frames, crf=crf, name=f"crf_{crf}")
        assert decode(path)["frames"].shape[0] == 3
        if spec["encoder"] in ("libx264", "libx265"):
            assert x26x_crf(path) == crf


# ---- the files -------------------------------------------------------------------------------

@pytest.mark.parametrize("codec,pix_fmt", PIX_FMTS)
def test_round_trip(tmp_path, codec, pix_fmt):
    frames = clip()
    path, count = encoded(tmp_path, codec, pix_fmt, frames=frames, audio=tone(2.0))
    got = decode(path)
    assert count == 9 and got["frames"].shape == frames.shape
    assert got["rate"] == Fraction(30000, 1001)
    assert got["pix_fmt"] == pix_fmt
    assert (got["colorspace"], got["primaries"], got["trc"], got["color_range"]) == (BT709, BT709, BT709, TV_RANGE)
    assert got["tag"] == video.CODECS[codec].get("tag", got["tag"])
    # the colours, decoded following the tags
    mean, largest = colour_error(got["frames"], frames)
    assert mean < TOLERANCE["mean"] and largest < TOLERANCE["max"], (mean, largest)
    # no matrix mismatch: the colour patches decoded as BT.601 are far off, as BT.709 they are not
    right = patch_interiors((got["frames"] - frames).abs() * 255).mean()
    wrong = patch_interiors((decode(path, bt601=True)["frames"] - frames).abs() * 255).mean()
    assert right < 2 and wrong > 6, (right, wrong)
    # the audio: 2 s cut to the video's 9 frames, give or take one codec frame of padding
    rate = got["audio_rate"]
    expected = video.audio_samples(9, FPS, rate)
    assert got["audio_codec"] == {"aac": "aac", "libopus": "opus"}[video.CODECS[codec]["audio"]]
    assert got["audio_channels"] == 2
    assert expected - 1 <= got["audio_samples"] <= expected + 1024, (got["audio_samples"], expected)


@pytest.mark.parametrize("pix_fmt", video.CODECS["h264-mp4"]["pix_fmt"]["values"])
def test_lossless_coding_leaves_the_conversion(tmp_path, pix_fmt):
    """H.264 at crf 0 (lossless at 8 bits, near it at 10): what is left is the conversion and the
    YUV rounding, a fraction of a level on the gradient and on the colour patches."""
    frames = clip()
    path, _ = encoded(tmp_path, "h264-mp4", pix_fmt, frames=frames, crf=0)
    decoded = decode(path)["frames"]
    mean, largest = colour_error(decoded, frames)
    patches = (patch_interiors(decoded - frames).abs() * 255).mean()
    assert mean < 0.75 and largest < 4 and patches < 1.0, (mean, largest, patches)


def test_audio_shorter_than_the_video_is_kept_whole(tmp_path):
    path, _ = encoded(tmp_path, "h264-mp4", frames=clip(30), audio=tone(0.25, rate=48000, channels=1))
    got = decode(path)
    assert got["audio_channels"] == 1 and got["audio_rate"] == 48000
    assert 12000 <= got["audio_samples"] <= 12000 + 1024


def test_more_channels_get_their_default_layout(tmp_path):
    path, _ = encoded(tmp_path, "h264-mp4", frames=clip(6), audio=tone(0.5, rate=48000, channels=6))
    assert decode(path)["audio_channels"] == 6


def test_opus_resamples_to_48k(tmp_path):
    path, _ = encoded(tmp_path, "vp9-webm", frames=clip(30), audio=tone(2.0, rate=44100))
    got = decode(path)
    expected = video.audio_samples(30, FPS, 48000)
    assert got["audio_rate"] == 48000
    assert expected - 960 <= got["audio_samples"] <= expected + 960


def test_odd_sizes_are_padded_by_repeating_the_edge(tmp_path):
    frames = clip(3, 33, 65)
    path, _ = encoded(tmp_path, "h264-mp4", "yuv444p", frames=frames, crf=0)
    got = decode(path)
    assert (got["height"], got["width"]) == video.even_size(33, 65) == (34, 66)
    # lossless coding: what is left is the 8-bit YUV rounding, under 3 levels
    assert (got["frames"][:, :33, :65] - frames).abs().max() * 255 < 3
    assert (got["frames"][:, :, 65] - got["frames"][:, :, 64]).abs().max() * 255 < 3
    assert (got["frames"][:, 33] - got["frames"][:, 32]).abs().max() * 255 < 3


@pytest.mark.parametrize("codec", ["h264-mp4", "vp9-webm"])
def test_metadata_is_written_as_json(tmp_path, codec):
    spec = video.CODECS[codec]
    workflow = {"nodes": [{"id": 1, "type": "BCVSaveVideo"}], "note": "a; b = c # d\nline"}
    prompt = {"1": {"class_type": "BCVSaveVideo", "inputs": {}}}
    path = str(tmp_path / f"meta.{spec['extension']}")
    video.write_video(path, clip(2), FPS, codec, spec["crf"]["default"], spec["preset"]["default"],
                      spec["pix_fmt"]["default"], metadata={"workflow": workflow, "prompt": prompt})
    tags = {k.lower(): v for k, v in decode(path)["metadata"].items()}
    assert json.loads(tags["workflow"]) == workflow and json.loads(tags["prompt"]) == prompt

    path = str(tmp_path / f"plain.{spec['extension']}")
    video.write_video(path, clip(2), FPS, codec, spec["crf"]["default"], spec["preset"]["default"], spec["pix_fmt"]["default"])
    assert not {"workflow", "prompt"} & {k.lower() for k in decode(path)["metadata"]}


# ---- the conversion --------------------------------------------------------------------------

def bt709(rgb):
    """(luma, B - luma scaled, R - luma scaled) of float RGB [..., 3]: Y' and Cb, Cr in -0.5..0.5."""
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return luma, (rgb[..., 2] - luma) / (2 * (1 - 0.0722)), (rgb[..., 0] - luma) / (2 * (1 - 0.2126))


def ramp_frame(height=4, width=256):
    """Grey levels finer than 8 bits on the first half of the rows, a colour ramp on the rest."""
    ramp = 0.1 + 0.8 * torch.arange(width, dtype=torch.float32) / (width - 1)
    frame = ramp.view(1, width, 1).expand(height, width, 3).contiguous()
    frame[height // 2:, :, 0] = 0.8
    frame[height // 2:, :, 2] = 1.0 - ramp
    return frame


def test_eight_bit_rounds_half_up_and_luma_is_the_formula():
    """8 bits: x * 255 + 0.5 truncated, then swscale: Y is 16 + 219 * luma of the rounded RGB,
    correctly rounded."""
    convert = video._FrameConverter(2, 2, "yuv420p")
    convert(torch.tensor([0.5 / 255, 0.49 / 255, 254.5 / 255, 1.2]).view(2, 2, 1).expand(2, 2, 3))
    assert convert.ints[..., 0].tolist() == [[1, 0], [255, 255]]

    convert = video._FrameConverter(4, 256, "yuv420p")
    yuv = convert(ramp_frame())
    y = np.frombuffer(yuv.planes[0], dtype=np.uint8).reshape(4, -1)[:, :256].astype(np.float64)
    rounded = convert.ints.double().numpy() / 255
    assert np.abs(y - (16 + 219 * bt709(rounded)[0])).max() <= 0.51


def test_ten_bit_is_the_bt709_formula_on_the_float_frames():
    """10 bits: Y = 64 + 876 * luma, U / V = 512 + 896 * Cb / Cr averaged over each 2x2 block, from
    the float frames (the levels between 8-bit ones kept), rounded: within half a 10-bit level."""
    frame = ramp_frame()
    yuv = video._FrameConverter(4, 256, "yuv420p10le")(frame)
    assert (yuv.format.name, int(yuv.colorspace), int(yuv.color_range)) == ("yuv420p10le", 1, 1)
    planes = [np.frombuffer(p, dtype=np.uint16).reshape(p.height, -1)[:, :p.width].astype(np.float64) for p in yuv.planes]
    luma, cb, cr = (c.numpy() for c in bt709(frame.double()))
    block = lambda c: c.reshape(2, 2, 128, 2).mean((1, 3))  # noqa: E731
    assert np.abs(planes[0] - (64 + 876 * luma)).max() <= 0.5 + 1e-4
    assert np.abs(planes[1] - (512 + 896 * block(cb))).max() <= 0.5 + 1e-4
    assert np.abs(planes[2] - (512 + 896 * block(cr))).max() <= 0.5 + 1e-4


def test_frame_rate_is_the_exact_fraction():
    assert video.frame_rate(30000 / 1001) == Fraction(30000, 1001)
    assert video.frame_rate(24.0) == 24
    assert video.frame_rate(12.5) == Fraction(25, 2)
    assert video.frame_rate(23.976) == Fraction(2997, 125)


def test_audio_samples():
    assert video.audio_samples(30, 30.0, 48000) == 48000
    assert video.audio_samples(9, FPS, 44100) == round(9 * 44100 * 1001 / 30000)


# ---- errors ----------------------------------------------------------------------------------

def test_settings_a_codec_does_not_take():
    video.check_settings("h264-mp4", 51, "medium", "yuv444p")
    with pytest.raises(ValueError, match="crf 52 is outside h264-mp4's range 0-51"):
        video.check_settings("h264-mp4", 52, "medium", "yuv420p")
    with pytest.raises(ValueError, match="crf 0 is outside av1-webm's range 1-63"):
        video.check_settings("av1-webm", 0, "8", "yuv420p")
    with pytest.raises(ValueError, match="preset '8' is not one of h264-mp4's"):
        video.check_settings("h264-mp4", 19, "8", "yuv420p")
    with pytest.raises(ValueError, match="pix_fmt 'yuv444p' is not one of h265-mp4's"):
        video.check_settings("h265-mp4", 22, "medium", "yuv444p")
    with pytest.raises(ValueError, match="codec 'gif' is not one of"):
        video.check_settings("gif", 19, "medium", "yuv420p")


def test_a_missing_encoder_is_named(tmp_path, monkeypatch):
    monkeypatch.setattr(av, "codecs_available", set(av.codecs_available) - {"libx264"})
    with pytest.raises(RuntimeError) as error:
        video.check_encoders("h264-mp4")
    message = str(error.value)
    assert "h264-mp4 needs the encoder libx264" in message
    present = message.split("Encoders it has: ")[1].split(".")[0].split(", ")
    assert "libx265" in present and "libx264" not in present
    path = tmp_path / "never.mp4"
    with pytest.raises(RuntimeError, match="libx264"):
        video.write_video(str(path), clip(2), FPS, "h264-mp4", 19, "medium", "yuv420p")
    assert not path.exists()


def test_empty_and_mismatched_frames(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        video.write_video(str(tmp_path / "a.mp4"), clip(0), FPS, "h264-mp4", 19, "medium", "yuv420p")
    frames = [clip(1, 64, 96)[0], clip(1, 32, 96)[0]]
    with pytest.raises(ValueError, match="every frame must have the first frame's size"):
        video.write_video(str(tmp_path / "b.mp4"), frames, FPS, "h264-mp4", 19, "medium", "yuv420p")
    assert not (tmp_path / "b.mp4").exists()


def test_has_audio():
    assert not video.has_audio(None)
    assert not video.has_audio({"waveform": torch.zeros(1, 2, 0), "sample_rate": 44100})
    assert video.has_audio(tone(0.1))
