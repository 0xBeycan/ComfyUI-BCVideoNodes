"""libs/video_decode.py: the frame selection on the force_fps time grid (hand-derived indices), the
probe and the one-frame-at-a-time decode, the colour conversion against the BT.709 / BT.601
formulas written out here, the display rotation, and the audio of a time range. The clips are
lossless FFV1 files written by tests/video_input_fakes.py."""
from fractions import Fraction

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")

from video_input_fakes import grey_clip, grey_index, ramp_audio, video, write_clip, yuv_frame  # noqa: E402

# --- frame selection ----------------------------------------------------------------------------
# Output frame m is the first source frame at or after m / force_fps.


def test_30_to_24_drops_one_frame_in_five():
    # source frame i is at i / 30: output 1 (1/24 s) falls after frame 1 (1/30 s), so frame 2
    assert video.select_frames(30, 20, 24) == [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19]


def test_60_to_30_keeps_every_other_frame():
    assert video.select_frames(60, 12, 30) == [0, 2, 4, 6, 8, 10]


def test_60_to_24_keeps_two_frames_in_five():
    # output m at m / 24 = 2.5 m source frames: 0, 2.5 -> 3, 5, 7.5 -> 8, 10, ...
    assert video.select_frames(60, 20, 24) == [0, 3, 5, 8, 10, 13, 15, 18]


def test_29_97_to_24_starts_as_30_to_24_and_parts_at_output_201():
    fps = 30000 / 1001
    assert video.select_frames(fps, 20, 24) == [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19]
    # output 201 at 201 / 24 = 8.375 s: frame 251 is at 251 * 1001 / 30000 = 8.3750 s (after it by
    # 0.03 ms), where 30 fps has frame 251 at 8.3667 s (before it) and takes frame 252
    assert video.select_frames(fps, 300, 24)[201] == 251
    assert video.select_frames(30, 300, 24)[201] == 252


def test_without_force_fps_every_frame_is_kept():
    assert video.select_frames(30, 7) == [0, 1, 2, 3, 4, 5, 6]
    assert video.select_frames(30000 / 1001, 5, None) == [0, 1, 2, 3, 4]


def test_a_rate_above_the_source_repeats_frames():
    # 24 -> 30: each output is the first frame at or after m / 30; at the exact boundary (frame 4 at
    # 4/24 = output 5 at 5/30) the float sum falls just past it, so frame 5 repeats, not frame 4
    assert video.select_frames(24, 12, 30) == [0, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9, 10, 11]


def test_twice_the_rate_shows_frame_0_once_then_every_frame_twice():
    # output m at m / 32 s = (m / 2) / 16 s: the first frame at or after it is ceil(m / 2); the grid
    # ends after the last frame, which shows twice. 30 -> 60 the same.
    assert video.select_frames(16, 6, 32) == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert video.select_frames(30, 6, 60) == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_16_to_24_repeats_every_other_frame():
    # output m at m / 24 s: frame ceil(2 m / 3)
    assert video.select_frames(16, 8, 24) == [0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7]


def test_29_97_to_30_repeats_frame_1000():
    # output m: frame ceil(1000 m / 1001), which is m up to output 1000; output 1001 is at
    # 1001 / 30 s, frame 1000's time exactly, so frame 1000 shows twice
    kept = video.select_frames(30000 / 1001, 1200, 30)
    assert kept[:1001] == list(range(1001)) and kept[1001] == 1000 and kept[1002:] == list(range(1001, 1200))


def test_a_rate_a_hair_below_the_source_drops_frame_1():
    # 29.97 on a 30000/1001 video: output 1 at 1 / 29.97 s falls 0.03 us after frame 1 (1001 / 30000 s),
    # so it is frame 2; the next drop is ten thousand frames on
    assert video.select_frames(30000 / 1001, 10, 29.97) == [0, 2, 3, 4, 5, 6, 7, 8, 9]


def test_a_one_frame_video():
    assert video.select_frames(30, 1, 24) == [0]


# --- probe and decode -----------------------------------------------------------------------------

@pytest.mark.parametrize("rate", [30, Fraction(30000, 1001), 60])
def test_probe_reads_the_rate_the_count_and_the_size(tmp_path, rate):
    path = grey_clip(tmp_path / "clip.mkv", 12, rate, width=64, height=32)
    source = video.probe(path)
    assert source == {"fps": float(rate), "frames": 12, "width": 64, "height": 32, "start": 0.0, "audio": False}


def test_probe_says_whether_the_file_has_sound(tmp_path):
    path = grey_clip(tmp_path / "clip.mkv", 3, audio=ramp_audio(0.5))
    assert video.probe(path)["audio"] is True


def decoded_levels(path, indices, frames, to_end=False):
    """[(frame index read from the pixels, positions)] of kept_frames."""
    return [(grey_index(pixels[0, 0, 0] / 255), list(positions))
            for pixels, positions in video.kept_frames(path, indices, frames, to_end)]


def test_kept_frames_gives_each_kept_frame_once_with_its_positions(tmp_path):
    path = grey_clip(tmp_path / "clip.mkv", 12)
    assert decoded_levels(path, [0, 2, 3, 3, 3, 7], 12) == [(0, [0]), (2, [1]), (3, [2, 3, 4]), (7, [5])]


def test_kept_frames_raises_when_the_decoder_disagrees_with_the_count(tmp_path):
    path = grey_clip(tmp_path / "clip.mkv", 12)
    with pytest.raises(video.FrameCountChanged) as fewer:
        decoded_levels(path, [0, 13], 14)  # frame 13 never comes
    assert fewer.value.frames == 12
    with pytest.raises(video.FrameCountChanged) as more:
        decoded_levels(path, [0, 5], 9, to_end=True)  # the end is checked
    assert more.value.frames == 12
    assert decoded_levels(path, [0, 5], 9) == [(0, [0]), (5, [1])]  # stops at the last kept frame


# --- colour ---------------------------------------------------------------------------------------
# Four 16x16 blocks of flat (Y, Cb, Cr); a block's centre is compared with the conversion formula.
BLOCKS = [(60, 100, 170), (200, 150, 90), (120, 60, 200), (150, 128, 128)]


def colour_clip(path, **tags):
    y = np.concatenate([np.full((16, 16), b[0], np.uint8) for b in BLOCKS], axis=1)
    u = np.concatenate([np.full((8, 8), b[1], np.uint8) for b in BLOCKS], axis=1)
    v = np.concatenate([np.full((8, 8), b[2], np.uint8) for b in BLOCKS], axis=1)
    return write_clip(path, [yuv_frame(y, u, v)], **tags)


def formula(y, cb, cr, kr, kb, full_range):
    """RGB in 0-255 of one (Y, Cb, Cr) with the matrix (kr, kb): Y' = kr R + (1 - kr - kb) G + kb B."""
    if full_range:
        luma, pb, pr = y / 255, (cb - 128) / 255, (cr - 128) / 255
    else:
        luma, pb, pr = (y - 16) / 219, (cb - 128) / 224, (cr - 128) / 224
    r = luma + 2 * (1 - kr) * pr
    b = luma + 2 * (1 - kb) * pb
    g = (luma - kr * r - kb * b) / (1 - kr - kb)
    return np.clip(np.array([r, g, b]) * 255, 0, 255)


def block_centres(path):
    with av.open(path) as container:
        pixels = video.rgb(next(container.decode(video=0)))
    return [pixels[8, 16 * i + 8].astype(np.float64) for i in range(len(BLOCKS))]


@pytest.mark.parametrize("tags, matrix, full_range", [
    ({"colorspace": 1, "color_range": 1}, (0.2126, 0.0722), False),  # BT.709, limited
    ({"colorspace": 1, "color_range": 2}, (0.2126, 0.0722), True),   # BT.709, full
    ({"colorspace": 6, "color_range": 1}, (0.299, 0.114), False),    # BT.601 (SMPTE 170M), limited
    ({}, (0.299, 0.114), False),                                     # untagged: FFmpeg's BT.601 limited
])
def test_rgb_follows_the_stream_matrix_and_range(tmp_path, tags, matrix, full_range):
    path = colour_clip(tmp_path / "colour.mkv", **tags)
    for block, got in zip(BLOCKS, block_centres(path)):
        expected = formula(*block, *matrix, full_range)
        assert np.abs(got - expected).max() <= 1.0, (block, got, expected)


def test_rgb_is_not_dark_on_a_width_off_the_16_grid(tmp_path):
    # swscale's default path loses ~2 levels at such widths (1080 wide); 72 = 4.5 x 16
    y = np.full((16, 72), 142, np.uint8)
    u = v = np.full((8, 36), 128, np.uint8)
    path = write_clip(tmp_path / "wide.mkv", [yuv_frame(y, u, v)], colorspace=1, color_range=1)
    with av.open(path) as container:
        pixels = video.rgb(next(container.decode(video=0)))
    assert np.abs(pixels.astype(np.float64) - (142 - 16) * 255 / 219).max() <= 0.5


def test_rgb_turns_the_picture_as_the_display_matrix_says(tmp_path):
    # a white block in the top-left corner of a 64x32 frame; shown turned 90 degrees
    # counter-clockwise, the frame is 32 wide and 64 tall and the block is in the bottom-left corner
    y = np.full((32, 64), 16, np.uint8)
    y[:16, :16] = 235
    u = v = np.full((16, 32), 128, np.uint8)
    path = write_clip(tmp_path / "turned.mkv", [yuv_frame(y, u, v)], rotation=90)
    with av.open(path) as container:
        pixels = video.rgb(next(container.decode(video=0)))
    assert pixels.shape == (64, 32, 3)
    assert pixels[-8, 8].min() == 255 and pixels[8, 8].max() == 0
    assert video.probe(path)["width"] == 32 and video.probe(path)["height"] == 64


# --- audio ----------------------------------------------------------------------------------------

def test_read_audio_cuts_the_range_to_the_sample(tmp_path):
    samples, rate = ramp_audio(1.0)
    path = grey_clip(tmp_path / "sound.mkv", 30, audio=(samples, rate))
    audio = video.read_audio(path, 4 / 30, 9 / 30)
    assert audio["sample_rate"] == 48000
    waveform = audio["waveform"]
    assert waveform.shape == (1, 2, 14400)  # 9 frames at 30 fps, 48 kHz
    expected = samples[:, 6400:6400 + 14400] / 32768  # from 4 / 30 s
    assert np.array_equal(waveform[0].numpy(), expected.astype(np.float32))


def test_read_audio_is_shorter_when_the_audio_ends_first(tmp_path):
    samples, rate = ramp_audio(0.5)
    path = grey_clip(tmp_path / "short.mkv", 30, audio=(samples, rate))
    assert video.read_audio(path, 0.25, 0.5)["waveform"].shape == (1, 2, 12000)


def test_read_audio_is_none_without_an_audio_stream(tmp_path):
    assert video.read_audio(grey_clip(tmp_path / "silent.mkv", 3), 0, 0.1) is None
