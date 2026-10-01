"""pipelines/video_input.py: Load Video end to end on small lossless clips (the frames it keeps,
4n+1, the orientation and size, the audio range, video_info, every error), the load again when
the decoder disagrees with the container's count, Load Reference Image and Conform Video."""
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("av")
pytest.importorskip("comfy.utils")

from video_input_fakes import grey_clip, grey_index, ramp_audio, video  # noqa: E402


def load(path, model="Wan", resolution="480p", orientation="auto", force_fps="", start_frame=1, frame_count=""):
    return video.load_video(str(path), model, resolution, orientation, force_fps, start_frame, frame_count)


def frame_indices(images):
    """The grey clip's frame index of every loaded frame (a flat frame stays flat through lanczos)."""
    return [grey_index(float(frame[0, 0, 0])) for frame in images]


# --- frames ---------------------------------------------------------------------------------------

def test_every_frame_cut_to_4n_plus_1(tmp_path):
    images, _, info = load(grey_clip(tmp_path / "clip.mkv", 30))
    assert frame_indices(images) == list(range(29))  # 30 -> 29 = 4 * 7 + 1
    assert images.dtype == torch.float32 and images.is_contiguous()


def test_start_frame_and_frame_count(tmp_path):
    images, _, _ = load(grey_clip(tmp_path / "clip.mkv", 30), start_frame=5, frame_count="11")
    assert frame_indices(images) == list(range(4, 13))  # 11 -> 9 frames from frame 5 (index 4)


@pytest.mark.parametrize("rate, force, expected", [
    (30, "24", [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20]),
    (60, "30", [0, 2, 4, 6, 8, 10, 12, 14, 16]),
    (30000 / 1001, "24", [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20]),
])
def test_force_fps_keeps_the_frames_of_its_time_grid(tmp_path, rate, force, expected):
    from fractions import Fraction

    path = grey_clip(tmp_path / "clip.mkv", 22 if rate != 60 else 18, Fraction(rate).limit_denominator(1001))
    images, _, info = load(path, force_fps=force)
    assert frame_indices(images) == expected  # 17 and 9 kept: already 4n + 1
    assert info["loaded_fps"] == float(force) and info["source_fps"] == pytest.approx(rate)


def test_errors_say_what_to_change(tmp_path):
    path = grey_clip(tmp_path / "clip.mkv", 30)
    cases = [
        ({"force_fps": "fast"}, "force_fps must be empty (the source frame rate) or a number above 0, such as 24 or "
                                "29.97; got 'fast'."),
        ({"force_fps": "0"}, "force_fps must be empty"),
        ({"force_fps": "-24"}, "force_fps must be empty"),
        ({"frame_count": "0"}, "frame_count must be empty (every frame from start_frame on) or a whole number of at "
                               "least 1; got '0'."),
        ({"frame_count": "2.5"}, "frame_count must be empty"),
        ({"start_frame": 31}, "start_frame 31 is past the end of the video: it has 30 frames. Set start_frame to 30 or "
                              "less."),
        ({"start_frame": 25, "frame_count": "10"}, "start_frame 25 with frame_count 10 ends at frame 34, past the "
                                                   "video's 30 frames. Set frame_count to 6 or less, or start earlier."),
        # counted after force_fps: 30 frames at 30 fps are 24 frames at 24 fps
        ({"force_fps": "24", "start_frame": 25}, "start_frame 25 is past the end of the video: it has 24 frames at "
                                                 "force_fps 24. Set start_frame to 24 or less."),
        ({"start_frame": 0}, "start_frame must be 1 or more (the first frame is 1); got 0."),
        ({"resolution": "704p"}, "resolution '704p' does not belong to model Wan; pick one of 480p, 720p."),
    ]
    for widgets, message in cases:
        with pytest.raises(ValueError) as error:
            load(path, **widgets)
        assert str(error.value).startswith(message), widgets


@pytest.mark.parametrize("rate, force, message", [
    (30, "31", "force_fps 31 is above the video's frame rate 30: the loader only lowers the frame rate (it keeps "
               "or drops real frames, never repeats them). Leave force_fps empty or set it to 30 or less."),
    (30, "300", "force_fps 300 is above the video's frame rate 30"),
    (30000 / 1001, "30", "force_fps 30 is above the video's frame rate 29.97"),
])
def test_force_fps_above_the_source_rate_is_an_error(tmp_path, monkeypatch, rate, force, message):
    from fractions import Fraction

    path = grey_clip(tmp_path / "clip.mkv", 12, Fraction(rate).limit_denominator(1001))

    def allocate(*args, **kwargs):
        raise AssertionError("a batch was allocated before force_fps was checked")

    monkeypatch.setattr(torch, "empty", allocate)
    with pytest.raises(ValueError) as error:
        load(path, force_fps=force)
    assert str(error.value).startswith(message)


@pytest.mark.parametrize("rate, force", [(30, "30"), (30000 / 1001, "29.97"), (30000 / 1001, "29.97003")])
def test_force_fps_at_the_source_rate_keeps_every_frame(tmp_path, rate, force):
    # 29.97 typed for a 30000/1001 video is its rate: its grid, a hair slower, would drop frame 1
    from fractions import Fraction

    images, _, info = load(grey_clip(tmp_path / "clip.mkv", 9, Fraction(rate).limit_denominator(1001)), force_fps=force)
    assert frame_indices(images) == list(range(9)) and info["loaded_fps"] == rate


def test_the_spec_range_example():
    # 200 frames: start 100 + count 50 is valid, start 180 + count 50 is not
    kept = list(range(200))
    assert video.loaded_frames(kept, 100, 50, 4) == list(range(99, 148))  # 50 -> 49
    with pytest.raises(ValueError, match="ends at frame 229, past the video's 200 frames"):
        video.loaded_frames(kept, 180, 50, 4)


@pytest.mark.parametrize("count, loaded", [(1, 1), (4, 1), (5, 5), (8, 5), (9, 9), (80, 77), (81, 81)])
def test_4n_plus_1_counts(count, loaded):
    assert len(video.loaded_frames(list(range(100)), 1, count, 4)) == loaded


# --- size and orientation -------------------------------------------------------------------------

@pytest.mark.parametrize("width, height, orientation, size", [
    (32, 64, "auto", (480, 832)),       # portrait source
    (64, 32, "auto", (832, 480)),       # landscape source
    (48, 48, "auto", (832, 480)),       # square: landscape
    (64, 32, "portrait", (480, 832)),   # forced: a centre crop of the landscape source
    (32, 64, "landscape", (832, 480)),
])
def test_orientation_and_size(tmp_path, width, height, orientation, size):
    images, _, info = load(grey_clip(tmp_path / "clip.mkv", 1, width=width, height=height), orientation=orientation)
    assert tuple(images.shape) == (1, size[1], size[0], 3)
    assert (info["loaded_width"], info["loaded_height"]) == size
    assert info["orientation"] == ("portrait" if size[1] > size[0] else "landscape")


def test_frames_are_the_fit_of_the_decoded_frame(tmp_path):
    # the loader's frame is resize.fit of the decoded RGB frame: the same crop and lanczos
    import av

    path = grey_clip(tmp_path / "clip.mkv", 1, width=64, height=48)
    with av.open(path) as container:
        pixels = video.rgb(next(container.decode(video=0)))
    expected = torch.empty((480, 832, 3))
    video.fit(pixels, expected)
    images, _, _ = load(path)
    assert torch.equal(images[0], expected)


# --- audio and video_info -------------------------------------------------------------------------

def test_audio_is_the_loaded_range(tmp_path):
    samples, rate = ramp_audio(1.0)
    path = grey_clip(tmp_path / "clip.mkv", 30, audio=(samples, rate))
    _, audio, _ = load(path, start_frame=5, frame_count="11")  # 9 frames from 4 / 30 s
    assert audio["sample_rate"] == 48000
    assert np.array_equal(audio["waveform"][0].numpy(), (samples[:, 6400:6400 + 14400] / 32768).astype(np.float32))


def test_audio_with_force_fps_spans_the_kept_frames(tmp_path):
    samples, rate = ramp_audio(1.0)
    path = grey_clip(tmp_path / "clip.mkv", 30, audio=(samples, rate))
    _, audio, info = load(path, force_fps="24", start_frame=3, frame_count="9")  # 9 frames at 24 fps from 2 / 24 s
    assert info["loaded_duration"] == pytest.approx(9 / 24)
    assert np.array_equal(audio["waveform"][0].numpy(), (samples[:, 4000:4000 + 18000] / 32768).astype(np.float32))


def test_video_info_keys_order_and_values(tmp_path):
    _, audio, info = load(grey_clip(tmp_path / "clip.mkv", 30, width=64, height=32), resolution="720p")
    assert audio is None
    assert list(info) == ["model", "resolution", "orientation", "source_fps", "source_frame_count",
                          "source_duration", "source_width", "source_height", "loaded_fps", "loaded_frame_count",
                          "loaded_duration", "loaded_width", "loaded_height"]
    assert list(info) == list(video.VideoInfo.__annotations__)
    assert info == {"model": "Wan", "resolution": "720p", "orientation": "landscape",
                    "source_fps": 30.0, "source_frame_count": 30, "source_duration": 1.0,
                    "source_width": 64, "source_height": 32,
                    "loaded_fps": 30.0, "loaded_frame_count": 29, "loaded_duration": pytest.approx(29 / 30),
                    "loaded_width": 1280, "loaded_height": 720}


# --- the decoder's count wins ---------------------------------------------------------------------

@pytest.mark.parametrize("promised", [27, 33])
def test_a_wrong_container_count_loads_again_with_the_decoders(tmp_path, monkeypatch, promised):
    path = grey_clip(tmp_path / "clip.mkv", 30)
    expected, _, expected_info = load(path)
    probe = video.probe

    def lying_probe(p):
        return {**probe(p), "frames": promised}

    monkeypatch.setattr(video, "probe", lying_probe)
    images, _, info = load(path)
    assert torch.equal(images, expected) and info == expected_info


def test_a_wrong_count_still_raises_for_a_range_past_the_real_end(tmp_path, monkeypatch):
    path = grey_clip(tmp_path / "clip.mkv", 30)
    probe = video.probe
    monkeypatch.setattr(video, "probe", lambda p: {**probe(p), "frames": 40})
    with pytest.raises(ValueError, match="start_frame 25 with frame_count 10 ends at frame 34, past the video's 30"):
        load(path, start_frame=25, frame_count="10")


# --- Load Reference Image -------------------------------------------------------------------------

def rgba_image(path, width, height, seed=0):
    from PIL import Image

    pixels = np.random.default_rng(seed).integers(0, 256, (height, width, 4), dtype=np.uint8)
    Image.fromarray(pixels, "RGBA").save(path)
    return pixels


def test_reference_image_is_fitted_and_its_alpha_is_the_mask(tmp_path):
    pixels = rgba_image(tmp_path / "ref.png", 90, 120)
    image, mask = video.load_reference_image(str(tmp_path / "ref.png"), 48, 64)
    expected, alpha = torch.empty((64, 48, 3)), torch.empty((64, 48))
    video.fit(pixels[..., :3], expected)
    video.fit(pixels[..., 3], alpha)
    assert image.shape == (1, 64, 48, 3) and torch.equal(image[0], expected)
    assert mask.shape == (1, 64, 48) and torch.equal(mask[0], 1.0 - alpha)


def test_reference_image_without_alpha_gets_cores_empty_mask(tmp_path):
    from PIL import Image

    Image.new("RGB", (20, 10), (255, 0, 0)).save(tmp_path / "ref.jpg")
    image, mask = video.load_reference_image(str(tmp_path / "ref.jpg"), 16, 8)
    assert image.shape == (1, 8, 16, 3) and torch.equal(mask, torch.zeros((1, 64, 64)))


def test_reference_image_follows_its_exif_orientation(tmp_path):
    from PIL import Image

    stored = Image.new("RGB", (40, 20), (0, 0, 0))
    stored.paste((255, 255, 255), (0, 0, 10, 10))  # white in the top-left corner as stored
    exif = Image.Exif()
    exif[0x0112] = 6  # shown turned 90 degrees clockwise: 20 wide, 40 tall, white top-right
    stored.save(tmp_path / "ref.png", exif=exif)
    image, _ = video.load_reference_image(str(tmp_path / "ref.png"), 20, 40)
    assert image[0, 3, 16].min() > 0.9 and image[0, 3, 3].max() < 0.1


# --- Conform Video --------------------------------------------------------------------------------

def test_conform_returns_the_input_itself_at_a_standard_size():
    images = torch.rand((2, 1280, 720, 3))
    assert video.conform_video(images, "crop", "lanczos") is images


def test_conform_crop_fits_frame_by_frame():
    images = torch.rand((2, 1280, 704, 3), generator=torch.Generator().manual_seed(0))
    out = video.conform_video(images, "crop", "lanczos")
    assert out.shape == (2, 1280, 720, 3) and out.dtype == images.dtype
    for index in range(2):
        expected = torch.empty((1280, 720, 3))
        video.fit(images[index], expected, "crop", "lanczos")
        assert torch.equal(out[index], expected)


def test_conform_pad_puts_black_bars_around_the_whole_frame():
    images = torch.full((1, 896, 512, 3), 0.5)  # SCAIL 512p, conformed to 480p (480x854)
    out = video.conform_video(images, "pad", "bilinear")
    # contain 512x896 in 480x854: scale min(0.9375, 0.9531) = 0.9375 -> 480x840, 7 rows of bar above
    assert out.shape == (1, 854, 480, 3)
    assert torch.equal(out[0, :7], torch.zeros(7, 480, 3)) and torch.equal(out[0, 847:], torch.zeros(7, 480, 3))
    assert torch.allclose(out[0, 7:847], torch.full((840, 480, 3), 0.5))
