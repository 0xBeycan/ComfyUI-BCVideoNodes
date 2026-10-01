"""libs/resize.py and libs/video_sizes.py: the crop and contain boxes (hand-computed), the fit
against comfy.utils.lanczos and comfy.utils.common_upscale on the same crop, no resampling at the
target size, the model table, the orientation rule and Conform Video's target table."""
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from video_input_fakes import video  # noqa: E402


def comfy_utils():
    return pytest.importorskip("comfy.utils")


# --- boxes ----------------------------------------------------------------------------------------

def test_crop_box_cuts_the_centre_to_the_target_aspect():
    # 1080x1920 (0.5625) to 480x832 (0.5769): full width, 1080 / 0.5769 = 1872 rows, 24 off each end
    assert video.crop_box(1080, 1920, 480, 832) == (0, 24, 1080, 1872)
    # the same aspect: nothing cut
    assert video.crop_box(1080, 1920, 720, 1280) == (0, 0, 1080, 1920)
    # SCAIL 704x1280 to 720x1280: 704 / 0.5625 = 1251.6 -> 1252 rows, 14 off the top
    assert video.crop_box(704, 1280, 720, 1280) == (0, 14, 704, 1252)
    # landscape 1920x1080 forced portrait 720x1280: 1080 * 0.5625 = 607.5 -> 608 columns (int: 607)
    assert video.crop_box(1920, 1080, 720, 1280) == (656, 0, 608, 1080)


def test_contain_box_centres_the_whole_frame():
    # 720x1280 into 1280x720: scale 0.5625 -> 405x720, (1280 - 405) // 2 = 437 columns of bar on the left
    assert video.contain_box(720, 1280, 1280, 720) == (437, 0, 405, 720)
    # 704x1280 into 720x1280: scale 1.0 (height-bound) -> 704x1280, 8 columns each side
    assert video.contain_box(704, 1280, 720, 1280) == (8, 0, 704, 1280)


# --- fit ------------------------------------------------------------------------------------------

def random_pixels(height, width, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)


def comfy_lanczos(pixels, width, height):
    """comfy.utils.lanczos of `pixels` as an IMAGE frame would reach it: uint8 / 255, [1, C, H, W]."""
    frame = torch.from_numpy(pixels.astype(np.float32) / 255.0).movedim(-1, 0)[None]
    return comfy_utils().lanczos(frame, width, height)[0].movedim(0, -1)


def test_crop_then_lanczos_equals_comfy_lanczos_on_the_same_crop():
    pixels = random_pixels(96, 54)
    out = torch.empty((40, 24, 3))
    video.fit(pixels, out)
    x, y, w, h = video.crop_box(54, 96, 24, 40)
    assert torch.equal(out, comfy_lanczos(pixels[y:y + h, x:x + w], 24, 40))


def test_a_float_frame_takes_the_same_path():
    pixels = random_pixels(96, 54, seed=1)
    frame = torch.from_numpy(pixels.astype(np.float32) / 255.0)
    out, reference = torch.empty((40, 24, 3)), torch.empty((40, 24, 3))
    video.fit(frame, out)
    video.fit(pixels, reference)
    assert torch.equal(out, reference)


def test_a_float_frame_is_rounded_to_its_nearest_level_not_truncated():
    # 100.4 / 255 is level 100, 100.6 / 255 level 101; a flat frame stays flat through lanczos
    for level, expected in ((100.4, 100), (100.6, 101)):
        out = torch.empty((8, 8, 3))
        video.fit(torch.full((16, 16, 3), level / 255), out)
        assert torch.equal(out, torch.full((8, 8, 3), expected / 255.0))


def test_no_resampling_at_the_target_size():
    pixels = random_pixels(40, 24, seed=2)
    out = torch.empty((40, 24, 3))
    video.fit(pixels, out)
    assert torch.equal(out, torch.from_numpy(pixels.astype(np.float32) / 255.0))


def test_a_mask_plane_is_fitted_like_a_channel():
    plane = random_pixels(96, 54, seed=3)[..., 0]
    out, image = torch.empty((40, 24)), torch.empty((40, 24, 3))
    video.fit(plane, out)
    video.fit(np.repeat(plane[..., None], 3, axis=2), image)
    assert torch.equal(out, image[..., 0])


def test_pad_centres_the_fitted_frame_between_black_bars():
    pixels = random_pixels(64, 36, seed=4)
    out = torch.full((36, 64, 3), 0.5)
    video.fit(pixels, out, video.PAD)
    x, y, w, h = video.contain_box(36, 64, 64, 36)
    assert (x, y, w, h) == (22, 0, 20, 36)  # scale 0.5625: 20x36, (64 - 20) // 2 = 22
    assert torch.equal(out[:, :x], torch.zeros(36, x, 3)) and torch.equal(out[:, x + w:], torch.zeros(36, 64 - x - w, 3))
    assert torch.equal(out[:, x:x + w], comfy_lanczos(pixels, w, h))


@pytest.mark.parametrize("method", ["bicubic", "bilinear", "area", "nearest-exact", "bislerp"])
def test_the_other_methods_are_comfy_common_upscale(method):
    frame = torch.rand((48, 30, 3), generator=torch.Generator().manual_seed(5))
    out = torch.empty((20, 12, 3))
    video.fit(frame, out, video.CROP, method)
    x, y, w, h = video.crop_box(30, 48, 12, 20)
    crop = frame[y:y + h, x:x + w].movedim(-1, 0)[None]
    assert torch.equal(out, comfy_utils().common_upscale(crop, 12, 20, method, "disabled")[0].movedim(0, -1))


def test_the_methods_are_common_upscales_lanczos_first():
    assert video.METHODS[0] == video.LANCZOS == "lanczos"
    assert set(video.METHODS) == {"lanczos", "bicubic", "bilinear", "area", "nearest-exact", "bislerp"}


def test_unknown_fit_and_method_raise():
    with pytest.raises(ValueError, match="fit 'stretch' is not one of crop, pad"):
        video.fit(random_pixels(8, 8), torch.empty((4, 4, 3)), "stretch")
    with pytest.raises(ValueError, match="method 'cubic' is not one of"):
        video.fit(random_pixels(8, 8), torch.empty((4, 4, 3)), video.CROP, "cubic")


# --- sizes ----------------------------------------------------------------------------------------

def test_the_model_table():
    assert video.MODELS == {
        "Wan": {"frames": 4, "sizes": {"480p": [480, 832], "720p": [720, 1280]}},
        "SCAIL": {"frames": 4, "sizes": {"512p": [512, 896], "704p": [704, 1280]}},
    }
    assert video.RESOLUTIONS == ["480p", "720p", "512p", "704p"]
    assert video.model_size("SCAIL", "704p") == [704, 1280]


def test_a_resolution_of_another_model_is_rejected():
    with pytest.raises(ValueError, match="resolution '512p' does not belong to model Wan; pick one of 480p, 720p."):
        video.model_size("Wan", "512p")
    with pytest.raises(ValueError, match="model 'Hunyuan' is not one of Wan, SCAIL"):
        video.model_size("Hunyuan", "720p")


def test_orientation_portrait_only_when_taller_than_wide():
    assert video.orientation_of(1080, 1920) == "portrait"
    assert video.orientation_of(1920, 1080) == "landscape"
    assert video.orientation_of(720, 720) == "landscape"
    assert video.oriented([480, 832], "landscape") == [832, 480]
    assert video.oriented([480, 832], "portrait") == [480, 832]


@pytest.mark.parametrize("size, target", [
    ((600, 1066), [720, 1280]),    # 600: |log 720/600| 0.18 < |log 480/600| 0.22
    ((512, 896), [480, 854]),      # SCAIL 512p -> 480p (x0.94)
    ((704, 1280), [720, 1280]),    # SCAIL 704p -> 720p (x1.02)
    ((480, 832), [480, 854]),      # Wan 480p -> 480p
    ((1440, 2560), [1080, 1920]),  # above 1080 -> 1080p
    ((587, 1044), [480, 854]),     # the 480p / 720p midpoint is 587.9
    ((588, 1045), [720, 1280]),
    ((881, 1566), [720, 1280]),    # the 720p / 1080p midpoint is 881.8
    ((882, 1568), [1080, 1920]),
    ((1280, 704), [1280, 720]),    # landscape: the same ladder, swapped
    ((896, 512), [854, 480]),
    ((2560, 1440), [1920, 1080]),
    ((720, 720), [1280, 720]),     # square: landscape
])
def test_conform_targets(size, target):
    assert video.conform_size(*size) == target
