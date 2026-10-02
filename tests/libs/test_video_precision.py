"""Load Video's precision and the half-precision reads of libs/video.py: float16 keeps every 8-bit
level k / 255 and requantized gives back the float32 values a float32 load holds, exactly, for all
256 levels (bfloat16 too); a float32 tensor passes through untouched; HalfFrames reads a half clip
as float32 numpy frames one at a time, and the clip stays as it is. Synthetic tensors, no ComfyUI.
"""
import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from video_input_fakes import levels, video  # noqa: E402


def stored(dtype):
    """The 256 levels as Load Video stores a decoded uint8 pixel in a batch of `dtype`: copied, then / 255."""
    return torch.empty(256, dtype=dtype).copy_(torch.arange(256, dtype=torch.uint8)).div_(255)


def test_the_widget_values():
    assert list(video.PRECISIONS) == ["fp32", "fp16"] and (video.FP32, video.FP16) == ("fp32", "fp16")
    assert video.precision_dtype("fp32") == torch.float32 and video.precision_dtype("fp16") == torch.float16
    with pytest.raises(ValueError) as error:
        video.precision_dtype("fp8")
    assert str(error.value) == "precision must be one of fp32, fp16; got 'fp8'."


def test_float16_stores_every_level_as_its_nearest_value_and_gives_back_float32_exactly():
    exact = stored(torch.float32)
    half = stored(torch.float16)
    assert torch.equal(half, exact.half())  # the float16 nearest each level
    assert (half.float() != exact).sum() == 254  # widened as it is, only 0 and 1 are exact
    back = video.requantized(half)
    assert back.dtype == torch.float32 and torch.equal(back, exact)
    assert torch.equal(video.requantized(exact.bfloat16()), exact)


def test_float32_and_every_other_dtype_pass_through_untouched():
    for frames in (levels(2, 4, 3), levels(2, 4, 3).double(), torch.zeros(2, 4, 3, dtype=torch.uint8)):
        assert video.requantized(frames) is frames
    assert video.is_half(torch.zeros(1, dtype=torch.float16)) and video.is_half(torch.zeros(1, dtype=torch.bfloat16))
    assert not video.is_half(torch.zeros(1))


def test_a_window_is_read_as_a_new_float32_tensor_and_the_clip_stays_half():
    clip = levels(6, 4, 5, 3).half()
    before = clip.clone()
    window = video.requantized(clip[2:4])
    assert window.dtype == torch.float32 and window.shape == (2, 4, 5, 3)
    assert window.data_ptr() != clip.data_ptr() and torch.equal(clip, before)


def test_half_frames_are_read_as_float32_one_at_a_time():
    exact = levels(5, 6, 4, 3)
    frames = video.as_numpy(exact.half())
    assert isinstance(frames, video.HalfFrames)
    assert len(frames) == 5 and frames.shape == (5, 6, 4, 3) and frames.dtype == np.float32
    assert all(np.array_equal(frame, exact[i].numpy()) and frame.dtype == np.float32 for i, frame in enumerate(frames))
    assert np.array_equal(frames[1:3], exact[1:3].numpy()) and frames[1:3].dtype == np.float32


def test_a_float32_batch_is_its_own_memory_and_anything_else_an_array():
    exact = levels(2, 3, 4, 3)
    array = video.as_numpy(exact)
    assert isinstance(array, np.ndarray) and np.shares_memory(array, exact.numpy())
    assert np.array_equal(video.as_numpy([[0.5, 1.0]]), np.array([[0.5, 1.0]]))
