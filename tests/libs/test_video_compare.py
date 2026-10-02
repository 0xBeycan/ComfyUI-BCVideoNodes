"""The side-by-side frames of libs/video_compare.py: the common geometry of two clips (the shorter
count, the larger frame cut to even sides), a larger clip only cut, a smaller one scaled to fit and
letterboxed on black, A left of B in one reused buffer. Synthetic clips, no ComfyUI."""
import pytest

torch = pytest.importorskip("torch")

from video_output_fakes import video  # noqa: E402


def solid(frames, height, width, value):
    return torch.full((frames, height, width, 3), value, dtype=torch.float32)


def test_geometry_is_the_shorter_clip_and_the_larger_even_frame():
    sides = [("A", solid(6, 33, 65, 0.0)), ("B", solid(4, 32, 64, 1.0))]
    assert video.side_by_side_geometry(sides) == (4, 32, 64)
    assert video.side_by_side_geometry([("B", solid(5, 17, 9, 0.0))]) == (5, 16, 8)


def test_a_larger_clip_is_only_cut():
    frame = torch.rand(33, 65, 4)
    out = torch.empty(32, 64, 3)
    video.fit_into(out, frame)
    assert torch.equal(out, frame[:32, :64, :3])


def test_a_smaller_clip_is_scaled_and_letterboxed():
    out = torch.full((16, 32, 3), 0.5)
    video.fit_into(out, torch.ones(8, 8, 3))  # scaled x2 to 16x16, centred between black bars
    assert torch.equal(out[:, :8], torch.zeros(16, 8, 3)) and torch.equal(out[:, 24:], torch.zeros(16, 8, 3))
    assert torch.allclose(out[:, 8:24], torch.ones(16, 16, 3), atol=1e-6)
    out = torch.empty(16, 32, 3)
    video.fit_into(out, torch.zeros(8, 16, 3) + 0.25)  # the same aspect: fills the frame
    assert torch.allclose(out, torch.full((16, 32, 3), 0.25), atol=1e-6)


@pytest.mark.parametrize("height, width", [(33, 65), (8, 8)])  # only cut, then scaled up
def test_a_half_clip_is_fitted_from_its_float32_levels(height, width):
    from video_input_fakes import levels

    frame = levels(height, width, 3)
    expected, out = torch.empty(32, 64, 3), torch.empty(32, 64, 3)
    video.fit_into(expected, frame)
    video.fit_into(out, frame.half())
    assert torch.equal(out, expected)


def test_frames_are_a_then_b_in_one_buffer():
    a, b = solid(3, 8, 16, 0.0), solid(3, 16, 32, 1.0)
    sides = [("A", a), ("B", b)]
    frames, h, w = video.side_by_side_geometry(sides)
    seen = []
    for frame in video.side_by_side_frames(sides, frames, h, w):
        assert frame.shape == (16, 64, 3)
        assert torch.equal(frame[:, :32], torch.zeros(16, 32, 3))  # A 8x16 scaled x2 fills its half
        assert torch.allclose(frame[:, 32:], torch.ones(16, 32, 3))
        seen.append(frame.data_ptr())
    assert len(seen) == 3 and len(set(seen)) == 1
