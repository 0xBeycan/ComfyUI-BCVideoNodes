"""The tail padding of libs/video.py: hold_last and ping_pong, the two ways the chunk loop extends
a driving video past its last frame, each giving a window of the extended video. ping_pong is
checked against the index sequence of the official padding (Wan 2.2 wan/animate.py
inputs_padding, Wan-Animate-2 multiclip_utils.py zigzag_padding; the two are the same loop), whose
loop tests/sampler_fakes.py writes out. Runs without ComfyUI or any model.
"""
import pytest

torch = pytest.importorskip("torch")

from bcvideonodes.libs.video import TAIL_PADDING, hold_last, ping_pong  # noqa: E402
from sampler_fakes import held_last, official_padding  # noqa: E402


def index_video(length, dtype=torch.float32):
    return torch.arange(length, dtype=dtype).view(-1, 1, 1, 1).expand(-1, 4, 2, 3).contiguous()


def frame_indices(video):
    return [int(v) for v in video[:, 0, 0, 0].tolist()]


def test_the_widget_values():
    assert list(TAIL_PADDING) == ["last_frame", "ping_pong"]
    assert [function for function, _ in TAIL_PADDING.values()] == [hold_last, ping_pong]


def test_ten_frames_padded_by_six_play_backwards_from_the_end():
    assert frame_indices(ping_pong(index_video(10), 0, 16))[10:] == [8, 7, 6, 5, 4, 3]


def test_a_long_padding_turns_forward_again_without_repeating_the_edge():
    padded = frame_indices(ping_pong(index_video(4), 0, 14))
    assert padded == [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]


def test_one_frame_repeats_and_two_frames_alternate():
    assert frame_indices(ping_pong(index_video(1), 0, 5)) == [0, 0, 0, 0, 0]
    assert frame_indices(ping_pong(index_video(2), 0, 7)) == [0, 1, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("length", [1, 2, 3, 5, 10, 81])
@pytest.mark.parametrize("extra", [1, 3, 7, 68, 250])
def test_ping_pong_is_the_official_padding(length, extra):
    assert frame_indices(ping_pong(index_video(length), 0, length + extra)) == official_padding(length, length + extra)


@pytest.mark.parametrize("pad", [hold_last, ping_pong])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_shape_and_dtype_are_kept(pad, dtype):
    video = index_video(6, dtype)
    padded = pad(video, 0, 20)
    assert padded.shape == (20, *video.shape[1:]) and padded.dtype == dtype
    assert torch.equal(padded[:6], video)


def test_hold_last_repeats_the_last_frame():
    assert frame_indices(hold_last(index_video(4), 0, 8)) == [0, 1, 2, 3, 3, 3, 3, 3]


@pytest.mark.parametrize("pad", [hold_last, ping_pong])
def test_a_video_already_long_enough_is_unchanged(pad):
    video = index_video(9)
    assert torch.equal(pad(video, 0, 9), video)


# --- a window of the extended video: the frames a chunk reads, never the video extended whole ------

OFFICIAL = {hold_last: held_last, ping_pong: official_padding}


@pytest.mark.parametrize("pad", [hold_last, ping_pong])
@pytest.mark.parametrize("length", [1, 2, 5, 81])
@pytest.mark.parametrize("start, stop", [(0, 90), (70, 151), (80, 81), (85, 166), (200, 281), (3, 3)])
def test_a_window_is_those_frames_of_the_extended_video(pad, length, start, stop):
    assert frame_indices(pad(index_video(length), start, stop)) == OFFICIAL[pad](length, stop)[start:stop]


@pytest.mark.parametrize("pad", [hold_last, ping_pong])
def test_a_window_inside_the_video_is_a_view(pad):
    video = index_video(81)
    window = pad(video, 20, 81)
    assert window._base is video and frame_indices(window) == list(range(20, 81))


@pytest.mark.parametrize("pad", [hold_last, ping_pong])
def test_a_window_past_the_end_holds_just_its_frames(pad):
    video = index_video(81)
    window = pad(video, 76, 157)
    assert window.shape == (81, *video.shape[1:]) and window.is_contiguous()
    assert window.untyped_storage().nbytes() == 81 * video[0].numel() * video.element_size()
