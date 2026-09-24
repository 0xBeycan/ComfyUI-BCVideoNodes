"""The tail padding of libs/video.py: hold_last and ping_pong, the two ways the chunk loop extends
a driving video past its last frame. ping_pong is checked against the index sequence of the
official padding (Wan 2.2 wan/animate.py inputs_padding, Wan-Animate-2 multiclip_utils.py
zigzag_padding; the two are the same loop). Runs without ComfyUI or any model.
"""
import pytest

torch = pytest.importorskip("torch")

from bcvideonodes.libs.video import TAIL_PADDING, hold_last, ping_pong  # noqa: E402


def official_padding(length, target_len):
    """The frame indices the official padding picks for a `length`-frame input padded to
    `target_len`: its loop, run on the indices. At length 1 the official loop indexes past the
    end; the degenerate case repeats frame 0."""
    if length == 1:
        return [0] * target_len
    idx, flip, target = 0, False, []
    while len(target) < target_len:
        target.append(idx)
        idx += -1 if flip else 1
        if idx == 0 or idx == length - 1:
            flip = not flip
    return target[:target_len]


def index_video(length, dtype=torch.float32):
    return torch.arange(length, dtype=dtype).view(-1, 1, 1, 1).expand(-1, 4, 2, 3).contiguous()


def frame_indices(video):
    return [int(v) for v in video[:, 0, 0, 0].tolist()]


def test_the_widget_values():
    assert list(TAIL_PADDING) == ["last_frame", "ping_pong"]
    assert [function for function, _ in TAIL_PADDING.values()] == [hold_last, ping_pong]


def test_ten_frames_padded_by_six_play_backwards_from_the_end():
    assert frame_indices(ping_pong(index_video(10), 16))[10:] == [8, 7, 6, 5, 4, 3]


def test_a_long_padding_turns_forward_again_without_repeating_the_edge():
    padded = frame_indices(ping_pong(index_video(4), 14))
    assert padded == [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1]


def test_one_frame_repeats_and_two_frames_alternate():
    assert frame_indices(ping_pong(index_video(1), 5)) == [0, 0, 0, 0, 0]
    assert frame_indices(ping_pong(index_video(2), 7)) == [0, 1, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("length", [1, 2, 3, 5, 10, 81])
@pytest.mark.parametrize("extra", [1, 3, 7, 68, 250])
def test_ping_pong_is_the_official_padding(length, extra):
    assert frame_indices(ping_pong(index_video(length), length + extra)) == official_padding(length, length + extra)


@pytest.mark.parametrize("pad", [hold_last, ping_pong])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_shape_and_dtype_are_kept(pad, dtype):
    video = index_video(6, dtype)
    padded = pad(video, 20)
    assert padded.shape == (20, *video.shape[1:]) and padded.dtype == dtype
    assert torch.equal(padded[:6], video)


def test_hold_last_repeats_the_last_frame():
    assert frame_indices(hold_last(index_video(4), 8)) == [0, 1, 2, 3, 3, 3, 3, 3]


@pytest.mark.parametrize("pad", [hold_last, ping_pong])
def test_a_video_already_long_enough_is_unchanged(pad):
    video = index_video(9)
    assert torch.equal(pad(video, 9), video)
