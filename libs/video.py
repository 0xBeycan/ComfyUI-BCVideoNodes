"""Helpers on video tensors: IMAGE batches, frames first."""


def hold_last(video, frames):
    """``video`` with its last frame repeated up to ``frames`` frames."""
    import torch

    return torch.cat((video, video[-1:].expand(frames - video.shape[0], *video.shape[1:])), dim=0)


def ping_pong(video, frames):
    """``video`` extended up to ``frames`` frames by playing it back and forth from its end: after
    its last frame L-1 come L-2, L-3, ... 0, then 1, 2, ... again (a turn does not repeat the edge
    frame), the padding of the official Wan 2.2 Animate (wan/animate.py inputs_padding) and
    Wan-Animate-2 (multiclip_utils.py zigzag_padding). A single frame is repeated."""
    import torch

    length = video.shape[0]
    period = max(2 * (length - 1), 1)
    step = torch.arange(length, frames, device=video.device) % period
    return torch.cat((video, video[torch.where(step < length, step, period - step)]), dim=0)


LAST_FRAME, PING_PONG = "last_frame", "ping_pong"
# the tail_padding widget: its values, each with the function that extends a driving video past
# its last frame and the words the loop's log line names it by
TAIL_PADDING = {
    LAST_FRAME: (hold_last, "last frame held"),
    PING_PONG: (ping_pong, "ping_pong padded"),
}


def as_numpy(images):
    """An IMAGE batch as a numpy array: a tensor's own memory, anything else through np.asarray."""
    import numpy as np
    import torch

    return images.numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
