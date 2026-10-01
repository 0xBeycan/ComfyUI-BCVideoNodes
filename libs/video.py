"""Helpers on video tensors: IMAGE batches, frames first."""


def hold_last(video, start, stop):
    """Frames ``start`` .. ``stop`` of ``video`` extended past its end by repeating its last frame."""
    return _extended(video, start, stop, lambda index, length: index.clamp(max=length - 1))


def ping_pong(video, start, stop):
    """Frames ``start`` .. ``stop`` of ``video`` extended past its end by playing it back and forth
    from its end: after its last frame L-1 come L-2, L-3, ... 0, then 1, 2, ... again (a turn does
    not repeat the edge frame), the padding of the official Wan 2.2 Animate (wan/animate.py
    inputs_padding) and Wan-Animate-2 (multiclip_utils.py zigzag_padding). A single frame is
    repeated."""
    import torch

    def source(index, length):
        period = max(2 * (length - 1), 1)
        step = index % period
        return torch.where(step < length, step, period - step)

    return _extended(video, start, stop, source)


def _extended(video, start, stop, source):
    """Frames ``start`` .. ``stop`` of ``video`` extended past its end, ``source(index, length)``
    naming the frame of the video each index shows: a view of the video when they all lie inside
    it, else one gather of just these frames (never the video extended as a whole)."""
    import torch

    if stop <= video.shape[0]:
        return video[start:stop]
    return video[source(torch.arange(start, stop, device=video.device), video.shape[0])]


LAST_FRAME, PING_PONG = "last_frame", "ping_pong"
# the tail_padding widget: its values, each with the function that gives a window of a driving
# video extended past its last frame and the words the loop's log line names it by
TAIL_PADDING = {
    LAST_FRAME: (hold_last, "last frame held"),
    PING_PONG: (ping_pong, "ping_pong padded"),
}


def as_numpy(images):
    """An IMAGE batch as a numpy array: a tensor's own memory, anything else through np.asarray."""
    import numpy as np
    import torch

    return images.numpy() if isinstance(images, torch.Tensor) else np.asarray(images)
