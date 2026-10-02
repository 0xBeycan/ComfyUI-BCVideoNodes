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


# Load Video's precision widget: the dtype the loaded clip is stored in
FP32, FP16 = "fp32", "fp16"
PRECISIONS = {FP32: "float32", FP16: "float16"}


def precision_dtype(precision):
    """The torch dtype of a precision widget value (PRECISIONS). Raises ValueError for another."""
    import torch

    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {', '.join(PRECISIONS)}; got {precision!r}.")
    return getattr(torch, PRECISIONS[precision])


def is_half(frames):
    """Whether the tensor `frames` is float16 or bfloat16."""
    import torch

    return frames.dtype in (torch.float16, torch.bfloat16)


def requantized(frames, out=None):
    """`frames` (a frame, or a chunk's window, of an IMAGE or MASK clip) ready for float32
    arithmetic: a half-precision tensor (is_half) as a new float32 one, or written into `out` (a
    float32 tensor of its shape), every value rounded to the nearest 8-bit level k / 255; any other
    tensor itself. float16 keeps every level within 2^-12 of
    it (its step near 1.0 is 2^-11), so a float16 clip of 8-bit frames (Load Video at precision
    fp16) comes back as exactly the float32 values a float32 load holds. Core resizes and scales a
    clip in the dtype it gets and cv2 refuses float16, so the pack reads a half clip through this
    a frame or a window at a time, never as a whole: the whole clip widened at once would give the
    saving back and hold both copies."""
    if not is_half(frames):
        return frames
    widened = frames.float() if out is None else out.copy_(frames)
    return widened.mul_(255).round_().div_(255)


class HalfFrames:
    """A half-precision IMAGE batch read as float32 numpy frames, each read requantized
    (requantized), so the batch is never widened as a whole: [i] is frame i, [a:b] those frames and
    [i, rows, columns] that region of frame i, each a new array (a region costs only its pixels).
    Iterating hands out the frames in two float32 frames used in turn, the next one requantized on a
    worker thread while the caller works on the one it has: the conversion runs while the caller's
    model does, instead of before it, and no frame-sized array is allocated per frame. A frame is
    valid until the next one is read. `shape` is the batch's, `dtype` the frames' (float32)."""

    def __init__(self, images):
        self.images = images
        self.shape = tuple(images.shape)

    @property
    def dtype(self):
        import numpy as np

        return np.dtype(np.float32)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        return requantized(self.images[index]).numpy()

    def __iter__(self):
        from concurrent.futures import ThreadPoolExecutor

        import torch

        if not len(self):
            return
        frames = [torch.empty(self.shape[1:], dtype=torch.float32) for _ in range(2)]

        def read(i):
            return requantized(self.images[i], frames[i % 2]).numpy()

        with ThreadPoolExecutor(1, thread_name_prefix="BCVideoNodes-frames") as worker:
            ahead = worker.submit(read, 0)
            for i in range(len(self)):
                frame = ahead.result()
                if i + 1 < len(self):
                    ahead = worker.submit(read, i + 1)  # into the other frame: this one stays as handed out
                yield frame


def as_numpy(images):
    """An IMAGE batch as numpy frames: a tensor's own memory, a half-precision tensor as HalfFrames
    (float32 frames, read one at a time; iterated, one frame ahead into two reused frames),
    anything else through np.asarray."""
    import numpy as np
    import torch

    if not isinstance(images, torch.Tensor):
        return np.asarray(images)
    return HalfFrames(images) if is_half(images) else images.numpy()
