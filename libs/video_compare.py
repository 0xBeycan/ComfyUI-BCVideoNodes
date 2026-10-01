"""Two IMAGE clips side by side in one frame, for the Video Comparer: the common geometry of the
clips, and the frames A | B, each built into one reused buffer that the video encoder reads before
the next is built. The browser then decodes a single stream, so the two halves cannot drift apart.
"""


def side_by_side_geometry(sides):
    """(frame count, height, width of one side) for (tag, IMAGE) pairs: the shorter clip's count and
    the larger frame, cut to even sides (4:2:0 wants them; an even half width also lands the A|B
    seam on a chroma block boundary)."""
    frames = min(images.shape[0] for _, images in sides)
    h = max(images.shape[1] for _, images in sides)
    w = max(images.shape[2] for _, images in sides)
    return frames, h - h % 2, w - w % 2


def fit_into(out, frame):
    """Writes `frame` [H, W, C] into `out` [h, w, 3]: a clip at least as large only loses the odd
    row / column an even size cuts off; a smaller clip is scaled to fit (bicubic, antialiased) and
    letterboxed on black."""
    import torch.nn.functional as F

    h, w = out.shape[0], out.shape[1]
    fh, fw = frame.shape[0], frame.shape[1]
    if fh >= h and fw >= w:
        out.copy_(frame[:h, :w, :3])
        return
    scale = min(w / fw, h / fh)
    nw, nh = max(1, round(fw * scale)), max(1, round(fh * scale))
    scaled = F.interpolate(frame[:, :, :3].float().movedim(-1, 0)[None], size=(nh, nw), mode="bicubic",
                           align_corners=False, antialias=True)
    out.zero_()
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y:y + nh, x:x + nw] = scaled[0].movedim(0, -1).clamp_(0, 1)


def side_by_side_frames(sides, frames, h, w):
    """Yields `frames` frames [h, w * len(sides), 3] with the (tag, IMAGE) sides left to right; the
    same buffer every time, so it is to be consumed before the next one is asked for."""
    import torch

    buffer = torch.empty((h, w * len(sides), 3), dtype=torch.float32)
    for i in range(frames):
        for k, (_, images) in enumerate(sides):
            fit_into(buffer[:, k * w:(k + 1) * w], images[i])
        yield buffer
