"""The one reader of a BBOX input, shared by Pose Detection and SAM3's box_keypoint mode so both
accept the same boxes and reject the same mistakes with the same message."""
import json
import numbers

KJ_KEYS = ("startX", "startY", "endX", "endY")


def _box(value):
    """One box as four floats (x1, y1, x2, y2), or None when `value` is not a box. A box is a
    KJNodes {"startX", "startY", "endX", "endY"} dict, put in corner order, or four or more
    numbers of which the first four are the corners (a fifth, a score, is not read)."""
    if isinstance(value, dict):
        if not all(k in value for k in KJ_KEYS):
            return None
        x1, y1, x2, y2 = (float(value[k]) for k in KJ_KEYS)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
    if isinstance(value, (list, tuple)) and len(value) >= 4 and all(
            isinstance(v, numbers.Real) and not isinstance(v, bool) for v in value):
        return tuple(float(v) for v in value[:4])
    return None


def parse_bboxes(bboxes, frames):
    """`frames` boxes as (x1, y1, x2, y2) float tuples in pixels, one per frame.

    Accepts a BBOX value or its JSON string: one box, or a list of one box per frame, each four
    numbers x1, y1, x2, y2 (anything after the fourth is ignored) or a KJNodes
    {"startX", "startY", "endX", "endY"} dict. A single box is used on every frame. numpy arrays
    and tensors are read as the nested lists they hold. What the caller makes of a box (score,
    widening) is its own business."""
    if isinstance(bboxes, str):
        try:
            bboxes = json.loads(bboxes)
        except json.JSONDecodeError as e:
            raise ValueError(f"bboxes must be a BBOX value or its JSON string, found {bboxes!r} ({e})") from e
    if hasattr(bboxes, "tolist") and not isinstance(bboxes, (list, tuple, dict)):
        bboxes = bboxes.tolist()
    single = _box(bboxes)
    if single is not None:
        boxes = [single]
    elif isinstance(bboxes, (list, tuple)) and len(bboxes):
        boxes = [_box(b.tolist() if hasattr(b, "tolist") else b) for b in bboxes]
        if any(b is None for b in boxes):
            raise ValueError(f"bboxes must hold (x1, y1, x2, y2) boxes (four numbers, or KJNodes startX/startY/endX/endY "
                             f"dicts), got {list(bboxes)[:3]!r}")
    else:
        raise ValueError(f"bboxes must be a box or a list of (x1, y1, x2, y2) boxes, got {bboxes!r}")
    bad = [list(b) for b in boxes if b[2] <= b[0] or b[3] <= b[1]]
    if bad:
        raise ValueError(f"bboxes must be (x1, y1, x2, y2) with x1 < x2 and y1 < y2, got {bad[:3]}")
    if len(boxes) == 1:
        return boxes * frames
    if len(boxes) != frames:
        raise ValueError(f"bboxes holds {len(boxes)} boxes for {frames} frames; give one per frame or a single one")
    return boxes
