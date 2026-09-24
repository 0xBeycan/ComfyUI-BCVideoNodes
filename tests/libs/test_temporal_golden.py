"""G13: the temporal layer, bit exact, recorded from the production code (specs refactor plan
8.2): `boxes_over_time` and `keypoints_over_time` on a seeded 60-frame noisy sequence at two
parameter sets, `snap_to_frame` and `widen_over_time` on a six-box table, and the undetected
entries both box functions hand back as the very objects they were given.

No cv2 pixel goes into these digests, so they do not depend on the cv2 build. They are in
tests/goldens/test_temporal_golden.json:

    python -m pytest tests/libs/test_temporal_golden.py
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from golden import check, digest  # noqa: E402
from pose_fakes import pose  # noqa: E402

from bcvideonodes.libs import temporal  # noqa: E402

N, K = 60, 133
BOX_W, BOX_H = 120.0, 300.0
UNDETECTED = (0, 1, 17, 30, 31, 32, 59)
# (box_window, max_gap, max_step, max_residual): the defaults, and a second set that bridges
# less, over faster motion, and calls a glitch sooner
PARAMETERS = {
    "defaults": (temporal.BOX_WINDOW, temporal.MAX_GAP, temporal.MAX_STEP, temporal.MAX_RESIDUAL),
    "tight": (1, 2, 0.12, 0.15),
}


def walk(i):
    """The person's box on frame i without noise: two pixels a frame to the right."""
    return 100.0 + 2.0 * i, 50.0


def sequence():
    """60 frames of a walking person: detector boxes with noise and seven undetected frames
    (the whole frame, score -1), and 133 keypoints with noise into which unconfident runs of
    1, 2, 5 and 6 frames, a jump too fast to bridge at the defaults, single and double glitches
    and a glitch on the last frame are written."""
    rng = np.random.default_rng(13)
    boxes = []
    for i in range(N):
        if i in UNDETECTED:
            boxes.append(np.array([0.0, 0.0, 640.0, 960.0, -1.0]))
            continue
        x, y = walk(i)
        jitter = rng.normal(0.0, 1.5, 4)
        boxes.append(np.array([x + jitter[0], y + jitter[1], x + BOX_W + jitter[2], y + BOX_H + jitter[3],
                               0.9 - abs(rng.normal(0.0, 0.03))]))
    kp = np.zeros((N, K, 3))
    k = np.arange(K)
    for i in range(N):
        x, y = walk(i)
        kp[i, :, 0] = x + 10.0 + (k % 10) * 10.0 + rng.normal(0.0, 0.8, K)
        kp[i, :, 1] = y + 10.0 + (k // 10) * 20.0 + rng.normal(0.0, 0.8, K)
        kp[i, :, 2] = np.clip(0.9 + rng.normal(0.0, 0.03, K), 0.0, 1.0)
    kp[10, 5, 2] = 0.2                      # one frame
    kp[20:22, 6, 2] = 0.2                   # two frames
    kp[40:45, 7, 2] = 0.3                   # five: MAX_GAP
    kp[45:51, 8, 2] = 0.3                   # six: one more than MAX_GAP
    kp[24:26, 9, 2] = 0.2                   # a gap over a 100 px jump
    kp[26:, 9, 0] += 100.0
    kp[15, 20, 0] += 150.0                  # a single glitch
    kp[33:35, 21, 0] += 150.0               # two wrong frames in a row
    kp[59, 22, 0] += 150.0                  # a glitch with nothing after it
    return boxes, kp


def listed(arrays):
    """One digest over a list of arrays, each with its own dtype and shape."""
    return digest([digest(a) for a in arrays])


def values(arrays):
    return repr([(a.dtype.str, a.tolist()) for a in arrays])


@pytest.mark.parametrize("name", list(PARAMETERS))
def test_temporal_golden(name):
    box_window, max_gap, max_step, max_residual = PARAMETERS[name]
    raw, kp = sequence()
    boxes = temporal.boxes_over_time(raw, box_window)
    out, source = temporal.keypoints_over_time(kp, boxes, max_gap=max_gap, max_step=max_step,
                                               max_residual=max_residual)
    if name == "defaults":
        # the sequence reaches every label
        assert {temporal.RECOVERED, temporal.REPLACED, temporal.DROPPED} <= set(np.unique(source).tolist())
    check(__file__, f"{name}.input", digest(kp))
    check(__file__, f"{name}.boxes_over_time", listed(boxes))
    check(__file__, f"{name}.keypoints_over_time", digest(out))
    check(__file__, f"{name}.source", digest(source))


W, H = 120, 160
# six boxes on a 120x160 frame: inside, 5 px from the left edge, undetected, close to the top
# and the bottom, float32 and 4 px from the right edge, undetected again
TABLE = (
    np.array([30.0, 20.0, 90.0, 130.0, 0.9]),
    np.array([5.0, 20.0, 65.0, 130.0, 0.8]),
    np.array([0.0, 0.0, 120.0, 160.0, -1.0]),
    np.array([40.0, 3.0, 100.0, 150.0, 0.7]),
    np.array([50.0, 30.0, 116.0, 140.0, 0.95], dtype=np.float32),
    np.array([0.0, 0.0, 120.0, 160.0, -1.0]),
)


def test_snap_to_frame_golden():
    snapped = [pose.snap_to_frame(b, W, H) for b in TABLE]
    check(__file__, "snap_to_frame", values(snapped))
    check(__file__, "snap_to_frame.digest", listed(snapped))


@pytest.mark.parametrize("box_window", [0, 1, 4])
def test_widen_over_time_golden(box_window):
    table = list(TABLE)
    widened = pose.widen_over_time(table, box_window)
    # an undetected frame is handed back as it came in, the same object
    assert widened[2] is table[2] and widened[5] is table[5]
    check(__file__, f"widen_over_time.{box_window}", values(widened))
    check(__file__, f"widen_over_time.{box_window}.digest", listed(widened))


def test_boxes_over_time_without_a_detection_golden():
    table = [TABLE[2], TABLE[5]]
    kept = temporal.boxes_over_time(table)
    # nothing to fill from: the float64 boxes come back as the same objects
    assert kept[0] is table[0] and kept[1] is table[1]
    check(__file__, "boxes_over_time.undetected", values(kept))
