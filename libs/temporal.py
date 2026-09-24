"""The temporal layer over a per-frame detector and a per-frame pose model: a person moves
continuously, so the frames around a frame say a great deal about what it should hold.

Two functions, the same idea applied to the box and to the keypoints:

  boxes_over_time      a frame the detector missed takes the box its nearest detections
                       either side imply instead of surrendering the whole frame, and every
                       box is then widened to the boxes around it
  keypoints_over_time  a keypoint the model lost for a frame or two is filled in from the
                       confident frames either side, and one that sits far from what those
                       frames imply is a glitch and is taken out

Neither invents evidence. An undetected frame keeps its score of -1, so the guard and the
mask prompt still know the detector failed there; and every keypoint value that did not come
from the model is named in the `source` array `keypoints_over_time` returns, so a filled
value can never be read as a measurement.

The module constants below are the measured defaults; `PoseConfig` in pipelines/pose.py
overrides the box window, the gap, the step and the residual per run through the functions'
arguments.
"""
import warnings

import numpy as np

# The detector's box jumps around between frames - it shrinks to the upper body when the
# person comes close, and around the blur when they move fast - and both the pose crop and
# the mask prompt then miss the legs or the feet. The box of each frame is widened to the
# boxes of its neighbours within this many frames, which the person cannot leave that fast.
BOX_WINDOW = 4

# Where a keypoint's value came from, per frame and keypoint.
MEASURED = 0    # the pose model's own output, untouched
RECOVERED = 1   # below the drawing threshold, filled in from the anchors either side
REPLACED = 2    # confident but far from its neighbours in time, and replaced by them
DROPPED = 3     # the same, with nothing trustworthy to put in its place

# A keypoint is an anchor when the drawing would draw it: draw_aapose_new keeps a limb only
# when both its ends reach 0.5, so 0.5 is the line this layer has to work on.
ANCHOR_CONF = 0.5
# The longest run of unanchored frames worth bridging. Measured on three clips, two thirds
# of the runs that fall below ANCHOR_CONF between two anchors are five frames or shorter.
# Bridging longer ones keeps paying in keypoints filled but stops paying in keypoints filled
# correctly: of everything a five-frame bridge writes, 68% lands within 0.05 box diagonals of
# where ViTPose puts the same keypoint, against 59% at eight frames and 47% at twelve. The
# difference is a keypoint that really is out of sight - a hand behind the back, a foot
# outside the crop - which neither model can see and this one would be inventing. Re-swept on
# eight clips: 60% of the runs are five frames or shorter, and on ViTPose the share of bridged
# values that land near a second model's falls off past six frames, so five stays.
MAX_GAP = 5
# A gap is filled from the straight line between the anchors either side of it, which only
# says anything while the person is not travelling faster than this fraction of their box
# diagonal per frame. Over the same clips the 99th percentile of the per-frame step of a
# confident keypoint is 0.019 to 0.059 box diagonals; past 0.06 the straight line is a guess,
# and doubling the limit to 0.12 wrote three times as many values that land far from
# ViTPose's without filling in a single extra limb. Re-swept on eight clips the same holds:
# on ViTPose 0.12 adds 38 values far from a second model's and at most 19 limb-frames over
# 3075 frames.
MAX_STEP = 0.06
# A keypoint is tested against the median of the anchored positions within this many frames
# either side, not against the line between the two nearest anchors. The line has the frames
# on either side of a glitch as its ends, so the glitch drags the line onto itself and the
# test rejects those neighbours instead of the glitch. A median over nine frames survives
# four wrong ones, and it needs a majority of them - MEDIAN_MIN - to say
# anything at all, so it stays silent about a keypoint whose neighbours are not confident
# either. Swept on three clips against ViTPose: a nine-frame window takes 13.4 box diagonals
# of error out of the keypoints ViTPose can judge, against 4.8 at seven frames and 10.2 at
# thirteen, where the window is long enough that real motion starts to read as a glitch.
# Re-swept on eight clips it still is: at MAX_RESIDUAL 0.25 seven frames
# take 22.7 out and eleven 26.5, but eleven falls apart below that tolerance (7.2 at 0.15
# against 20.6 for nine).
MEDIAN_WINDOW = 4
MEDIAN_MIN = 5
# Further than this from that median and the keypoint is a glitch rather than motion.
# Re-swept on all eight clips, judged against a second pose model: 0.15 - chosen on three
# slow clips - is harmful on the fast clips, where a limb moves up to 0.10 box diagonals a
# frame and real motion reads as a glitch; 0.18-0.25 remove the same error, and above 0.25
# glitches survive. On ViTPose the pass is neutral from 0.18 up (within one box diagonal
# either way) and harmful below. One number for all 133 keypoints, so it is the body and the hands it catches; a
# face keypoint never moves far enough from its neighbours in box diagonals to be rejected.
MAX_RESIDUAL = 0.25
# Taking a keypoint out of the anchors leaves the median of its neighbours cleaner than it
# was, which can expose the next one along. A second pass changed nothing on these three
# clips and costs one more median, which is worth it for a run these clips do not contain.
REJECT_PASSES = 2


def _envelope(near, score):
    """The union of the boxes `near` as one (x1, y1, x2, y2, score) box."""
    return np.array([min(b[0] for b in near), min(b[1] for b in near),
                     max(b[2] for b in near), max(b[3] for b in near), score])


def boxes_over_time(bboxes, box_window=BOX_WINDOW):
    """Every frame's person box as (x1, y1, x2, y2, score), made steady over time.

    A frame the detector missed comes in as the whole frame with a score of -1. The person
    was somewhere between where the neighbouring detections put them, so the box is taken
    from those instead; only its coordinates change, the score stays -1. Every box is then
    widened to the boxes within `box_window` frames either side.
    """
    boxes = [np.asarray(b, dtype=np.float64) for b in bboxes]
    detected = [i for i, b in enumerate(boxes) if b[4] > 0]
    if not detected:
        return boxes
    filled = []
    for i, box in enumerate(boxes):
        if box[4] > 0:
            filled.append(box)
            continue
        before = [j for j in detected if j < i]
        after = [j for j in detected if j > i]
        if before and after:
            lo, hi = boxes[before[-1]], boxes[after[0]]
            t = (i - before[-1]) / (after[0] - before[-1])
            corners = lo[:4] * (1 - t) + hi[:4] * t
        else:
            # nothing on one side: the person is where the last detection left them
            corners = boxes[(before or after)[-1 if before else 0]][:4]
        filled.append(np.array([*corners, box[4]]))
    widened = []
    for i, box in enumerate(filled):
        near = filled[max(0, i - box_window):i + box_window + 1]
        widened.append(_envelope(near, box[4]))
    return widened


def _neighbourhood_median(xy, anchor):
    """The componentwise median of the anchored positions within MEDIAN_WINDOW frames either
    side of each frame, and how many frames went into it."""
    B = len(xy)
    around = np.full((2 * MEDIAN_WINDOW + 1, *xy.shape), np.nan)
    for n, shift in enumerate(range(-MEDIAN_WINDOW, MEDIAN_WINDOW + 1)):
        here = slice(max(0, -shift), min(B, B - shift))
        there = slice(max(0, shift), min(B, B + shift))
        around[n, here] = np.where(anchor[there, :, None], xy[there], np.nan)
    seen = np.count_nonzero(~np.isnan(around[..., 0]), axis=0)
    with warnings.catch_warnings():
        # a keypoint with no confident frame anywhere near it has an all-NaN window; `seen`
        # is what the caller filters on, the median of that window is never read
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(around, axis=0), seen


def _brackets(anchor):
    """Per frame and keypoint, the nearest anchored frame strictly before and strictly after
    it, -1 when there is none."""
    frames = np.broadcast_to(np.arange(anchor.shape[0])[:, None], anchor.shape)
    before = np.full(anchor.shape, -1)
    before[1:] = np.maximum.accumulate(np.where(anchor, frames, -1), axis=0)[:-1]
    after = np.full(anchor.shape, anchor.shape[0])
    after[:-1] = np.minimum.accumulate(np.where(anchor, frames, anchor.shape[0])[::-1], axis=0)[::-1][1:]
    return before, np.where(after < anchor.shape[0], after, -1)


def _from_anchors(xy, conf, anchor, scale, max_gap, max_step):
    """What the anchors either side of each frame say: the position on the straight line
    between them, the confidence such a value deserves, and whether that pair is close enough
    in time and slow enough in space to be believed at all."""
    before, after = _brackets(anchor)
    # -1 means there is no anchor on that side; the values read at 0 are masked out by
    # `trusted` below and never reach the output
    lo, hi = np.clip(before, 0, None)[..., None], np.clip(after, 0, None)[..., None]
    span = np.maximum(after - before, 1)
    start, end = np.take_along_axis(xy, lo, axis=0), np.take_along_axis(xy, hi, axis=0)
    frames = np.broadcast_to(np.arange(len(xy))[:, None], anchor.shape)
    position = start + (end - start) * ((frames - before) / span)[..., None]
    confidence = np.minimum(np.take_along_axis(conf, lo[..., 0], axis=0),
                            np.take_along_axis(conf, hi[..., 0], axis=0))
    trusted = ((before >= 0) & (after >= 0) & (after - before <= max_gap + 1)
               & (np.linalg.norm(end - start, axis=2) <= max_step * scale[:, None] * span))
    return position, confidence, trusted


def all_measured(kp2ds):
    """The [frames, keypoints] source array before any label: every keypoint MEASURED."""
    return np.full(kp2ds.shape[:2], MEASURED, dtype=np.int8)


def keypoints_over_time(kp2ds, boxes, max_gap=MAX_GAP, max_step=MAX_STEP, max_residual=MAX_RESIDUAL):
    """`kp2ds` is [frames, keypoints, 3] - x, y in frame pixels and a confidence - and
    `boxes` the per-frame person box the distances are measured against. `max_gap`,
    `max_step` and `max_residual` are MAX_GAP, MAX_STEP and MAX_RESIDUAL unless overridden.

    Returns the same array with the keypoints the model dropped for a frame or two filled in
    from the confident frames either side, the ones that sit far from what those frames imply
    replaced or dropped, and a [frames, keypoints] array of MEASURED / RECOVERED / REPLACED /
    DROPPED saying which is which.
    """
    kp2ds = np.array(kp2ds, dtype=np.float64)
    source = all_measured(kp2ds)
    if len(kp2ds) < 3:
        return kp2ds, source
    xy, conf = kp2ds[..., :2], kp2ds[..., 2]
    # every distance below is in box diagonals, so that a threshold means the same thing on a
    # person who fills the frame and on one who is half of it away
    box = np.asarray(boxes, dtype=np.float64)
    scale = np.maximum(np.hypot(box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]), 1.0)

    anchor = conf >= ANCHOR_CONF
    rejected = np.zeros_like(anchor)
    for _ in range(REJECT_PASSES):
        median, seen = _neighbourhood_median(xy, anchor)
        glitch = (anchor & (seen >= MEDIAN_MIN)
                  & (np.linalg.norm(xy - median, axis=2) > max_residual * scale[:, None]))
        if not glitch.any():
            break
        anchor &= ~glitch
        rejected |= glitch

    position, confidence, trusted = _from_anchors(xy, conf, anchor, scale, max_gap, max_step)
    fill = ~anchor & trusted
    # a glitch with no trustworthy line to fall back on is worse than a missing keypoint: it
    # is the one thing that would be drawn in the wrong place and counted as evidence
    drop = rejected & ~fill
    xy[fill] = position[fill]
    conf[fill] = confidence[fill]
    conf[drop] = 0.0
    source[fill] = np.where(rejected[fill], REPLACED, RECOVERED)
    source[drop] = DROPPED
    return kp2ds, source


def widen_over_time(bboxes, box_window=BOX_WINDOW):
    """The box logic without the temporal layer: each detected box widened to the boxes of
    its neighbours within `box_window` frames. Undetected frames (score -1) are left alone:
    they are the whole frame already."""
    out = []
    for i, bbox in enumerate(bboxes):
        if bbox[4] <= 0:
            out.append(bbox)
            continue
        near = [b for b in bboxes[max(0, i - box_window):i + box_window + 1] if b[4] > 0]
        out.append(_envelope(near, bbox[4]))
    return out
