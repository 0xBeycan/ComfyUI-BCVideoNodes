"""What the pose and the mask checks share: the guard constants, the metrics rows of each group
and of both, GuardFailed, the pose_data readers and the per-frame geometry both groups measure."""
from dataclasses import asdict
from typing import Optional, TypedDict

import numpy as np

from ...libs.pose_data import Detection, PoseData, PoseMeta


TORSO = [0, 1, 2, 5, 8, 11]  # nose, neck, shoulders, hips: cannot jump a quarter of the body in one frame
LIMB_ENDS = [3, 6, 4, 7, 9, 12, 10, 13, 18, 19]  # elbows, wrists, knees, ankles, feet
# The keypoint sets of pose_metas_original beside the body; a piece of mask holding any of
# them belongs to the person.
WHOLE_BODY = ("keypoints_body", "keypoints_left_hand", "keypoints_right_hand", "keypoints_face")
# Checks that cannot tell a defect from something the scene really does: reported, never stop.
WARNINGS = {"pose_limb_gap", "mask_attached_leak", "mask_specks", "mask_missed_limb"}
# In the order each group tests them on a frame; the report lists the checks in the order
# they first fired, and this order breaks the tie between two that first fire on one frame.
POSE_CHECKS = ("pose_incomplete", "pose_jump", "pose_spike", "pose_limb_gap", "subject_switch")
MASK_CHECKS = ("mask_empty", "mask_leak", "mask_attached_leak", "mask_fragmented", "mask_specks",
               "mask_missing_keypoints", "mask_missed_limb", "body_not_drawn", "mask_unstable")
BOX_MARGIN = 0.10
# Completeness is measured against the frames within this many either side. A limb that at
# least this share of them draw is one the pipeline can find on this material, so losing it
# is a defect; a limb the whole neighbourhood is missing is the person being framed that way
# (a close-up has no legs) and is not expected of this frame. Both numbers are about how fast
# a shot changes, not about any model: +/-8 frames is about a quarter of a second, and a
# quarter of the window is enough for a limb that is only visible part of the time. A loss
# longer than the window is expected by its own neighbours; body_not_drawn catches it.
COMPLETENESS_WINDOW = 8
COMPLETENESS_SHARE = 0.25
# A spike is out and back: the keypoint returns within this many frames to within half the
# jump threshold of where it left.
SPIKE_RETURN = 3
# pose_limb_gap: a limb missing for this many frames or more (a sixth of a second), and at
# most GAP_MAX (a second - a limb gone longer is the framing or the body hiding it, and a body
# the pose really lost for that long is body_not_drawn), that at least GAP_SHARE of the
# GAP_CONTEXT frames either side draw. Head limbs are left out: a head turned away from the
# camera draws no face, which is right.
GAP_MIN, GAP_MAX, GAP_CONTEXT, GAP_SHARE = 5, 30, 15, 0.8
# The zone around the drawn skeleton that is the body it accounts for: this many body scales
# either side of every drawn limb and keypoint (hands included). The body scale is the widest
# of the shoulders, the hips and 1.5 x neck-to-nose, whichever are drawn - the size of the
# person on this frame, which the detector box is not in a close-up.
SKELETON_REACH = 0.5
# mask_attached_leak compares the frame's mask with the union of the masks this many frames
# either side, grown by 2% of the person's size, and leaves out LEAK_REACH body scales around
# the drawn skeleton: a limb in motion falls inside one or the other, a patch of background
# that joins the mask for a frame does not. The full SKELETON_REACH would excuse background
# taken in against the body as well.
LEAK_WINDOW = 2
LEAK_REACH = 0.25
# The box-based checks compare the mask with the detector's box, so they only mean anything
# on a frame the detector and the pose model agree on. On a motion-blurred frame the box
# shrinks around the blurred body while the mask (carried by the tracker) still covers the
# person, which read as a leak; such a frame is left to the pose checks instead.
RELIABLE_KEYPOINTS = 8
RELIABLE_CONF = 0.5
# The mask is compared with the boxes of this many frames either side, grown by BOX_MARGIN:
# the person cannot leave that envelope in a few frames, while a mask that jumps to the
# background or to somebody else still falls outside it.
BOX_WINDOW = 4
SPECK_FRACTION = 0.01     # detached pieces above this fraction of the main region are reported
FRAGMENT_FRACTION = 0.05  # and above this one they count as a second object

# The per-frame measurements of each group, and of both together in the order `metrics`
# lists them. `frame` and `box_iou_prev` are in both: the mask checks need the box motion.
# `detected` and `persons` are the detector's, kept as data: the guard does not judge it.
POSE_ROW = ("frame", "detected", "persons", "pose_conf", "drawn_keypoints", "drawn_limbs",
            "box_iou_prev", "torso_jump", "pose_completeness", "lost_limbs", "limb_spikes", "limb_gaps")
MASK_ROW = ("frame", "mask_area", "mask_to_box", "box_reliable", "mask_outside_box", "attached_leak",
            "fragments", "keypoint_recall", "missed_keypoints", "missed_limbs", "body_not_drawn",
            "box_iou_prev", "mask_iou_prev")
PREPROCESS_ROW = ("frame", "detected", "persons", "pose_conf", "drawn_keypoints", "drawn_limbs",
                  "mask_area", "mask_to_box", "box_reliable", "mask_outside_box", "attached_leak", "fragments",
                  "keypoint_recall", "missed_keypoints", "missed_limbs", "body_not_drawn", "box_iou_prev",
                  "mask_iou_prev", "torso_jump", "pose_completeness", "lost_limbs", "limb_spikes", "limb_gaps")


# The rows above as the dicts the checks build: the same keys in the same order, which is the
# order the metrics JSON lists them in (tests/pipelines/test_guard_rows.py holds each to its tuple).
class PoseRow(TypedDict):
    frame: int
    detected: bool
    persons: int
    pose_conf: float
    drawn_keypoints: int
    drawn_limbs: int
    box_iou_prev: Optional[float]
    torso_jump: float
    pose_completeness: float
    lost_limbs: list[str]
    limb_spikes: list[str]
    limb_gaps: list[str]


class MaskRow(TypedDict):
    frame: int
    mask_area: float
    mask_to_box: float
    box_reliable: bool
    mask_outside_box: float
    attached_leak: float
    fragments: list[float]
    keypoint_recall: float
    missed_keypoints: list[str]
    missed_limbs: list[str]
    body_not_drawn: float
    box_iou_prev: Optional[float]
    mask_iou_prev: Optional[float]


class PreprocessRow(TypedDict):
    frame: int
    detected: bool
    persons: int
    pose_conf: float
    drawn_keypoints: int
    drawn_limbs: int
    mask_area: float
    mask_to_box: float
    box_reliable: bool
    mask_outside_box: float
    attached_leak: float
    fragments: list[float]
    keypoint_recall: float
    missed_keypoints: list[str]
    missed_limbs: list[str]
    body_not_drawn: float
    box_iou_prev: Optional[float]
    mask_iou_prev: Optional[float]
    torso_jump: float
    pose_completeness: float
    lost_limbs: list[str]
    limb_spikes: list[str]
    limb_gaps: list[str]


class GuardFailed(RuntimeError):
    """An enabled check failed; the message is the report."""


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _pose_inputs(pose_data: PoseData) -> tuple[list[PoseMeta], list[Detection]]:
    """The per-frame keypoints and detections of `pose_data`, checked."""
    pose_metas = pose_data.get("pose_metas_original") if isinstance(pose_data, dict) else None
    detections = pose_data.get("detections") if isinstance(pose_data, dict) else None
    if pose_metas is None or detections is None:
        raise ValueError("pose_data has no per-frame keypoints and detections; it must come from Pose Detection "
                         "or WanAnimate Preprocess")
    if len(pose_metas) != len(detections):
        raise ValueError(f"pose_data has {len(pose_metas)} frames of keypoints but {len(detections)} of detections")
    return pose_metas, detections


def _draw_threshold(pose_data: PoseData) -> float:
    """The keypoint confidence the pose images were drawn with: the guards judge the skeleton
    the diffusion model sees, so they count the keypoints and limbs that are drawn."""
    threshold = pose_data.get("draw_threshold") if isinstance(pose_data, dict) else None
    if threshold is None:
        raise ValueError("pose_data has no draw_threshold; it must come from Pose Detection or WanAnimate Preprocess")
    return threshold


def _thresholds(pose_data: PoseData, config):
    """What a check runs with: the draw threshold of `pose_data`, then every field of `config`."""
    return {"draw_threshold": _draw_threshold(pose_data), **asdict(config)}


def _in_frame(kps, W, H):
    return (kps[:, 0] >= 0) & (kps[:, 0] < W) & (kps[:, 1] >= 0) & (kps[:, 1] < H)


def box_sides(x1, y1, x2, y2):
    """The width and height of a box, at least one pixel each."""
    return max(x2 - x1, 1.0), max(y2 - y1, 1.0)


def _frame_pose(meta: PoseMeta, det: Detection, W, H, draw_threshold):
    """One frame's box size and diagonal, its body keypoints in pixels, which are drawn, and
    which of those lie inside the frame (a keypoint off the canvas is not seen)."""
    x1, y1, x2, y2 = det["bbox"]
    bw, bh = box_sides(x1, y1, x2, y2)
    diag = float(np.hypot(bw, bh))
    kps = np.asarray(meta["keypoints_body"], dtype=np.float64).copy()
    kps[:, 0] *= W
    kps[:, 1] *= H
    drawn = kps[:, 2] >= draw_threshold
    return bw, bh, diag, kps, drawn, drawn & _in_frame(kps, W, H)


def _box_iou_prev(detections: list[Detection], i):
    """Box IoU with the previous frame, None on the first frame or when either was not detected."""
    if i == 0:
        return None
    det, prev = detections[i], detections[i - 1]
    return _box_iou(det["bbox"], prev["bbox"]) if (det["score"] > 0 and prev["score"] > 0) else None


def _flag(flags, name, i):
    """Record in `flags` (check name -> frames) that the check `name` fired on frame `i`."""
    flags.setdefault(name, []).append(i)
