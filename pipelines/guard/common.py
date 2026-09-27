"""What the pose and the mask checks share: the guard constants, the metrics rows of each group
and of both, and of the SCAIL-2 guard, GuardFailed, the pose_data readers and the per-frame
geometry both groups measure."""
from dataclasses import asdict
from typing import Optional, TypedDict

import numpy as np

from ...libs.keypoints import L_HIP, L_SHOULDER, LIMBS, NECK, R_HIP, R_SHOULDER
from ...libs.pose_data import Detection, PoseData, PoseMeta


TORSO = [0, 1, 2, 5, 8, 11]  # nose, neck, shoulders, hips: cannot jump a quarter of the body in one frame
LIMB_ENDS = [3, 6, 4, 7, 9, 12, 10, 13, 18, 19]  # elbows, wrists, knees, ankles, feet
# Each arm and leg beyond its torso joint (elbow, wrist; knee, ankle, foot), and the arm or leg
# every body keypoint carries: the torso joint its own, a limb keypoint the one it is part of.
R_ARM, L_ARM, R_LEG, L_LEG = (3, 4), (6, 7), (9, 10, 19), (12, 13, 18)
CARRIES = {2: R_ARM, 3: R_ARM, 4: R_ARM, 5: L_ARM, 6: L_ARM, 7: L_ARM,
           8: R_LEG, 9: R_LEG, 10: R_LEG, 19: R_LEG, 11: L_LEG, 12: L_LEG, 13: L_LEG, 18: L_LEG}
# The hand keypoints of each arm (a wrist's own hand).
HANDS = {R_ARM: "keypoints_right_hand", L_ARM: "keypoints_left_hand"}
# The keypoint sets of pose_metas_original beside the body; a piece of mask holding any of
# them belongs to the person.
WHOLE_BODY = ("keypoints_body", "keypoints_left_hand", "keypoints_right_hand", "keypoints_face")
# Only damage diffusion cannot absorb stops: an empty, leaking or split mask, a torso jump, a
# subject switch; for SCAIL-2 no person to drive and no character on the reference. Every other
# check is a warning: reported, never stops.
WARNINGS = {"pose_incomplete", "pose_spike", "pose_limb_gap", "mask_attached_leak", "mask_specks",
            "mask_missing_keypoints", "mask_missed_limb", "body_not_drawn", "mask_unstable", "mask_loss",
            "driving_empty", "driving_fragmented", "reference_fragmented", "reference_misaligned"}
# In the order each group tests them on a frame; the report lists the checks in the order
# they first fired, and this order breaks the tie between two that first fire on one frame.
POSE_CHECKS = ("pose_incomplete", "pose_jump", "pose_spike", "pose_limb_gap", "subject_switch")
MASK_CHECKS = ("mask_empty", "mask_leak", "mask_attached_leak", "mask_fragmented", "mask_specks",
               "mask_missing_keypoints", "mask_missed_limb", "body_not_drawn", "mask_unstable", "mask_loss")
# The mask checks that read nothing from pose_data; the others need it.
POSE_FREE_MASK_CHECKS = ("mask_fragmented", "mask_specks", "mask_loss")
# The SCAIL-2 guard's: the driving-frame checks - its own, then the Mask Guard's (mask_loss
# always, the rest with pose_data; driving_fragmented stands for mask_fragmented and mask_specks,
# a split-up driving mask being normal on SCAIL-2's material) - then the reference checks.
SCAIL2_DRIVING_CHECKS = ("no_driving_person", "driving_empty", "driving_fragmented") + tuple(
    name for name in MASK_CHECKS if name not in ("mask_fragmented", "mask_specks"))
SCAIL2_REFERENCE_CHECKS = ("reference_empty", "reference_fragmented", "reference_misaligned")
SCAIL2_CHECKS = SCAIL2_DRIVING_CHECKS + SCAIL2_REFERENCE_CHECKS
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
# A limb's width, in body scales. A joint closer to the frame edge than that is cut by it; the
# end of a limb is the disc of that width around its keypoint; a limb that runs outside the mask
# for more than that beyond its end's own distance from it lies beside the body, not out of it.
LIMB_WIDTH = 0.25
# mask_loss: a region the mask holds on the frame before a run of up to LOSS_WINDOW frames (a
# quarter of a second) and on the frame after it, and drops on every frame of the run. A limb
# that moved away over the run and came back leaves the same trace, but it is somewhere while it
# is away: mask the ends do not hold turns up within LOSS_REACH of the shorter side per frame of
# the run (up to LOSS_REACH_FRAMES frames) around the region, at least LOSS_GAIN of its area on
# average; a dropped part is nowhere. A run of two frames or more has to be LOSS_RUN times as
# thick as max_mask_loss: over several frames the mask's own changes leave out-and-back traces
# too (a limb moving inside the area both ends hold, background both ends take in between the
# legs). When the drawn skeleton crosses the region on at least LOSS_ON_BODY of the frames of the
# run, the pose sees the body there, and LOSS_DRAWN of max_mask_loss is thick enough. No model
# reads the raw mask: the Wan Animate workflow grows it into the final mask (FINAL_GROW,
# FINAL_BLOCK below) first, so on the raw mask a region is dropped only where the final of every
# frame of the run leaves it out - what the grow restores never reaches the sampler.
LOSS_WINDOW = 8
LOSS_REACH, LOSS_REACH_FRAMES = 0.05, 4
LOSS_GAIN = 0.25
LOSS_RUN = 2.0
LOSS_DRAWN = 0.5
LOSS_ON_BODY = 0.5
# The final mask the Wan Animate workflow feeds the sampler: the raw mask grown by
# GrowMaskWithBlur (expand FINAL_GROW, tapered) and cut into BlockifyMask's blocks of FINAL_BLOCK
# px. The Mask Guard's mask_loss counts on the raw mask only what that final leaves out (above);
# the WanAnimate Preprocess Guard judges the final itself. BlockifyMask lays its grid from each
# frame's own box, so the grid moves from frame to frame, and fills every block that holds a
# grown pixel: the final's outline lies
# FINAL_GROW to FINAL_GROW + FINAL_BLOCK beyond the raw mask's, FINAL_PAD on average, and moves by
# up to a block between frames with no change in the raw mask. Its outline says nothing finer, so
# the measures of the final allow for it: mask_attached_leak grows the neighbouring frames' masks
# by a block, body_not_drawn grows the skeleton zone by FINAL_PAD, a detached piece is measured
# with FINAL_PAD taken off its outline, and mask_loss counts only a region holding a drawn
# keypoint on every frame of its run (FINAL_ON_BODY; a block holding a keypoint inside the raw
# mask is always on, a limb line runs through background blocks the moving grid turns on and off).
FINAL_GROW, FINAL_BLOCK = 10, 32
FINAL_PAD = FINAL_GROW + FINAL_BLOCK // 2
FINAL_ON_BODY = 1.0
# The driving mask as SCAIL-2 reads it. Core's WanSCAILToVideo area-resizes the colored mask to
# half the generation size and cuts each colour channel at 225/255, then _extract_mask_to_28ch
# area-pools that 8x to the latent grid, one cell per 16 x 16 px of the generation, and stacks 4
# frames per latent frame; nothing grows it and no noise mask follows. The model gets each cell as
# the share of it the person fills, and a cell reads as her when she fills at least LATENT_READ of
# it. A hole or a sliver inside a cell, or a keypoint just outside her in a cell she fills half
# of, never reaches the model; a dropped hand empties cells. So the SCAIL-2 guard counts a dropped
# region (mask_loss) only where the mask and that reading both drop it, and a drawn keypoint the
# reading holds is inside the mask (mask_missing_keypoints, mask_missed_limb).
LATENT_READ = 0.5
# The person faces the camera when her right shoulder is left of her left one on the image by at
# least this share of her torso (neck to the middle of the hips): turned to profile the shoulders
# close up (a quarter turn short of profile they are still about half the torso apart), with her
# back to the camera they swap sides.
FRONTAL_SHARE = 0.3
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
# Without pose_data the mask row's pose-based measurements are None (lists empty).
POSE_ROW = ("frame", "detected", "persons", "pose_conf", "drawn_keypoints", "drawn_limbs",
            "box_iou_prev", "torso_jump", "pose_completeness", "lost_limbs", "limb_spikes", "limb_gaps")
MASK_ROW = ("frame", "mask_area", "mask_to_box", "box_reliable", "mask_outside_box", "attached_leak",
            "fragments", "keypoint_recall", "missed_keypoints", "missed_limbs", "body_not_drawn",
            "box_iou_prev", "mask_iou_prev", "mask_loss", "mask_loss_run", "mask_loss_drawn")
PREPROCESS_ROW = ("frame", "detected", "persons", "pose_conf", "drawn_keypoints", "drawn_limbs",
                  "mask_area", "mask_to_box", "box_reliable", "mask_outside_box", "attached_leak", "fragments",
                  "keypoint_recall", "missed_keypoints", "missed_limbs", "body_not_drawn", "box_iou_prev",
                  "mask_iou_prev", "mask_loss", "mask_loss_run", "mask_loss_drawn", "torso_jump", "pose_completeness", "lost_limbs", "limb_spikes",
                  "limb_gaps")

# The SCAIL-2 guard's per-frame measurements of the colored driving mask: its own, then the
# mask row's (the pose-based ones None without pose_data).
SCAIL2_ROW = ("frame", "mask_area", "fragments", "latent_kept", "mask_iou_prev", "mask_loss", "mask_loss_run",
              "mask_loss_drawn", "mask_to_box", "box_reliable", "mask_outside_box", "attached_leak",
              "keypoint_recall", "missed_keypoints", "missed_limbs", "body_not_drawn", "box_iou_prev")
# and its one record of the reference mask; `cropped` is data: core center-crops the reference
# to the generation's aspect ratio whatever the guard says
SCAIL2_REFERENCE = ("mode", "area", "fragments", "cropped", "iou_first_frame", "scale_first_frame", "flags")

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
    mask_to_box: Optional[float]
    box_reliable: bool
    mask_outside_box: Optional[float]
    attached_leak: Optional[float]
    fragments: list[float]
    keypoint_recall: Optional[float]
    missed_keypoints: list[str]
    missed_limbs: list[str]
    body_not_drawn: Optional[float]
    box_iou_prev: Optional[float]
    mask_iou_prev: Optional[float]
    mask_loss: Optional[float]
    mask_loss_run: Optional[float]
    mask_loss_drawn: Optional[float]


class PreprocessRow(TypedDict):
    frame: int
    detected: bool
    persons: int
    pose_conf: float
    drawn_keypoints: int
    drawn_limbs: int
    mask_area: float
    mask_to_box: Optional[float]
    box_reliable: bool
    mask_outside_box: Optional[float]
    attached_leak: Optional[float]
    fragments: list[float]
    keypoint_recall: Optional[float]
    missed_keypoints: list[str]
    missed_limbs: list[str]
    body_not_drawn: Optional[float]
    box_iou_prev: Optional[float]
    mask_iou_prev: Optional[float]
    mask_loss: Optional[float]
    mask_loss_run: Optional[float]
    mask_loss_drawn: Optional[float]
    torso_jump: float
    pose_completeness: float
    lost_limbs: list[str]
    limb_spikes: list[str]
    limb_gaps: list[str]


class Scail2Row(TypedDict):
    frame: int
    mask_area: float
    fragments: list[float]
    latent_kept: Optional[float]
    mask_iou_prev: Optional[float]
    mask_loss: Optional[float]
    mask_loss_run: Optional[float]
    mask_loss_drawn: Optional[float]
    mask_to_box: Optional[float]
    box_reliable: bool
    mask_outside_box: Optional[float]
    attached_leak: Optional[float]
    keypoint_recall: Optional[float]
    missed_keypoints: list[str]
    missed_limbs: list[str]
    body_not_drawn: Optional[float]
    box_iou_prev: Optional[float]


class Scail2Reference(TypedDict):
    """The SCAIL-2 guard's measurements of the reference mask (one record per run)."""
    mode: Optional[str]
    area: float
    fragments: list[float]
    cropped: float
    iou_first_frame: Optional[float]
    scale_first_frame: Optional[float]
    flags: list[str]


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


def _thresholds(pose_data: Optional[PoseData], config):
    """What a check runs with: the draw threshold of `pose_data` (None without it), then every
    field of `config`."""
    return {"draw_threshold": None if pose_data is None else _draw_threshold(pose_data), **asdict(config)}


def _in_frame(kps, W, H):
    return (kps[:, 0] >= 0) & (kps[:, 0] < W) & (kps[:, 1] >= 0) & (kps[:, 1] < H)


def box_sides(x1, y1, x2, y2):
    """The width and height of a box, at least one pixel each."""
    return max(x2 - x1, 1.0), max(y2 - y1, 1.0)


def _keypoint_rows(meta: PoseMeta, key):
    """The keypoint set `key` of `meta` as [K, 3] float64 rows (x, y, confidence)."""
    return np.asarray(meta[key], dtype=np.float64).reshape(-1, 3)


def _body(meta: PoseMeta, W, H, draw_threshold):
    """One frame's body keypoints in pixels [20, 3] and which of them are drawn."""
    kps = _keypoint_rows(meta, "keypoints_body") * np.array([W, H, 1.0])
    return kps, kps[:, 2] >= draw_threshold


def _hand(meta: PoseMeta, arm, W, H, draw_threshold, fingers_only=False):
    """The drawn keypoints of the hand of `arm` (R_ARM or L_ARM) in pixels [K, 2]; with
    `fingers_only`, without the hand's own wrist point (its first keypoint)."""
    if HANDS[arm] not in meta:
        return np.empty((0, 2))
    hand = _keypoint_rows(meta, HANDS[arm])[1 if fingers_only else 0:] * np.array([W, H, 1.0])
    return hand[hand[:, 2] >= draw_threshold, :2]


def _frame_pose(meta: PoseMeta, det: Detection, W, H, draw_threshold):
    """One frame's box size and diagonal, its body keypoints in pixels, which are drawn, and
    which of those lie inside the frame (a keypoint off the canvas is not seen)."""
    x1, y1, x2, y2 = det["bbox"]
    bw, bh = box_sides(x1, y1, x2, y2)
    diag = float(np.hypot(bw, bh))
    kps, drawn = _body(meta, W, H, draw_threshold)
    return bw, bh, diag, kps, drawn, drawn & _in_frame(kps, W, H)


def body_scale(kps, drawn):
    """The person's size on this frame in pixels: the widest of the shoulders, the hips and
    1.5 x neck-to-nose, whichever are drawn (at least 1)."""
    def length(a, b):
        return float(np.hypot(*(kps[a, :2] - kps[b, :2]))) if drawn[a] and drawn[b] else 0.0

    return max(length(2, 5), length(8, 11), 1.5 * length(1, 0), 1.0)


def facing_camera(kps, drawn):
    """Whether the person faces the camera (FRONTAL_SHARE). Without both shoulders it cannot be
    told and counts as facing; without the neck or a hip only their sides are compared."""
    if not (drawn[R_SHOULDER] and drawn[L_SHOULDER]):
        return True
    width = kps[L_SHOULDER, 0] - kps[R_SHOULDER, 0]
    hips = [kps[j, :2] for j in (R_HIP, L_HIP) if drawn[j]]
    if not (drawn[NECK] and hips):
        return bool(width > 0)
    return bool(width >= FRONTAL_SHARE * float(np.hypot(*(kps[NECK, :2] - np.mean(hips, axis=0)))))


def out_of_shot(point, W, H, scale):
    """Whether a keypoint the pose model placed at `point` is out of the shot: outside the frame
    or within LIMB_WIDTH body scales of its edge. The model places a joint it cannot see at the
    edge of what it sees."""
    x, y = point
    return min(x, W - 1 - x, y, H - 1 - y) < LIMB_WIDTH * scale


def _to_segment(point, a, b):
    """The distance from `point` to the segment a-b."""
    ab = b - a
    t = float(np.clip(np.dot(point - a, ab) / max(float(np.dot(ab, ab)), 1e-9), 0.0, 1.0))
    return float(np.hypot(*(point - (a + t * ab))))


def hidden_by_body(point, kps, drawn, hands, own, scale):
    """Whether a keypoint the pose model placed at `point` lies on another drawn part of the
    person, within SKELETON_REACH body scales of it: the torso, the head, the other limbs or a
    hand. What the body, the hair or a hand hides, the model places on what hides it. `own` is
    the arm or leg the limb belongs to, which does not hide it; `hands` the drawn keypoints of
    the hands of the other arms."""
    reach = SKELETON_REACH * scale
    for a, b in LIMBS:
        if drawn[a] and drawn[b] and a not in own and b not in own and _to_segment(point, kps[a, :2], kps[b, :2]) <= reach:
            return True
    others = [kps[j, :2] for j in np.flatnonzero(drawn) if j not in own] + list(hands)
    return any(float(np.hypot(*(point - p))) <= reach for p in others)


def accounted_limbs(meta: PoseMeta, W, H, draw_threshold):
    """Per limb of LIMBS, whether the frame accounts for it: the limb is drawn, or it cannot be
    seen - the person does not face the camera, or each of its undrawn ends is out of the shot
    or hidden by another part of the body. Only the rest is a missing limb: the person faces the
    camera, the limb is in the shot and in sight, and the pose does not draw it."""
    kps, drawn = _body(meta, W, H, draw_threshold)
    limbs = np.array([drawn[a] and drawn[b] for a, b in LIMBS], dtype=bool)
    if limbs.all():
        return limbs
    if not facing_camera(kps, drawn):
        return np.ones(len(LIMBS), dtype=bool)
    scale = body_scale(kps, drawn)
    for j, (a, b) in enumerate(LIMBS):
        if limbs[j]:
            continue
        own = set(CARRIES.get(a, ())) | set(CARRIES.get(b, ()))
        arms = [arm for arm in HANDS if not own & set(arm)]
        hands = [p for arm in arms for p in _hand(meta, arm, W, H, draw_threshold)]
        limbs[j] = all(out_of_shot(kps[e, :2], W, H, scale) or hidden_by_body(kps[e, :2], kps, drawn, hands, own, scale)
                       for e in (a, b) if not drawn[e])
    return limbs


def out_of_shot_limbs(meta: PoseMeta, W, H, draw_threshold):
    """The limbs with one end drawn and the other out of the shot, as (drawn end, where the
    model placed the other) in pixels: the part of the body in the frame that the pose image
    cannot draw, since the rest of the limb is beyond the edge."""
    kps, drawn = _body(meta, W, H, draw_threshold)
    scale = body_scale(kps, drawn)
    out = []
    for a, b in LIMBS:
        if drawn[a] == drawn[b]:
            continue
        end, other = (a, b) if drawn[a] else (b, a)
        if out_of_shot(kps[other, :2], W, H, scale):
            out.append((kps[end, :2], kps[other, :2]))
    return out


def _box_iou_prev(detections: list[Detection], i):
    """Box IoU with the previous frame, None on the first frame or when either was not detected."""
    if i == 0:
        return None
    det, prev = detections[i], detections[i - 1]
    return _box_iou(det["bbox"], prev["bbox"]) if (det["score"] > 0 and prev["score"] > 0) else None


def _flag(flags, name, i):
    """Record in `flags` (check name -> frames) that the check `name` fired on frame `i`."""
    flags.setdefault(name, []).append(i)
