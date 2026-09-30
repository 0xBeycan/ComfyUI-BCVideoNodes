"""What the pose and the mask checks share: the guard constants, the metrics rows of each group
and of both, and of the SCAIL-2 guard, GuardFailed, the pose_data readers and the per-frame
geometry both groups measure."""
from dataclasses import asdict
from typing import Optional, TypedDict

import numpy as np

from ...libs.keypoints import LIMBS, in_frame
from ...libs.pose_data import Detection, PoseData, PoseMeta


TORSO = [0, 1, 2, 5, 8, 11]  # nose, neck, shoulders, hips: cannot jump a quarter of the body in one frame
LIMB_ENDS = [3, 6, 4, 7, 9, 12, 10, 13, 18, 19]  # elbows, wrists, knees, ankles, feet: pose_spike's keypoints
# The hand keypoint sets of pose_metas_original, and every keypoint set beside the body; a piece of
# mask holding any of them belongs to the person.
HANDS = ("keypoints_right_hand", "keypoints_left_hand")
WHOLE_BODY = ("keypoints_body", "keypoints_left_hand", "keypoints_right_hand", "keypoints_face")
# Only damage the diffusion model cannot absorb stops the workflow: a person the mask leaves empty,
# the head or a whole limb outside the mask, a region of her the model loses, a large piece torn off
# the mask; for SCAIL-2 no person to drive and no character on the reference. Every other check is a
# warning: reported, never stops.
WARNINGS = {"pose_jump", "pose_spike", "subject_switch", "mask_leak", "mask_attached_leak", "driving_empty",
            "reference_fragmented", "reference_misaligned"}
# In the order each group tests them on a frame; the report lists the checks in the order
# they first fired, and this order breaks the tie between two that first fire on one frame.
POSE_CHECKS = ("pose_jump", "pose_spike", "subject_switch")
MASK_CHECKS = ("mask_empty", "mask_leak", "mask_attached_leak", "mask_fragmented", "mask_head_out", "mask_limb_out",
               "mask_loss")
# The mask checks that read nothing from pose_data; the others need it.
POSE_FREE_MASK_CHECKS = ("mask_fragmented", "mask_loss")
# The SCAIL-2 guard's: the driving-frame checks - its own (driving_fragmented stands for
# mask_fragmented), then the Mask Guard's (mask_loss and the rest with pose_data) - then the
# reference checks.
SCAIL2_DRIVING_CHECKS = ("no_driving_person", "driving_empty", "driving_fragmented") + tuple(
    name for name in MASK_CHECKS if name != "mask_fragmented")
SCAIL2_REFERENCE_CHECKS = ("reference_empty", "reference_fragmented", "reference_misaligned")
SCAIL2_CHECKS = SCAIL2_DRIVING_CHECKS + SCAIL2_REFERENCE_CHECKS
# The SCAIL-2 guard's mask_loss needs pose_data: on its latent grid (LATENT_READ) a limb in motion
# empties whole cells of a correct mask, which only the pose tells from a dropped part of her.
SCAIL2_POSE_FREE_CHECKS = ("no_driving_person", "driving_empty", "driving_fragmented")
BOX_MARGIN = 0.10
# pose_spike: a spike is out and back: the keypoint returns within this many frames to within half
# the jump threshold of where it left.
SPIKE_RETURN = 3
# mask_attached_leak compares the frame's mask with the union of the masks this many frames
# either side, grown by 2% of the person's size, and leaves out LEAK_REACH body scales (the body
# scale: body_scale) around the drawn skeleton: a limb in motion falls inside one or the other, a
# patch of background that joins the mask for a frame does not. The same frames tell a piece of her
# the mask split off (her part holds it on two of them) from a leak island (mask.mask_regions).
LEAK_WINDOW = 2
LEAK_REACH = 0.25
# A limb's width, in body scales: a keypoint the pose model places closer to the frame edge than
# that is out of the shot (out_of_shot).
LIMB_WIDTH = 0.25
# mask_loss: a region of her the model loses. The mask holds it on the frame before a run and drops
# it on every frame of the run: a closed run of up to LOSS_WINDOW frames (a quarter of a second)
# that the frame after holds again, or, with pose_data, an open run to the last frame of a stretch
# of frames with a mask (the clip's end, or an empty stretch) or from the first. A limb that moved
# away over the run and came back leaves the same trace, but it is somewhere while it is away: mask
# the ends do not hold turns up within LOSS_REACH of the shorter side per frame of the run (up to
# LOSS_REACH_FRAMES frames) around the region, at least LOSS_GAIN of its area on average; a dropped
# part is nowhere. Background that joins the mask around a run and is gone beyond it blinks off the
# same way: the region counts when the pose has the body in it on a frame at the run's ends, or the
# mask holds it on another frame within LOSS_WINDOW on each side it has.
# Only a hand-sized loss or bigger counts: a piece of at least LOSS_HAND of her mask (on the frame
# the run is anchored on). Set on the test clips: the smallest real loss the models read was 2.1% of
# her (a hand at the frame edge), the largest piece below it that passed every other test 1.15% (a
# toy the mask took in blinking off); the crumbs a correct mask drops for a frame are well under 1%.
LOSS_WINDOW = 8
LOSS_REACH, LOSS_REACH_FRAMES = 0.05, 4
LOSS_GAIN = 0.25
LOSS_HAND = 0.015
# mask_head_out: the head outside the mask - the drawn nose, or at least `head_out_eyes_ears` of
# the drawn eyes and ears (MaskGuardConfig, where the default's reasons are), inside the frame and
# outside the (slightly grown) mask - fails.
HEAD_OUT_KEYPOINT = "nose"
HEAD_OUT_SIDES = ("r_eye", "l_eye", "r_ear", "l_ear")
# The final mask the Wan Animate workflow feeds the sampler: the raw mask grown by
# GrowMaskWithBlur (expand FINAL_GROW, tapered) and cut into BlockifyMask's blocks of FINAL_BLOCK
# px, laid from each frame's own grown box, so the grid moves from frame to frame; every block that
# holds a grown pixel is on (mask.block_grid, mask.final_mask). The model reads the final a block at
# a time: a region of her it loses is a whole block (mask_loss), on the raw mask a block the final
# of the frame leaves off. The final's outline lies FINAL_GROW to FINAL_GROW + FINAL_BLOCK beyond
# the raw mask's, FINAL_PAD on average, and moves by up to a block between frames with no change in
# the raw mask, so the measures of the final allow for it: mask_attached_leak grows the
# neighbouring frames' masks by a block, a detached piece is measured with FINAL_PAD taken off its
# outline, and a block the final drops counts only where it holds a drawn keypoint on every frame
# of the run (a block holding a keypoint inside the raw mask is always on).
FINAL_GROW, FINAL_BLOCK = 10, 32
FINAL_PAD = FINAL_GROW + FINAL_BLOCK // 2
# The driving mask as SCAIL-2 reads it. Core's WanSCAILToVideo area-resizes the colored mask to
# half the generation size and cuts each colour channel at 225/255, then _extract_mask_to_28ch
# area-pools that 8x to the latent grid, one cell per 16 x 16 px of the generation, and stacks 4
# frames per latent frame; nothing grows it and no noise mask follows. The model gets each cell as
# the share of it the person fills, and a cell reads as her when she fills at least LATENT_READ of
# it. A region of her the model loses is a whole cell of that grid (mask_loss); a drawn keypoint
# the reading holds is inside the mask (mask_head_out, mask_limb_out).
LATENT_READ = 0.5
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
POSE_ROW = ("frame", "detected", "persons", "pose_conf", "drawn_keypoints", "drawn_limbs", "box_iou_prev", "torso_jump",
            "limb_spikes")
MASK_ROW = ("frame", "mask_area", "mask_to_box", "box_reliable", "mask_outside_box", "attached_leak", "fragments",
            "head_out", "limbs_out", "box_iou_prev", "mask_loss")
PREPROCESS_ROW = ("frame", "detected", "persons", "pose_conf", "drawn_keypoints", "drawn_limbs", "mask_area", "mask_to_box",
                  "box_reliable", "mask_outside_box", "attached_leak", "fragments", "head_out", "limbs_out",
                  "box_iou_prev", "mask_loss", "torso_jump", "limb_spikes")

# The SCAIL-2 guard's per-frame measurements of the colored driving mask: its own, then the
# mask row's (the pose-based ones None without pose_data).
SCAIL2_ROW = ("frame", "mask_area", "fragments", "latent_kept", "mask_loss", "mask_to_box", "box_reliable",
              "mask_outside_box", "attached_leak", "head_out", "limbs_out", "box_iou_prev")
# and its one record of the reference mask; `cropped` is data: core center-crops the reference
# to the generation's aspect ratio whatever the guard says
SCAIL2_REFERENCE = ("mode", "area", "fragments", "cropped", "iou_first_frame", "scale_first_frame", "flags")

# The rows above as the dicts the checks build: the same keys in the same order, which is the
# order the metrics JSON lists them in (tests/pipelines/test_guard_rows.py holds each to its tuple).
# head_out names the head keypoints outside the mask, limbs_out the limbs wholly outside it
# (mask.limbs_out), mask_loss the largest share of her mask the model loses on the frame.
class PoseRow(TypedDict):
    frame: int
    detected: bool
    persons: int
    pose_conf: float
    drawn_keypoints: int
    drawn_limbs: int
    box_iou_prev: Optional[float]
    torso_jump: float
    limb_spikes: list[str]


class MaskRow(TypedDict):
    frame: int
    mask_area: float
    mask_to_box: Optional[float]
    box_reliable: bool
    mask_outside_box: Optional[float]
    attached_leak: Optional[float]
    fragments: list[float]
    head_out: list[str]
    limbs_out: list[str]
    box_iou_prev: Optional[float]
    mask_loss: Optional[float]


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
    head_out: list[str]
    limbs_out: list[str]
    box_iou_prev: Optional[float]
    mask_loss: Optional[float]
    torso_jump: float
    limb_spikes: list[str]


class Scail2Row(TypedDict):
    frame: int
    mask_area: float
    fragments: list[float]
    latent_kept: Optional[float]
    mask_loss: Optional[float]
    mask_to_box: Optional[float]
    box_reliable: bool
    mask_outside_box: Optional[float]
    attached_leak: Optional[float]
    head_out: list[str]
    limbs_out: list[str]
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
    """The keypoint confidence the pose images were drawn with. The guards count the model's
    keypoints and limbs that reach it (a closed decision), not what the images show: a part a
    draw rule, draw_head off or a 0 stick width leaves out of the images still counts."""
    threshold = pose_data.get("draw_threshold") if isinstance(pose_data, dict) else None
    if threshold is None:
        raise ValueError("pose_data has no draw_threshold; it must come from Pose Detection or WanAnimate Preprocess")
    return threshold


def _thresholds(pose_data: Optional[PoseData], config):
    """What a check runs with: the draw threshold of `pose_data` (None without it), then every
    field of `config`."""
    return {"draw_threshold": None if pose_data is None else _draw_threshold(pose_data), **asdict(config)}


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


def _hand(meta: PoseMeta, key, W, H, draw_threshold):
    """The drawn keypoints of the hand keypoint set `key` (one of HANDS) in pixels [K, 2]."""
    if key not in meta:
        return np.empty((0, 2))
    hand = _keypoint_rows(meta, key) * np.array([W, H, 1.0])
    return hand[hand[:, 2] >= draw_threshold, :2]


def _frame_pose(meta: PoseMeta, det: Detection, W, H, draw_threshold):
    """One frame's box size and diagonal, its body keypoints in pixels, which are drawn, and
    which of those lie inside the frame (a keypoint off the canvas is not seen)."""
    x1, y1, x2, y2 = det["bbox"]
    bw, bh = box_sides(x1, y1, x2, y2)
    diag = float(np.hypot(bw, bh))
    kps, drawn = _body(meta, W, H, draw_threshold)
    return bw, bh, diag, kps, drawn, drawn & in_frame(kps, W, H)


def body_scale(kps, drawn):
    """The person's size on this frame in pixels: the widest of the shoulders, the hips and
    1.5 x neck-to-nose, whichever are drawn (at least 1)."""
    def length(a, b):
        return float(np.hypot(*(kps[a, :2] - kps[b, :2]))) if drawn[a] and drawn[b] else 0.0

    return max(length(2, 5), length(8, 11), 1.5 * length(1, 0), 1.0)


def out_of_shot(point, W, H, scale):
    """Whether a keypoint the pose model placed at `point` is out of the shot: outside the frame
    or within LIMB_WIDTH body scales of its edge. The model places a joint it cannot see at the
    edge of what it sees."""
    x, y = point
    return min(x, W - 1 - x, y, H - 1 - y) < LIMB_WIDTH * scale


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
