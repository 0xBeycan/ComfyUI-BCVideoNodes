"""Checks on the preprocess output: is the pose plausible and does the mask agree with it?

Every check is normalised by the person's size (the detector box) and, where possible,
crosses one signal with another that was produced independently: the SAM mask against the
pose model's keypoints, the mask's motion against the box's motion. Pure geometry on the mask
cannot tell a stable wrong mask from a right one; the pose can.

A pose check has to judge the drawn skeleton, not whether a component hiccuped. A detector
that missed a box or saw a second person changes nothing about what comes out - the mask is
carried by the tracker and the pose model poses the foreground person either way - so those
are warnings. A confidence threshold is worse than useless: it is calibrated to one model's
heatmap maxima, and on a different pose model a skeleton that visibly collapsed still read as
confident. What survives a change of model is the skeleton against its own neighbours.

The checks come in two groups that run on their own: `check_pose` needs only `pose_data`,
`check_mask` needs the mask and the `pose_data` it is checked against (the keypoints and
boxes are what the mask is judged by). `combine_guards` merges the two results into the one
report, metrics and timeline of the whole preprocess.

Pose checks, per frame, from `pose_data` (keypoints, detections):

  no_detection        the detector found nobody, the whole frame was used as the box; the
                      mask is unaffected, so this is a warning
  pose_incomplete     the frame draws less than `min_pose_completeness` of the limbs the
                      frames around it draw (a collapsed or half-missing skeleton); the
                      report names the limbs it lost
  pose_jump           torso keypoints moved more than `max_torso_jump` box diagonals in
                      one frame while the box hardly moved (pose glitch, not motion)
  subject_switch      box IoU with the previous frame below 0.3 (the detector picked
                      someone / something else)
  multi_person        the detector saw more than one person at 30% or more; a second person
                      in shot is scene content, so this is a warning

Mask checks, per frame, from the mask against `pose_data`:

  mask_empty          no mask, or mask area below `min_mask_to_box` of the box area - only
                      on a frame whose box the detector and the pose model agree on, since
                      an empty mask elsewhere is the pose failing, not the mask
  mask_leak           more than `max_mask_outside_box` of the mask lies outside the boxes of
                      the frames around it, grown by 10% (background or a neighbour pulled
                      in); only checked on a frame whose box the detector and the pose model
                      agree on
  mask_fragmented     a second region at least 5% of the largest one (a ghost, a second
                      person, a split body); smaller detached pieces such as a shadow
                      blob are reported as mask_specks (warning only). A piece holding the
                      person's own confident keypoints is that same person - a hand the frame
                      edge cut away from the body - and counts as neither
  mask_missing_keypoints
                      fewer than `min_keypoint_recall` of the confident keypoints fall
                      inside the mask (a missed limb, hand or foot); the report names them
  mask_unstable       mask IoU with the previous frame below `min_mask_iou` while the box
                      IoU is above 0.7 (the mask changed, the person did not)

Each group is switched on separately. Everything measured is always reported and plotted; a
failed check of an enabled group stops the workflow, since sampling on a wrong mask or pose is
wasted. Thresholds are starting points: run with the switches off on clips known to be good
and bad and read `metrics` before trusting them.
"""
import json
from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
import torch

from . import log

BODY_NAMES = ["nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
              "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
              "l_foot", "r_foot"]
TORSO = [0, 1, 2, 5, 8, 11]  # nose, neck, shoulders, hips: cannot jump a quarter of the body in one frame
# The limbs the pose images are drawn from, as pairs of body keypoints: the same list as
# human_visualization.draw_aapose_new's limbSeq, zero-based. A limb is drawn when the pose
# model is sure of both its ends, so counting them counts what ends up on screen.
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
         (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (13, 18), (10, 19)]
WARNINGS = {"no_detection", "multi_person", "mask_specks"}
# In the order each group tests them on a frame; the report lists the checks in the order
# they first fired, and this order breaks the tie between two that first fire on one frame.
POSE_CHECKS = ("no_detection", "pose_incomplete", "pose_jump", "subject_switch", "multi_person")
MASK_CHECKS = ("mask_empty", "mask_leak", "mask_fragmented", "mask_specks", "mask_missing_keypoints", "mask_unstable")
BOX_MARGIN = 0.10
# Completeness is measured against the frames within this many either side. A limb that at
# least this share of them draw is one the pipeline can find on this material, so losing it
# is a defect; a limb the whole neighbourhood is missing is the person being framed that way
# (a close-up has no legs) and is not expected of this frame. Both numbers are about how fast
# a shot changes, not about any model: +/-8 frames is about a quarter of a second, and a
# quarter of the window is enough for a limb that is only visible part of the time.
COMPLETENESS_WINDOW = 8
COMPLETENESS_SHARE = 0.25
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
POSE_ROW = ("frame", "detected", "persons", "pose_conf", "confident_keypoints", "drawn_limbs",
            "box_iou_prev", "torso_jump", "pose_completeness", "lost_limbs")
MASK_ROW = ("frame", "mask_area", "mask_to_box", "box_reliable", "mask_outside_box", "fragments",
            "keypoint_recall", "missed_keypoints", "box_iou_prev", "mask_iou_prev")
PREPROCESS_ROW = ("frame", "detected", "persons", "pose_conf", "confident_keypoints", "drawn_limbs",
                  "mask_area", "mask_to_box", "box_reliable", "mask_outside_box", "fragments",
                  "keypoint_recall", "missed_keypoints", "box_iou_prev", "mask_iou_prev", "torso_jump",
                  "pose_completeness", "lost_limbs")


def _threshold(default, low, high, tooltip):
    return field(default=default, metadata={"min": low, "max": high, "step": 0.05, "tooltip": tooltip})


@dataclass
class PoseGuardConfig:
    """Thresholds of the pose checks. Which keypoints count as found is PoseConfig.min_keypoint_conf,
    read from pose_data."""
    min_pose_completeness: float = _threshold(0.6, 0.0, 1.0, "pose_incomplete: the frame draws less than this share of the limbs the frames around it draw")
    max_torso_jump: float = _threshold(0.25, 0.0, 2.0, "pose_jump: torso keypoints moving more than this fraction of the box diagonal in one frame while the box stays")


@dataclass
class MaskGuardConfig:
    """Thresholds of the mask checks. Which keypoints count as found is PoseConfig.min_keypoint_conf,
    read from pose_data."""
    min_mask_to_box: float = _threshold(0.15, 0.0, 1.0, "mask_empty: mask area below this fraction of the box area")
    max_mask_outside_box: float = _threshold(0.10, 0.0, 1.0, "mask_leak: more than this fraction of the mask outside the box grown by 10%")
    min_keypoint_recall: float = _threshold(0.9, 0.0, 1.0, "mask_missing_keypoints: fewer than this fraction of the confident keypoints inside the mask")
    min_mask_iou: float = _threshold(0.6, 0.0, 1.0, "mask_unstable: mask IoU with the previous frame below this while the box IoU is above 0.7")


class GuardFailed(RuntimeError):
    """An enabled check failed; the message is the report."""


def _iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _ranges(frames):
    """[1, 2, 3, 7, 9, 10] -> '1-3, 7, 9-10'"""
    out, start, prev = [], None, None
    for f in frames:
        if start is None:
            start = prev = f
        elif f == prev + 1:
            prev = f
        else:
            out.append(f"{start}-{prev}" if prev > start else str(start))
            start = prev = f
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else str(start))
    return ", ".join(out)


def _pose_inputs(pose_data):
    """The per-frame keypoints and detections of `pose_data`, checked."""
    pose_metas = pose_data.get("pose_metas_original") if isinstance(pose_data, dict) else None
    detections = pose_data.get("detections") if isinstance(pose_data, dict) else None
    if pose_metas is None or detections is None:
        raise ValueError("pose_data has no per-frame keypoints and detections; it must come from Pose Detection "
                         "or WanAnimate Preprocess")
    if len(pose_metas) != len(detections):
        raise ValueError(f"pose_data has {len(pose_metas)} frames of keypoints but {len(detections)} of detections")
    return pose_metas, detections


def _min_keypoint_conf(pose_data):
    """The keypoint confidence the pose was made with: one threshold for every node that reads
    pose_data, so the guards judge the same keypoints the pose images draw."""
    pose_config = pose_data.get("pose_config") if isinstance(pose_data, dict) else None
    if not isinstance(pose_config, dict) or "min_keypoint_conf" not in pose_config:
        raise ValueError("pose_data has no pose_config.min_keypoint_conf; it must come from Pose Detection "
                         "or WanAnimate Preprocess")
    return pose_config["min_keypoint_conf"]


def _mask_inputs(mask, pose_data):
    """The mask as [frames, height, width] booleans and the pose it is checked against, checked
    against each other."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.dim() != 3:
        raise ValueError(f"mask must be [frames, height, width], got a tensor of shape {tuple(mask.shape)}")
    masks = (mask.cpu().numpy() > 0.5)
    pose_metas, detections = _pose_inputs(pose_data)
    N, H, W = masks.shape
    if len(pose_metas) != N:
        raise ValueError(f"mask has {N} frames, pose_data {len(pose_metas)}; connect the outputs of the same clip")
    if pose_metas and (pose_metas[0]["height"], pose_metas[0]["width"]) != (H, W):
        raise ValueError(f"mask is {W}x{H} but the pose was found on {pose_metas[0]['width']}x{pose_metas[0]['height']} "
                         "frames; connect the mask straight from the tracker, before any resize")
    return masks, pose_metas, detections


def _frame_pose(meta, det, W, H, min_keypoint_conf):
    """One frame's box size and diagonal, its body keypoints in pixels and which are confident."""
    x1, y1, x2, y2 = det["bbox"]
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    diag = float(np.hypot(bw, bh))
    kps = np.asarray(meta["keypoints_body"], dtype=np.float64).copy()
    kps[:, 0] *= W
    kps[:, 1] *= H
    confident = kps[:, 2] >= min_keypoint_conf
    return bw, bh, diag, kps, confident


def _box_iou_prev(detections, i):
    """Box IoU with the previous frame, None on the first frame or when either was not detected."""
    if i == 0:
        return None
    det, prev = detections[i], detections[i - 1]
    return _box_iou(det["bbox"], prev["bbox"]) if (det["score"] > 0 and prev["score"] > 0) else None


def box_envelopes(detections, N, W, H):
    """Per frame, the union of the detector boxes within BOX_WINDOW frames either side,
    grown by BOX_MARGIN and clipped to the frame."""
    grown = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        if det["score"] <= 0:
            grown.append(None)
            continue
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        grown.append((max(0.0, x1 - BOX_MARGIN * bw), max(0.0, y1 - BOX_MARGIN * bh),
                      min(float(W), x2 + BOX_MARGIN * bw), min(float(H), y2 + BOX_MARGIN * bh)))
    out = []
    for i in range(N):
        near = [b for b in grown[max(0, i - BOX_WINDOW):i + BOX_WINDOW + 1] if b is not None]
        if not near:
            out.append((0.0, 0.0, float(W), float(H)))
        else:
            out.append((min(b[0] for b in near), min(b[1] for b in near),
                        max(b[2] for b in near), max(b[3] for b in near)))
    return out


def detached_fractions(mask, kps, confident):
    """The regions of `mask` detached from its largest one, as fractions of that one.

    A piece holding the person's own confident keypoints is not a second object: it is a part
    of them the frame edge or a gap in the mask has split off - a hand that leaves the shot and
    comes back at the corner is still her hand. Nothing else is excused. Asking only whether a
    piece touches the frame border is not enough to say the border is what separates it: on a
    clip where the body runs off the bottom of every frame, that excuses anything at any edge,
    including the objects beside her that the decoder took in."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 2:
        return []
    H, W = mask.shape
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = int(np.argmax(areas)) + 1
    xs = np.clip(kps[confident, 0].round().astype(int), 0, W - 1)
    ys = np.clip(kps[confident, 1].round().astype(int), 0, H - 1)
    person = set(labels[ys, xs].tolist()) if len(xs) else set()
    out = [round(float(stats[i, cv2.CC_STAT_AREA] / areas[main - 1]), 4)
           for i in range(1, count) if i != main and i not in person]
    return sorted((f for f in out if f >= SPECK_FRACTION), reverse=True)


def pose_completeness(drawn):
    """Per frame, the share of the limbs its neighbourhood draws that it draws itself, and
    which of them it lost. `drawn` is [N, len(LIMBS)] booleans."""
    N = len(drawn)
    shares, lost = [], []
    for i in range(N):
        window = drawn[max(0, i - COMPLETENESS_WINDOW):i + COMPLETENESS_WINDOW + 1]
        expected = window.mean(axis=0) >= COMPLETENESS_SHARE
        missing = expected & ~drawn[i]
        n = int(expected.sum())
        shares.append(float((drawn[i] & expected).sum() / n) if n else 1.0)
        lost.append([f"{BODY_NAMES[a]}-{BODY_NAMES[b]}" for (a, b), m in zip(LIMBS, missing) if m])
    return shares, lost


def pose_frame_metrics(pose_metas, detections, W, H, min_keypoint_conf):
    """One dict of raw pose measurements per frame (keys POSE_ROW); thresholds are applied
    afterwards. W and H are the size of the frames the pose was found on."""
    N = len(pose_metas)
    rows = []
    drawn = []
    prev = None
    for i in range(N):
        det = detections[i]
        _, _, diag, kps, confident = _frame_pose(pose_metas[i], det, W, H, min_keypoint_conf)
        drawn.append([bool(confident[a] and confident[b]) for a, b in LIMBS])
        m = {"frame": i, "detected": det["score"] > 0, "persons": det["persons"],
             "pose_conf": float(kps[:, 2].mean()), "confident_keypoints": int(confident.sum()),
             "drawn_limbs": int(sum(drawn[-1])), "box_iou_prev": _box_iou_prev(detections, i)}
        # torso motion against the previous frame
        if prev is not None:
            both = confident & prev["confident"]
            torso = [j for j in TORSO if both[j]]
            m["torso_jump"] = float(np.linalg.norm(kps[torso, :2] - prev["kps"][torso, :2], axis=1).max() / diag) if torso else 0.0
        else:
            m["torso_jump"] = 0.0
        rows.append(m)
        prev = {"kps": kps, "confident": confident}
    shares, lost = pose_completeness(np.array(drawn, dtype=bool).reshape(N, len(LIMBS)))
    for m, share, limbs in zip(rows, shares, lost):
        m["pose_completeness"], m["lost_limbs"] = share, limbs
    return rows


def mask_frame_metrics(masks, pose_metas, detections, min_keypoint_conf):
    """One dict of raw mask measurements per frame (keys MASK_ROW); thresholds are applied
    afterwards. `masks` is [N, H, W] booleans on the frames the pose was found on."""
    N, H, W = masks.shape
    envelopes = box_envelopes(detections, N, W, H)
    rows = []
    prev_mask = None
    for i in range(N):
        det = detections[i]
        bw, bh, diag, kps, confident = _frame_pose(pose_metas[i], det, W, H, min_keypoint_conf)
        mask = masks[i]
        area = int(mask.sum())

        m = {"frame": i, "mask_area": area / (H * W), "mask_to_box": area / (bw * bh)}
        m["box_reliable"] = bool(det["score"] > 0 and confident.sum() >= RELIABLE_KEYPOINTS
                                 and (kps[confident, 2].mean() if confident.any() else 0) >= RELIABLE_CONF)

        gx1, gy1, gx2, gy2 = envelopes[i]
        inside = int(mask[int(gy1):int(gy2), int(gx1):int(gx2)].sum())
        m["mask_outside_box"] = (area - inside) / area if area else 0.0

        # detached regions, as fractions of the largest one
        m["fragments"] = detached_fractions(mask, kps, confident) if area else []

        # keypoints inside the (slightly grown) mask
        if area and confident.any():
            k = max(3, int(0.02 * diag) | 1)
            grown = cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8))
            xs = np.clip(kps[:, 0].round().astype(int), 0, W - 1)
            ys = np.clip(kps[:, 1].round().astype(int), 0, H - 1)
            hit = grown[ys, xs].astype(bool) & confident
            m["keypoint_recall"] = float(hit.sum() / confident.sum())
            m["missed_keypoints"] = [BODY_NAMES[j] for j in np.flatnonzero(confident & ~hit)]
        else:
            m["keypoint_recall"] = 0.0 if confident.any() else 1.0
            m["missed_keypoints"] = [BODY_NAMES[j] for j in np.flatnonzero(confident)] if area == 0 else []

        # motion against the previous frame
        m["box_iou_prev"] = _box_iou_prev(detections, i)
        m["mask_iou_prev"] = _iou(mask, prev_mask) if prev_mask is not None else None

        rows.append(m)
        prev_mask = mask
    return rows


def pose_flags(rows, t):
    """The pose checks: check name -> frames it fired on, in the order the checks first fired."""
    flags = {}

    def flag(name, i):
        flags.setdefault(name, []).append(i)

    for m in rows:
        i = m["frame"]
        if not m["detected"]:
            flag("no_detection", i)
        if m["pose_completeness"] < t["min_pose_completeness"]:
            flag("pose_incomplete", i)
        if m["box_iou_prev"] is not None and m["box_iou_prev"] > 0.5 and m["torso_jump"] > t["max_torso_jump"]:
            flag("pose_jump", i)
        if m["box_iou_prev"] is not None and m["box_iou_prev"] < 0.3:
            flag("subject_switch", i)
        if m["persons"] > 1:
            flag("multi_person", i)
    return flags


def mask_flags(rows, t):
    """The mask checks: check name -> frames it fired on, in the order the checks first fired."""
    flags = {}

    def flag(name, i):
        flags.setdefault(name, []).append(i)

    for m in rows:
        i = m["frame"]
        if m["mask_area"] == 0 and m["box_reliable"]:
            # an empty mask on a frame the pose pipeline could not describe is a pose
            # failure, already flagged as no_detection / pose_incomplete
            flag("mask_empty", i)
        elif m["box_reliable"]:
            if m["mask_to_box"] < t["min_mask_to_box"]:
                flag("mask_empty", i)
            elif m["mask_outside_box"] > t["max_mask_outside_box"]:
                flag("mask_leak", i)
        if any(f >= FRAGMENT_FRACTION for f in m["fragments"]):
            flag("mask_fragmented", i)
        elif m["fragments"]:
            flag("mask_specks", i)
        if m["keypoint_recall"] < t["min_keypoint_recall"] and m["mask_area"] > 0:
            flag("mask_missing_keypoints", i)
        if (m["mask_iou_prev"] is not None and m["box_iou_prev"] is not None
                and m["box_iou_prev"] > 0.7 and m["mask_iou_prev"] < t["min_mask_iou"]):
            flag("mask_unstable", i)
    return flags


def longest_run(frames):
    """The longest run of consecutive frame numbers in a sorted list."""
    longest = run = 0
    prev = None
    for f in frames:
        run = run + 1 if prev is not None and f == prev + 1 else 1
        longest = max(longest, run)
        prev = f
    return longest


def write_report(title, rows, flags, enabled):
    """The report text and whether the enabled checks all passed."""
    n = len(rows)
    failed = [name for name in flags if name in enabled and name not in WARNINGS]
    lines = [f"{title}: {'FAILED' if failed else 'passed'} - "
             f"{len(failed)} check(s) failed on {len({i for name in failed for i in flags[name]})}/{n} frames"]
    for name, frames in flags.items():
        kind = "warning" if name in WARNINGS else ("fail" if name in enabled else "off")
        line = f"- {name} ({kind}): {len(frames)} frame(s), longest run {longest_run(frames)}: {_ranges(frames)}"
        key = {"mask_missing_keypoints": "missed_keypoints", "pose_incomplete": "lost_limbs"}.get(name)
        if key:
            missed = {}
            for i in frames:
                for k in rows[i][key]:
                    missed[k] = missed.get(k, 0) + 1
            line += " | missed: " + ", ".join(f"{k} x{c}" for k, c in sorted(missed.items(), key=lambda kv: -kv[1]))
        lines.append(line)
    return "\n".join(lines), not failed


PANEL_W, PANEL_H, MARGIN_L, MARGIN_R, MARGIN_T, GAP = 1200, 190, 210, 20, 34, 34
COLORS = {"blue": (31, 119, 180), "orange": (255, 127, 14), "green": (44, 160, 44), "red": (214, 39, 40), "grey": (150, 150, 150)}
# The timeline panels, as (title, [(legend, row key, colour)]). A series keeps its colour in
# every timeline it appears in.
POSE_PANELS = [
    ("pose", [("mean keypoint confidence", "pose_conf", "orange"), ("limbs vs neighbours", "pose_completeness", "green")]),
    ("motion", [("box IoU vs previous", "box_iou_prev", "orange"), ("torso jump / box diagonal", "torso_jump", "green")]),
]
MASK_PANELS = [
    ("mask", [("mask / box area", "mask_to_box", "blue"), ("mask outside box", "mask_outside_box", "orange")]),
    ("mask vs pose", [("keypoints inside mask", "keypoint_recall", "blue")]),
    ("motion", [("mask IoU vs previous", "mask_iou_prev", "blue"), ("box IoU vs previous", "box_iou_prev", "orange")]),
]
PREPROCESS_PANELS = [
    ("mask", [("mask / box area", "mask_to_box", "blue"), ("mask outside box", "mask_outside_box", "orange")]),
    ("pose", [("keypoints inside mask", "keypoint_recall", "blue"), ("mean keypoint confidence", "pose_conf", "orange"),
              ("limbs vs neighbours", "pose_completeness", "green")]),
    ("motion", [("mask IoU vs previous", "mask_iou_prev", "blue"), ("box IoU vs previous", "box_iou_prev", "orange"),
                ("torso jump / box diagonal", "torso_jump", "green")]),
]


def _polyline(img, values, x0, y0, w, h, color):
    pts = [(int(x0 + i / max(len(values) - 1, 1) * w), int(y0 + h - min(max(v, 0.0), 1.0) * h))
           for i, v in enumerate(values) if v is not None]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 1, cv2.LINE_AA)


def _panel(img, y0, title, series):
    """One panel: a 0..1 axis with gridlines and the named series drawn over it."""
    x0, w, h = MARGIN_L, PANEL_W - MARGIN_L - MARGIN_R, PANEL_H - GAP
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), COLORS["grey"], 1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(y0 + h - frac * h)
        if 0 < frac < 1:
            cv2.line(img, (x0, y), (x0 + w, y), (225, 225, 225), 1)
        cv2.putText(img, f"{frac:.2f}", (x0 - 38, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS["grey"], 1, cv2.LINE_AA)
    # title and legend on the line above the panel
    cv2.putText(img, title, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    legend_x = x0 + 90
    for name, values, color in series:
        _polyline(img, values, x0, y0, w, h, color)
        cv2.line(img, (legend_x, y0 - 14), (legend_x + 18, y0 - 14), color, 2)
        cv2.putText(img, name, (legend_x + 24, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        legend_x += 8 * len(name) + 56


def timeline_image(rows, flags, panels):
    """Metrics over frames with the flagged frames marked, as an IMAGE tensor (no plotting
    library needed, drawn with OpenCV). `panels` is one of the *_PANELS lists."""
    n = len(rows)
    names = list(flags) or ["(no flags)"]
    flag_h = 24 * len(names) + 40
    height = MARGIN_T + len(panels) * PANEL_H + flag_h + 30
    img = np.full((height, PANEL_W, 3), 255, np.uint8)
    y = MARGIN_T
    for title, series in panels:
        _panel(img, y, title, [(name, [m[key] for m in rows], COLORS[color]) for name, key, color in series])
        y += PANEL_H
    x0, w = MARGIN_L, PANEL_W - MARGIN_L - MARGIN_R
    cv2.putText(img, "flags", (x0, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x0, y), (x0 + w, y + flag_h - 30), COLORS["grey"], 1)
    for row, name in enumerate(names):
        cy = y + 16 + 24 * row
        cv2.putText(img, name, (12, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1, cv2.LINE_AA)
        color = COLORS["orange"] if name in WARNINGS else COLORS["red"]
        for f in flags.get(name, []):
            cx = int(x0 + f / max(n - 1, 1) * w)
            cv2.rectangle(img, (cx - 2, cy - 4), (cx + 2, cy + 4), color, -1)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        cx = int(x0 + frac * w)
        cv2.putText(img, str(int(frac * (n - 1))), (cx - 8, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLORS["grey"], 1, cv2.LINE_AA)
    cv2.putText(img, "frame", (PANEL_W // 2 - 20, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1, cv2.LINE_AA)
    return torch.from_numpy(img).float().unsqueeze(0) / 255.0


def _finish(title, guard, rows, flags, thresholds, enabled, panels, stop_on_fail):
    """Report, metrics and timeline of one guard run; stops the workflow on a failed enabled
    check when `stop_on_fail`. `guard` names the group in the metrics (None for the combined
    run, whose metrics keep the layout they always had)."""
    report, passed = write_report(title, rows, flags, enabled)
    record = {"guard": guard} if guard else {}
    record.update({"thresholds": thresholds, "enabled": sorted(enabled), "flags": flags, "frames": rows})
    metrics = json.dumps(record)
    timeline = timeline_image(rows, flags, panels)
    if stop_on_fail:
        # whoever decides whether the workflow stops is the one that logs the report
        log.info(report.replace("\n", "\n    "))
        if not passed:
            raise GuardFailed(report)
    return report, metrics, timeline


def _config(config, cls):
    if config is None:
        return cls()
    if not isinstance(config, cls):
        raise TypeError(f"expected a {cls.__name__}, got {type(config).__name__}")
    return config


def check_pose(pose_data, config=None, enabled=True, stop_on_fail=True):
    """The pose checks on `pose_data` alone.

    `config` is a PoseGuardConfig (None = defaults); the keypoint confidence threshold is the
    one the pose was made with, from `pose_data["pose_config"]`. `enabled` False still measures
    and reports every check but marks them off, so none can fail. With `stop_on_fail` a failed
    enabled check raises GuardFailed with the report; the wrapper that combines both groups
    passes False and lets `combine_guards` decide.

    Returns (pose_data unchanged, report, metrics JSON, timeline IMAGE)."""
    config = _config(config, PoseGuardConfig)
    pose_metas, detections = _pose_inputs(pose_data)
    W, H = (pose_metas[0]["width"], pose_metas[0]["height"]) if pose_metas else (0, 0)
    thresholds = {"min_keypoint_conf": _min_keypoint_conf(pose_data), **asdict(config)}
    with log.step(f"pose guard: checking {len(pose_metas)} frames ({'on' if enabled else 'off'})"):
        rows = pose_frame_metrics(pose_metas, detections, W, H, thresholds["min_keypoint_conf"])
        flags = pose_flags(rows, thresholds)
    report, metrics, timeline = _finish("Pose guard", "pose", rows, flags, thresholds,
                                        set(POSE_CHECKS if enabled else ()), POSE_PANELS, stop_on_fail)
    return pose_data, report, metrics, timeline


def check_mask(mask, pose_data, config=None, enabled=True, stop_on_fail=True):
    """The mask checks on `mask` [frames, H, W] (or one [H, W] frame) against the keypoints and
    boxes in `pose_data`, which must be of the same frames at the same size.

    `config` is a MaskGuardConfig (None = defaults); `enabled` and `stop_on_fail` as in
    `check_pose`. Returns (mask unchanged, report, metrics JSON, timeline IMAGE)."""
    config = _config(config, MaskGuardConfig)
    masks, pose_metas, detections = _mask_inputs(mask, pose_data)
    thresholds = {"min_keypoint_conf": _min_keypoint_conf(pose_data), **asdict(config)}
    with log.step(f"mask guard: checking {len(masks)} frames ({'on' if enabled else 'off'})"):
        rows = mask_frame_metrics(masks, pose_metas, detections, thresholds["min_keypoint_conf"])
        flags = mask_flags(rows, thresholds)
    report, metrics, timeline = _finish("Mask guard", "mask", rows, flags, thresholds,
                                        set(MASK_CHECKS if enabled else ()), MASK_PANELS, stop_on_fail)
    return mask, report, metrics, timeline


def combine_guards(pose_metrics, mask_metrics, stop_on_fail=True):
    """The one report of the whole preprocess from the `metrics` of `check_pose` and
    `check_mask` on the same clip: every frame's measurements side by side, the checks of both
    groups in the order they first fired, one timeline. With `stop_on_fail` a failed enabled
    check of either group raises GuardFailed with the combined report.

    Returns (report, metrics JSON, timeline IMAGE)."""
    pose, mask = json.loads(pose_metrics), json.loads(mask_metrics)
    if pose.get("guard") != "pose" or mask.get("guard") != "mask":
        raise ValueError(f"expected the metrics of check_pose and check_mask, got the metrics of "
                         f"{pose.get('guard')!r} and {mask.get('guard')!r}")
    if len(pose["frames"]) != len(mask["frames"]):
        raise ValueError(f"the pose was checked on {len(pose['frames'])} frames, the mask on {len(mask['frames'])}; "
                         "check the same clip")
    rows = []
    for p, m in zip(pose["frames"], mask["frames"]):
        for key in set(p) & set(m):
            if p[key] != m[key]:
                raise ValueError(f"frame {p['frame']}: the pose and the mask checks disagree on {key} "
                                 f"({p[key]} and {m[key]}); check the same clip with the same pose_data")
        merged = {**p, **m}
        rows.append({key: merged[key] for key in PREPROCESS_ROW})
    order = POSE_CHECKS + MASK_CHECKS
    both = {**pose["flags"], **mask["flags"]}
    flags = {name: both[name] for name in sorted(both, key=lambda name: (both[name][0], order.index(name)))}
    thresholds = {**pose["thresholds"], **mask["thresholds"]}
    enabled = set(pose["enabled"]) | set(mask["enabled"])
    return _finish("Preprocess guard", None, rows, flags, thresholds, enabled, PREPROCESS_PANELS, stop_on_fail)
