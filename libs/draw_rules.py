"""The draw rules: parts of a frame's pose left out of the pose image, never out of the pose data.

ViTPose places a hand, an arm or a wrist it cannot see on a visible part that looks like it: the
hidden hand on the other hand, the hidden arm along the other arm, an out-of-shot wrist on the leg
or the frame edge. It also invents a nose and eyes on the back of the head. The rules below
recognise these from the keypoints alone and name the parts not to draw. They are Pose Config
values, off by default; with every rule off nothing is left out.

The rules, and limb_dedup's two tests, read the model's keypoints (pose_data's
`pose_metas_original`, in the AAPose layout of libs/keypoints.py) at the draw threshold,
independently of each other: a part any enabled rule names is not drawn. The keypoints themselves
are never changed, so the guards, SAM 3.1 Multiplex and the face crop see what the model gave.

- limb_dedup (PoseConfig.limb_dedup): a hand or an arm drawn on its visible twin, by two tests.
  The hand test (duplicate_hands): two drawn hands whose matching keypoints nearly coincide are
  one hand drawn twice. When exactly one arm is intact (its elbow and wrist both drawn), the hand
  of the broken arm is the copy and is left out, if that arm is broken on the frame before or the
  frame after as well: a one-frame dip of an otherwise intact arm is no hidden arm. The arm test
  (mirrored_arms): both elbows and both wrists within MIRROR_DISTANCE body scales of each other,
  with one arm's forearm drawn and something of the other arm drawn, are one arm drawn twice. The
  arm whose elbow and wrist are less confident is the copy: its elbow, wrist and hand are left out.
- forearm_rule: a forearm is never drawn longer than its full length, and the median of its drawn
  length over the clip is at most that. A drawn forearm longer than the limit
  (PoseConfig.forearm_limit) times that median ends in a wrist put on something else (the leg,
  the frame edge): its wrist and hand are left out.
- back_view_face (PoseConfig.back_view_face): a face drawn on the back of the head. On a frame
  whose body is seen from behind (the model's left shoulder on the image left of its right one)
  and whose face is not seen (its jaw line's mean confidence under JAW_CUT), the nose and both eyes
  are left out: the model invents them there, up to 0.99 confident, and the profile drawn from
  them flips side. The ears stay and keep the head's place. Each frame is read alone.

Numpy only: the offline analysis reads it without ComfyUI or cv2.
"""
from typing import TypedDict

import numpy as np

from .keypoints import (L_ELBOW, L_EYE, L_HIP, L_SHOULDER, L_WRIST, NECK, NOSE, R_ELBOW, R_EYE, R_HIP, R_SHOULDER,
                        R_WRIST)

RULES = ("limb_dedup", "forearm_rule", "back_view_face")
# Each side's elbow and wrist keypoint.
SIDES = {"right": (R_ELBOW, R_WRIST), "left": (L_ELBOW, L_WRIST)}
# limb_dedup's hand test: the two hands are one when the median distance between their matching
# keypoints, over the DUP_MIN_SHARED or more drawn in both, is under DUP_RATIO of the larger drawn
# hand's diagonal. Real hands held together sit 0.14-0.50 apart, duplicates 0.07-0.20; the arm test,
# not the ratio, tells them apart.
DUP_RATIO = 0.6
DUP_MIN_SHARED = 3
# limb_dedup's arm test: the elbows and the wrists each closer than this many body scales (the
# widest of the shoulders, the hips and 1.5 x neck-to-nose, the model's positions whether drawn or
# not).
MIRROR_DISTANCE = 0.2
# back_view_face: the face is not seen when its jaw line's mean confidence is under this. Behind the
# head the skull hides the jaw line, and the model invents it less than the nose and eyes. Measured
# on bodies seen from behind: the backs of heads at most 0.808, the faces over the shoulder from 0.865.
JAW_CUT = 0.835
# The jaw line in keypoints_face: rows 1-17, COCO-WholeBody 23-39 (the 68-point face's 0-16).
# Row 0 is COCO-WholeBody 22, the right heel (pose2d_utils.split_kp2ds_for_aa takes 22-90).
JAW = slice(1, 18)
# What back_view_face leaves out: the nose and both eyes.
FACE = (NOSE, R_EYE, L_EYE)
# What a hidden keypoint's confidence is set to for drawing: below any draw threshold, 0 included.
HIDDEN = -np.inf


class Hidden(TypedDict):          # one frame: what the enabled rules leave out of its pose image
    body: list[int]                   # body keypoints (AAPose layout), ascending
    hands: list[str]                  # "left" / "right", ascending
    rules: dict[str, list[str]]       # rule -> where it fired, in RULES order: the sides, "head" for back_view_face


class Overlong(TypedDict):        # one side forearm_rule fired on, over the clip
    median: float                     # the forearm's median drawn length over the clip, px
    ratios: dict[int, float]          # frame -> its drawn length / median, on the frames over the limit


def _pixels(meta, key):
    """The keypoint set `key` of one pose meta as [K, 3] float64 rows: x and y in pixels, the confidence."""
    return np.asarray(meta[key], dtype=np.float64).reshape(-1, 3) * np.array([meta["width"], meta["height"], 1.0])


def _distance(points, a, b):
    return float(np.hypot(*(points[a, :2] - points[b, :2])))


def _drawn_diagonal(hand, threshold):
    """The diagonal of the box around the hand's drawn keypoints (at least one of them)."""
    drawn = hand[hand[:, 2] >= threshold, :2]
    return float(np.hypot(*(drawn.max(0) - drawn.min(0))))


def _hands_coincide(meta, threshold):
    """limb_dedup's hand test on one frame: whether its two drawn hands are one hand drawn twice."""
    left, right = _pixels(meta, "keypoints_left_hand"), _pixels(meta, "keypoints_right_hand")
    shared = (left[:, 2] >= threshold) & (right[:, 2] >= threshold)
    if shared.sum() < DUP_MIN_SHARED:
        return False
    diagonal = max(_drawn_diagonal(left, threshold), _drawn_diagonal(right, threshold), 1.0)
    return np.median(np.hypot(*(left[shared, :2] - right[shared, :2]).T)) / diagonal < DUP_RATIO


def duplicate_hands(pose_metas, threshold) -> dict[str, list[int]]:
    """limb_dedup's hand test over the clip: the sides (in SIDES order) whose hand is the other hand
    drawn again on some frame of `pose_metas`, with those frames, ascending. On such a frame the two
    hands coincide, the other arm is intact (its elbow and wrist drawn) and this side's arm is not,
    and this side's arm is not intact on the frame before or the frame after either (of those that
    exist, so a one-frame clip has none)."""
    intact = []
    for meta in pose_metas:
        drawn = _pixels(meta, "keypoints_body")[:, 2] >= threshold
        intact.append({side: bool(drawn[elbow] and drawn[wrist]) for side, (elbow, wrist) in SIDES.items()})
    frames = {side: [] for side in SIDES}
    for i, (meta, arms) in enumerate(zip(pose_metas, intact)):
        if arms["left"] == arms["right"]:
            continue
        copy = "left" if arms["right"] else "right"
        if any(not intact[j][copy] for j in (i - 1, i + 1) if 0 <= j < len(intact)) and _hands_coincide(meta, threshold):
            frames[copy].append(i)
    return {side: fired for side, fired in frames.items() if fired}


def mirrored_arms(pose_metas, threshold) -> dict[str, list[int]]:
    """limb_dedup's arm test over the clip: the sides (in SIDES order) whose arm is the other arm
    drawn again on some frame of `pose_metas`, with those frames, ascending. On such a frame both
    elbows and both wrists are closer than MIRROR_DISTANCE body scales, one arm's forearm (its elbow
    and wrist) is drawn and something of the other arm is, and this side's elbow and wrist are less
    confident together than the other side's (on equal confidence neither is the copy)."""
    frames = {side: [] for side in SIDES}
    for i, meta in enumerate(pose_metas):
        body = _pixels(meta, "keypoints_body")
        scale = max(_distance(body, R_SHOULDER, L_SHOULDER), _distance(body, R_HIP, L_HIP),
                    1.5 * _distance(body, NECK, NOSE), 1.0)
        reach = MIRROR_DISTANCE * scale
        if _distance(body, R_ELBOW, L_ELBOW) >= reach or _distance(body, R_WRIST, L_WRIST) >= reach:
            continue
        drawn = body[:, 2] >= threshold
        forearm = {side: drawn[elbow] and drawn[wrist] for side, (elbow, wrist) in SIDES.items()}
        some = {side: drawn[elbow] or drawn[wrist] for side, (elbow, wrist) in SIDES.items()}
        if not (forearm["right"] and some["left"] or forearm["left"] and some["right"]):
            continue
        conf = {side: body[elbow, 2] + body[wrist, 2] for side, (elbow, wrist) in SIDES.items()}
        if conf["right"] != conf["left"]:
            frames["right" if conf["right"] < conf["left"] else "left"].append(i)
    return {side: fired for side, fired in frames.items() if fired}


def overlong_forearms(pose_metas, threshold, limit) -> dict[str, Overlong]:
    """forearm_rule over the clip: the sides (in SIDES order) whose drawn forearm is longer than
    `limit` times that forearm's median drawn length over all of `pose_metas` on some frame, with
    that median and those frames (pass the whole clip: the median of a part of it is another
    reference). Nothing at limit 0, the rule off."""
    out = {}
    if limit <= 0:
        return out
    for side, (elbow, wrist) in SIDES.items():
        rows = [_pixels(meta, "keypoints_body")[[elbow, wrist]] for meta in pose_metas]
        length = np.array([_distance(r, 0, 1) for r in rows])
        drawn = np.array([r[0, 2] >= threshold and r[1, 2] >= threshold for r in rows], dtype=bool)
        if not drawn.any():
            continue
        median = float(np.median(length[drawn]))
        over = np.flatnonzero(drawn & (length > limit * median))
        if over.size:
            out[side] = {"median": median, "ratios": {int(i): float(length[i] / median) for i in over}}
    return out


def back_view_faces(pose_metas, threshold) -> list[int]:
    """back_view_face over the clip: the frames of `pose_metas`, ascending, whose nose or an eye is
    drawn on a body seen from behind with its face not seen. Seen from behind: the model's left
    shoulder lies on the image left of its right shoulder (their positions whether drawn or not).
    Face not seen: the mean confidence of the jaw line (JAW) is under JAW_CUT. Each frame is read
    alone; a frame whose nose and eyes are not drawn has nothing to leave out."""
    frames = []
    for i, meta in enumerate(pose_metas):
        body = _pixels(meta, "keypoints_body")
        if (body[L_SHOULDER, 0] < body[R_SHOULDER, 0] and _pixels(meta, "keypoints_face")[JAW, 2].mean() < JAW_CUT
                and (body[list(FACE), 2] >= threshold).any()):
            frames.append(i)
    return frames


def hidden_parts(frame_count, duplicates: dict[str, list[int]], mirrored: dict[str, list[int]],
                 overlong: dict[str, Overlong], back_view: list[int]) -> list[Hidden]:
    """Per frame of a clip of `frame_count` frames, what the enabled rules leave out of its pose
    image: limb_dedup's hand of each side `duplicates` (duplicate_hands) names on it and elbow,
    wrist and hand of each side `mirrored` (mirrored_arms) names on it, forearm_rule's wrist and
    hand of each side `overlong` (overlong_forearms) names on it, and back_view_face's nose and eyes
    when `back_view` (back_view_faces) holds it. Nothing on any frame when all four are empty."""
    out = []
    for i in range(frame_count):
        arms = [side for side, frames in mirrored.items() if i in frames]
        head = i in back_view
        named = {"limb_dedup": [side for side in SIDES if side in arms or i in duplicates.get(side, ())],
                 "forearm_rule": [side for side, over in overlong.items() if i in over["ratios"]],
                 "back_view_face": ["head"] if head else []}
        fired = {rule: named[rule] for rule in RULES if named[rule]}
        body = {SIDES[side][1] for side in named["forearm_rule"]}
        body |= {keypoint for side in arms for keypoint in SIDES[side]}
        body |= set(FACE) if head else set()
        out.append({"body": sorted(body), "hands": sorted({*named["limb_dedup"], *named["forearm_rule"]}),
                    "rules": fired})
    return out
