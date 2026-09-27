"""The draw rule: parts of a frame's pose left out of the pose image, never out of the pose data.

ViTPose places a wrist it cannot see on a visible part that looks like one: the leg, the frame
edge. The forearm ending in that wrist is drawn far longer than the arm is. The rule below
recognises this from the keypoints alone and names the parts not to draw. It is a Pose Config
value, 0 (off) by default; with it off nothing is left out.

The rule reads the model's keypoints (pose_data's `pose_metas_original`, in the AAPose layout of
libs/keypoints.py) at the draw threshold. The keypoints themselves are never changed, so the
guards, SAM 3.1 Multiplex and the face crop see what the model gave.

- forearm_rule: a forearm is never drawn longer than its full length, and the median of its drawn
  length over the clip is at most that. A drawn forearm longer than the limit
  (PoseConfig.forearm_limit) times that median ends in a wrist put on something else (the leg,
  the frame edge): its wrist and hand are left out.

Numpy only: the offline analysis reads it without ComfyUI or cv2.
"""
from typing import TypedDict

import numpy as np

from .keypoints import L_ELBOW, L_WRIST, R_ELBOW, R_WRIST

RULES = ("forearm_rule",)
# Each side's elbow and wrist keypoint.
SIDES = {"right": (R_ELBOW, R_WRIST), "left": (L_ELBOW, L_WRIST)}
# What a hidden keypoint's confidence is set to for drawing: below any draw threshold, 0 included.
HIDDEN = -np.inf


class Hidden(TypedDict):          # one frame: what the enabled rule leaves out of its pose image
    body: list[int]                   # body keypoints (AAPose layout), ascending
    hands: list[str]                  # "left" / "right", ascending
    rules: dict[str, list[str]]       # rule -> the sides it fired on, in RULES order


class Overlong(TypedDict):        # one side forearm_rule fired on, over the clip
    median: float                     # the forearm's median drawn length over the clip, px
    ratios: dict[int, float]          # frame -> its drawn length / median, on the frames over the limit


def _pixels(meta, key):
    """The keypoint set `key` of one pose meta as [K, 3] float64 rows: x and y in pixels, the confidence."""
    return np.asarray(meta[key], dtype=np.float64).reshape(-1, 3) * np.array([meta["width"], meta["height"], 1.0])


def _distance(points, a, b):
    return float(np.hypot(*(points[a, :2] - points[b, :2])))


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


def hidden_parts(frame_count, overlong: dict[str, Overlong]) -> list[Hidden]:
    """Per frame of a clip of `frame_count` frames, what the enabled rule leaves out of its pose
    image: the wrist and hand of each side `overlong` (overlong_forearms) names on it. Nothing on
    any frame when `overlong` is empty."""
    out = []
    for i in range(frame_count):
        sides = [side for side, over in overlong.items() if i in over["ratios"]]
        fired = {"forearm_rule": sides} if sides else {}
        out.append({"body": sorted(SIDES[side][1] for side in sides), "hands": sorted(sides), "rules": fired})
    return out
