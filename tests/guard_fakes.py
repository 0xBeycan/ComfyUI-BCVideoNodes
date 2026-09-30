"""The synthetic guard clip shared by the guard and node tests: a person-sized rectangle
drifting across the frame with a frontal skeleton inside it, the measured guard thresholds, and
the injector that makes body keypoints unconfident. The tests bind the guard package itself.
"""
import numpy as np
import torch

from bcvideonodes.pipelines import guard

N, H, W = 40, 320, 240
POSE_CONFIG = {"min_keypoint_conf": 0.3}   # Pose Detection's config; the guards do not read it
DRAW_THRESHOLD = 0.5                       # what the guards count: the keypoints the pose images draw
POSE = guard.PoseGuardConfig(max_torso_jump=0.25, max_limb_spike=0.08)
MASK = guard.MaskGuardConfig(min_mask_to_box=0.15, max_mask_outside_box=0.10, max_attached_leak=0.03, head_out_eyes_ears=2)

LEGS = [8, 9, 10, 11, 12, 13, 18, 19]   # the keypoints of both legs and both feet
# A person facing the camera, in pixels inside her 100 x 240 rectangle: her right side on the
# image's left, the shoulders 56 apart (the body scale), the arms hanging clear of the hips.
SKELETON = [(50, 20), (50, 42), (22, 48), (8, 80), (8, 112), (78, 48), (92, 80), (92, 112),   # nose .. l_wrist
            (35, 140), (35, 185), (35, 222), (65, 140), (65, 185), (65, 222),              # hips, knees, ankles
            (45, 15), (55, 15), (38, 18), (62, 18), (70, 234), (30, 234)]                  # eyes, ears, feet


def origin(i):
    """The top-left corner of frame `i`'s rectangle."""
    return 60 + i, 40


def clip():
    """A person-sized rectangle drifting slowly across the frame, with a frontal skeleton inside it."""
    masks = torch.zeros(N, H, W)
    metas, detections = [], []
    for i in range(N):
        x1, y1 = origin(i)
        x2, y2 = x1 + 100, y1 + 240
        masks[i, y1:y2, x1:x2] = 1.0
        pts = np.array([((x1 + x) / W, (y1 + y) / H, 0.9) for x, y in SKELETON])
        metas.append({"width": W, "height": H, "keypoints_body": pts, "keypoints_left_hand": np.zeros((21, 3)),
                      "keypoints_right_hand": np.zeros((21, 3))})
        detections.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "score": 0.95, "persons": 1})
    return masks, {"pose_metas_original": metas, "detections": detections, "pose_config": dict(POSE_CONFIG),
                   "draw_threshold": DRAW_THRESHOLD}


def arm(masks, frames, rows, width=30):
    """An arm `width` px wide held out from the body's right side at `rows` (relative to its top)
    on `frames`."""
    for i in frames:
        x1, y1 = origin(i)
        masks[i, y1 + rows[0]:y1 + rows[1], x1 + 100:x1 + 100 + width] = 1.0


def drop_keypoints(pose_data, frames, indices, conf=0.05):
    """Make the named body keypoints unconfident on `frames`, as a pose model losing them."""
    for i in frames:
        pts = pose_data["pose_metas_original"][i]["keypoints_body"].copy()
        pts[indices, 2] = conf
        pose_data["pose_metas_original"][i]["keypoints_body"] = pts


def place_keypoints(pose_data, frames, index, x, y, conf=None):
    """Put body keypoint `index` at pixel (x, y) of its frame on `frames` (and at `conf`, if
    given): where the pose model places a keypoint it cannot see."""
    for i in frames:
        pts = pose_data["pose_metas_original"][i]["keypoints_body"].copy()
        pts[index, :2] = (x / W, y / H)
        if conf is not None:
            pts[index, 2] = conf
        pose_data["pose_metas_original"][i]["keypoints_body"] = pts


def hand(x, y, conf=0.9):
    """A drawn hand of 21 keypoints around pixel (x, y), as pose_metas_original holds it."""
    return np.array([((x + dx) / W, (y + dy) / H, conf) for dx in (-4, -2, 0, 2, 4) for dy in (-4, 0, 4, 8)] +
                    [(x / W, y / H, conf)])
