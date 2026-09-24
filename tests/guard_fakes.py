"""The synthetic guard clip shared by the guard and node tests: a person-sized rectangle
drifting across the frame with keypoints inside it, the measured guard thresholds, and the
injector that makes body keypoints unconfident. The tests bind the guard package itself.
"""
import numpy as np
import torch

from bcvideonodes.pipelines import guard

N, H, W = 40, 320, 240
POSE_CONFIG = {"min_keypoint_conf": 0.3}   # Pose Detection's config; the guards do not read it
DRAW_THRESHOLD = 0.5                       # what the guards count: the keypoints the pose images draw
POSE = guard.PoseGuardConfig(min_pose_completeness=0.6, max_torso_jump=0.25, max_limb_spike=0.08)
MASK = guard.MaskGuardConfig(min_mask_to_box=0.15, max_mask_outside_box=0.10, max_attached_leak=0.03,
                             min_keypoint_recall=0.9, max_body_not_drawn=0.25, min_mask_iou=0.6)

LEGS = [8, 9, 10, 11, 12, 13, 18, 19]   # the keypoints of both legs and both feet


def clip():
    """A person-sized rectangle drifting slowly across the frame, with keypoints inside it."""
    masks = torch.zeros(N, H, W)
    metas, detections = [], []
    for i in range(N):
        x1, y1 = 60 + i, 40
        x2, y2 = x1 + 100, y1 + 240
        masks[i, y1:y2, x1:x2] = 1.0
        # 20 body keypoints spread inside the rectangle, all confident
        xs = np.linspace(x1 + 10, x2 - 10, 5)
        ys = np.linspace(y1 + 10, y2 - 10, 4)
        pts = np.array([(x, y, 0.9) for y in ys for x in xs])
        pts[:, 0] /= W
        pts[:, 1] /= H
        metas.append({"width": W, "height": H, "keypoints_body": pts})
        detections.append({"bbox": [float(x1), float(y1), float(x2), float(y2)], "score": 0.95, "persons": 1})
    return masks, {"pose_metas_original": metas, "detections": detections, "pose_config": dict(POSE_CONFIG),
                   "draw_threshold": DRAW_THRESHOLD}


def drop_keypoints(pose_data, frames, indices, conf=0.05):
    """Make the named body keypoints unconfident on `frames`, as a pose model losing them."""
    for i in frames:
        pts = pose_data["pose_metas_original"][i]["keypoints_body"].copy()
        pts[indices, 2] = conf
        pose_data["pose_metas_original"][i]["keypoints_body"] = pts
