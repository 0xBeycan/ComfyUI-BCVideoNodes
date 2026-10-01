"""The synthetic guard clip shared by the guard and node tests: a person-sized rectangle
drifting across the frame with a frontal skeleton inside it, the measured guard thresholds, and
the injector that makes body keypoints unconfident. The tests bind the guard package itself.

Also the final mask the guards judge and WanAnimate Preprocess outputs: the `final` Names
(libs/mask.py) and the final mask and the painted frames written out from their definitions
(`grown_and_blockified`, `painted`), the references the pack's are held to.
"""
import numpy as np
import torch

from bcvideonodes.pipelines import guard
from names import Names, refs

final = Names("final", {
    **refs("libs.mask", "GROW", "BLOCK_SIZE", "FinalMaskConfig", "final_mask", "painted_black"),
})

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


def grown_and_blockified(mask, grow, block_size):
    """The final mask of the [N, H, W] MASK `mask` written out from its definition, frame by frame
    over the whole frame: MaskGrow's 8-bit quantisation (clip(255 x value) as uint8) and `grow`
    dilations of the 3 x 3 cross, then BlockifyMask over the grown pixels' box - side // block_size
    blocks (at least one) of side // blocks px, the last taking the remainder, each block 1 when it
    holds a grown pixel; block_size 0 cuts no blocks, the grown pixels are the final."""
    import cv2

    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    out = torch.zeros(mask.shape)
    for f, frame in enumerate(mask.numpy()):
        levels = np.clip(255.0 * frame, 0, 255).astype(np.uint8)
        grown = (cv2.dilate(levels, cross, iterations=grow) if grow else levels) > 0
        if not block_size:
            out[f] = torch.from_numpy(grown).float()
            continue
        ys, xs = np.nonzero(grown)
        if not len(ys):
            continue

        def cuts(start, stop):
            blocks = max(1, (stop - start) // block_size)
            size = (stop - start) // blocks
            return [(start + i * size, start + (i + 1) * size if i < blocks - 1 else stop) for i in range(blocks)]

        for a, b in cuts(ys.min(), ys.max() + 1):
            for c, d in cuts(xs.min(), xs.max() + 1):
                if grown[a:b, c:d].any():
                    out[f, a:b, c:d] = 1.0
    return out


def painted(images, mask):
    """Draw Mask On Image's blend written out for the colour 0, 0, 0 at full opacity:
    image x (1 - mask) + 0 x mask."""
    m = mask.unsqueeze(-1)
    return images * (1 - m) + torch.zeros(3) * m
