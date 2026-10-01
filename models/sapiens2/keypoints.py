"""Sapiens2's keypoints: its 308-keypoint output size, the names of keypoints 0-69 (body, feet, hands,
arm and neck extras; the rest are face), and the 70 -> 133 table into COCO-WholeBody, the layout the
pack's pose drawing reads (libs/pose_utils/pose2d_utils.py split_kp2ds_for_aa: body 0-22, face 22:91,
left hand 91:112, right hand 112:133). The table covers everything but the face: Pose Detection with a
Sapiens2 pose_model takes the 68 face points from ViTPose-H (pipelines/sapiens2_pose.py).

Sources (fetched 2026-10-01):
- SAPIENS2_70_NAMES: facebookresearch/sapiens2 sapiens/pose/configs/_base_/keypoints308.py
  (`dataset_info`, the "goliath" 308 keypoints) ids 0-69. SAM 3D Body's MHR70
  (facebookresearch/sam-3d-body sam_3d_body/metadata/mhr70.py) is the same 70 names, name for name,
  so the table serves both.
- COCO_WHOLEBODY_NAMES: the COCO-WholeBody definition as mmpose ships it
  (configs/_base_/datasets/coco_wholebody.py), copied verbatim into the same sapiens2 file as
  `coco_wholebody_info`, ids 0-132.

Numpy only.
"""
import numpy as np

NUM_KEYPOINTS = 308

SAPIENS2_70_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",                               # 0-4
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",                         # 5-8
    "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",        # 9-14
    "left_big_toe", "left_small_toe", "left_heel",                                          # 15-17
    "right_big_toe", "right_small_toe", "right_heel",                                       # 18-20
    # right hand 21-41: per finger tip first (4 = tip ... third_joint = the finger's base), wrist last
    "right_thumb4", "right_thumb3", "right_thumb2", "right_thumb_third_joint",
    "right_forefinger4", "right_forefinger3", "right_forefinger2", "right_forefinger_third_joint",
    "right_middle_finger4", "right_middle_finger3", "right_middle_finger2", "right_middle_finger_third_joint",
    "right_ring_finger4", "right_ring_finger3", "right_ring_finger2", "right_ring_finger_third_joint",
    "right_pinky_finger4", "right_pinky_finger3", "right_pinky_finger2", "right_pinky_finger_third_joint",
    "right_wrist",                                                                          # 41
    # left hand 42-62, same order
    "left_thumb4", "left_thumb3", "left_thumb2", "left_thumb_third_joint",
    "left_forefinger4", "left_forefinger3", "left_forefinger2", "left_forefinger_third_joint",
    "left_middle_finger4", "left_middle_finger3", "left_middle_finger2", "left_middle_finger_third_joint",
    "left_ring_finger4", "left_ring_finger3", "left_ring_finger2", "left_ring_finger_third_joint",
    "left_pinky_finger4", "left_pinky_finger3", "left_pinky_finger2", "left_pinky_finger_third_joint",
    "left_wrist",                                                                           # 62
    "left_olecranon", "right_olecranon", "left_cubital_fossa", "right_cubital_fossa",       # 63-66
    "left_acromion", "right_acromion", "neck",                                              # 67-69
)

_FINGERS = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")


def _coco_hand(side):
    # per finger base -> tip (1 = base ... 4 = tip), after the hand root
    return [f"{side}_hand_root"] + [f"{side}_{finger}{j}" for finger in _FINGERS for j in (1, 2, 3, 4)]


COCO_WHOLEBODY_NAMES = tuple(
    ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
     "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
     "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle",
     "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]
    + [f"face-{i}" for i in range(68)]
    + _coco_hand("left")
    + _coco_hand("right")
)

# COCO-WholeBody index -> Sapiens2-308 (= MHR70) index. Everything but the face (23-90).
# Wrists: COCO 9 (left_wrist) and 91 (left_hand_root) both come from 62 (left_wrist); COCO 10
# and 112 from 41 (right_wrist). Fingers: Sapiens2 runs tip -> base, COCO base -> tip, so each
# 4-tuple is reversed; COCO <finger>1 (the base) is <finger>_third_joint, <finger>2-4 share the
# name. The same reversal as ComfyUI core's OPENPOSE_HAND21_TO_MHR70_{R,L}
# (comfy_extras/sam3d_body/export/glb_shared.py).
COCO_FROM_SAPIENS2 = {
    # body: 0-8 identical, wrists from the hand wrists, hips..ankles shifted by 2
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    9: 62, 10: 41,
    11: 9, 12: 10, 13: 11, 14: 12, 15: 13, 16: 14,
    # feet
    17: 15, 18: 16, 19: 17, 20: 18, 21: 19, 22: 20,
    # left hand (COCO 91-111)
    91: 62,
    92: 45, 93: 44, 94: 43, 95: 42,          # thumb
    96: 49, 97: 48, 98: 47, 99: 46,          # forefinger
    100: 53, 101: 52, 102: 51, 103: 50,      # middle
    104: 57, 105: 56, 106: 55, 107: 54,      # ring
    108: 61, 109: 60, 110: 59, 111: 58,      # pinky
    # right hand (COCO 112-132)
    112: 41,
    113: 24, 114: 23, 115: 22, 116: 21,
    117: 28, 118: 27, 119: 26, 120: 25,
    121: 32, 122: 31, 123: 30, 124: 29,
    125: 36, 126: 35, 127: 34, 128: 33,
    129: 40, 130: 39, 131: 38, 132: 37,
}
COCO_INDEX = np.array(sorted(COCO_FROM_SAPIENS2), dtype=np.int64)
SAPIENS2_INDEX = np.array([COCO_FROM_SAPIENS2[i] for i in COCO_INDEX], dtype=np.int64)
FACE = slice(23, 91)


def to_coco133(kp70):
    """[N, >=70, 3] (x, y, conf) in Sapiens2-308 (= MHR70) order -> [N, 133, 3] COCO-WholeBody. The
    face rows 23-90 (FACE) are left at 0, confidence 0."""
    kp70 = np.asarray(kp70, dtype=np.float32)
    if kp70.ndim != 3 or kp70.shape[1] < 70 or kp70.shape[2] != 3:
        raise ValueError(f"expected [N, >=70, 3] keypoints, got {kp70.shape}")
    out = np.zeros((kp70.shape[0], 133, 3), dtype=np.float32)
    out[:, COCO_INDEX] = kp70[:, SAPIENS2_INDEX]
    return out
