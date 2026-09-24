"""The POSEDATA contract: the keys and value types of the pose_data dict Pose Detection makes and
every pose, guard, SAM 3.1 Multiplex and face function reads.

Annotations only: at runtime pose_data is the plain dict it always was, and nothing converts it.
tests/pipelines/test_pose_data.py holds these keys, in order, to the dicts the pipeline writes.
"""
from typing import TypedDict

import numpy as np


class PoseMeta(TypedDict):          # one frame of pose_metas_original, keys in the order
    width: int                      # load_pose_metas_from_kp2ds_seq writes them
    height: int
    keypoints_body: np.ndarray          # [20, 3] x/W, y/H, conf; float64 (temporal on) or float32 (off)
    keypoints_left_hand: np.ndarray     # [21, 3]
    keypoints_right_hand: np.ndarray    # [21, 3]
    keypoints_face: np.ndarray          # [69, 3], row 0 = COCO-WholeBody keypoint 22


class Detection(TypedDict):         # one frame's person box as everything downstream sees it
    bbox: list[float]                   # 4 python floats (a list, not a tuple)
    score: float                        # -1.0 undetected, 1.0 supplied
    persons: int


# total=False: not every POSEDATA holds every key. detect writes the first five, pose_detection adds
# draw_threshold, and a pose_data saved without its AAPoseMeta objects has no pose_metas.
class PoseData(TypedDict, total=False):   # the POSEDATA dict, keys in the order they are written
    pose_metas: list                    # AAPoseMeta objects (libs/pose_utils, not imported here: cv2)
    pose_metas_original: list[PoseMeta]
    detections: list[Detection]
    keypoint_source: list[list[int]]    # [B][133], COCO-WholeBody layout
    pose_config: dict                   # dataclasses.asdict(PoseConfig), PoseConfig field order
    draw_threshold: float               # added by pose_detection, stored as given (no cast)
