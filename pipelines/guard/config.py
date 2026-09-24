"""The thresholds of the two groups, as config dataclasses (the guard nodes build their widgets
from them), and the config a check runs with."""
from dataclasses import dataclass, field


def _threshold(default, low, high, tooltip, step=0.05):
    return field(default=default, metadata={"min": low, "max": high, "step": step, "tooltip": tooltip})


@dataclass
class PoseGuardConfig:
    """Thresholds of the pose checks. Which keypoints count is the draw_threshold the pose
    images were drawn with, read from pose_data."""
    min_pose_completeness: float = _threshold(0.6, 0.0, 1.0, "pose_incomplete: the frame draws less than this share of the limbs the frames around it draw")
    max_torso_jump: float = _threshold(0.25, 0.0, 2.0, "pose_jump: torso keypoints moving more than this fraction of the box diagonal in one frame while the box stays")
    max_limb_spike: float = _threshold(0.08, 0.0, 1.0, "pose_spike: a limb keypoint jumping more than this fraction of the frame height and coming back within 3 frames", 0.01)


@dataclass
class MaskGuardConfig:
    """Thresholds of the mask checks. Which keypoints count is the draw_threshold the pose
    images were drawn with, read from pose_data."""
    min_mask_to_box: float = _threshold(0.15, 0.0, 1.0, "mask_empty: mask area below this fraction of the box area")
    max_mask_outside_box: float = _threshold(0.10, 0.0, 1.0, "mask_leak: more than this fraction of the mask outside the box grown by 10%")
    max_attached_leak: float = _threshold(0.03, 0.0, 1.0, "mask_attached_leak (warning): the person's mask grown by more than this fraction of its size where neither the neighbouring frames' masks nor the drawn skeleton are", 0.01)
    min_keypoint_recall: float = _threshold(0.9, 0.0, 1.0, "mask_missing_keypoints: fewer than this fraction of the drawn keypoints inside the mask")
    max_body_not_drawn: float = _threshold(0.25, 0.0, 1.0, "body_not_drawn: more than this fraction of the person's mask away from the drawn skeleton (a body the pose image does not draw)")
    min_mask_iou: float = _threshold(0.6, 0.0, 1.0, "mask_unstable: mask IoU with the previous frame below this while the box IoU is above 0.7")


def _config(config, cls):
    if config is None:
        return cls()
    if not isinstance(config, cls):
        raise TypeError(f"expected a {cls.__name__}, got {type(config).__name__}")
    return config
