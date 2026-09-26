"""The thresholds of the two groups and of the SCAIL-2 guard, as config dataclasses (the guard
nodes build their widgets from them), and the config a check runs with."""
from dataclasses import dataclass, field

# pose_spike's jump, the Pose Guard's default. The mask checks read it as well: a limb end the
# pose checks call a spike is a pose error, not a limb the mask lost.
LIMB_SPIKE = 0.08


def _threshold(default, low, high, tooltip, step=0.05):
    return field(default=default, metadata={"min": low, "max": high, "step": step, "tooltip": tooltip})


@dataclass
class PoseGuardConfig:
    """Thresholds of the pose checks. Which keypoints count is the draw_threshold the pose
    images were drawn with, read from pose_data."""
    min_pose_completeness: float = _threshold(0.6, 0.0, 1.0, "pose_incomplete (warning): the frame draws less than this share of the limbs the frames around it draw; a limb out of the shot or hidden (behind the body, turned away) is not counted as lost")
    max_torso_jump: float = _threshold(0.25, 0.0, 2.0, "pose_jump: torso keypoints moving more than this fraction of the box diagonal in one frame while the box stays")
    max_limb_spike: float = _threshold(LIMB_SPIKE, 0.0, 1.0, "pose_spike (warning): a limb keypoint jumping more than this fraction of the frame height and coming back within 3 frames", 0.01)


@dataclass
class MaskGuardConfig:
    """Thresholds of the mask checks. Which keypoints count is the draw_threshold the pose
    images were drawn with, read from pose_data."""
    min_mask_to_box: float = _threshold(0.15, 0.0, 1.0, "mask_empty: mask area below this fraction of the box area")
    max_mask_outside_box: float = _threshold(0.10, 0.0, 1.0, "mask_leak: more than this fraction of the mask outside the box grown by 10%")
    max_attached_leak: float = _threshold(0.03, 0.0, 1.0, "mask_attached_leak (warning): the person's mask grown by more than this fraction of its size where neither the neighbouring frames' masks nor the drawn skeleton are", 0.01)
    min_keypoint_recall: float = _threshold(0.9, 0.0, 1.0, "mask_missing_keypoints (warning): fewer than this fraction of the drawn keypoints inside the mask")
    max_body_not_drawn: float = _threshold(0.25, 0.0, 1.0, "body_not_drawn (warning): more than this fraction of the person's mask away from the drawn skeleton (a body the pose image does not draw)")
    min_mask_iou: float = _threshold(0.6, 0.0, 1.0, "mask_unstable (warning): mask IoU with the previous frame below this while the box IoU is above 0.7")
    max_mask_loss: float = _threshold(0.0185, 0.0, 1.0, "mask_loss (warning): the mask drops a region for 1 to 8 frames while holding it on the frames before and after, and no limb moved away to account for it; flagged over the whole run when the region is thicker than this on a single frame, twice this over a longer run, or half this where the drawn skeleton crosses it (with pose_data). Thickness is the radius of the largest disc the dropped region holds, as a fraction of the frame's shorter side: a dropped hand or foot holds one, the slivers a mask's outline jitters by do not. Set on the test clips.", 0.0005)


@dataclass
class SCAIL2GuardConfig:
    """Thresholds of the SCAIL-2 reference checks. A first value, not calibrated on real clips yet."""
    min_reference_iou: float = _threshold(0.4, 0.0, 1.0, "reference_misaligned (warning, replacement mode only): the character on the reference, center-cropped and resized as the core node does, overlaps the person on the first driving frame by less than this IoU (SCAIL-2 expects the reference posed like the first driving frame). Uncalibrated first value.")


def _config(config, cls):
    if config is None:
        return cls()
    if not isinstance(config, cls):
        raise TypeError(f"expected a {cls.__name__}, got {type(config).__name__}")
    return config
