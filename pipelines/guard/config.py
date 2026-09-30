"""The thresholds of the two groups and of the SCAIL-2 guard, as config dataclasses (the guard
nodes build their widgets from them), and the config a check runs with."""
from dataclasses import dataclass, field

# pose_spike's jump, the Pose Guard's default.
LIMB_SPIKE = 0.08


def _threshold(default, low, high, tooltip, step=0.05):
    return field(default=default, metadata={"min": low, "max": high, "step": step, "tooltip": tooltip})


@dataclass
class PoseGuardConfig:
    """Thresholds of the pose checks, all warnings. Which keypoints count is the draw_threshold the
    pose images were drawn with, read from pose_data."""
    max_torso_jump: float = _threshold(0.25, 0.0, 2.0, "pose_jump (warning): torso keypoints moving more than this fraction of the box diagonal in one frame while the box stays")
    max_limb_spike: float = _threshold(LIMB_SPIKE, 0.0, 1.0, "pose_spike (warning): a limb keypoint jumping more than this fraction of the frame height and coming back within 3 frames", 0.01)


@dataclass
class MaskGuardConfig:
    """Thresholds of the mask checks every guard of a mask runs: the Mask Guard, the WanAnimate
    Preprocess Guard and the SCAIL-2 guard on its driving mask. Which keypoints count is the
    draw_threshold the pose images were drawn with, read from pose_data. mask_loss, mask_limb_out
    and mask_fragmented have no threshold: what the model loses is a whole block or cell of what
    it reads, a limb is prompt_pose's rule, a piece is torn off at 5% of her."""
    min_mask_to_box: float = _threshold(0.15, 0.0, 1.0, "mask_empty (fail): mask area below this fraction of the box area, on a frame whose box the detector and the pose model agree on")
    max_mask_outside_box: float = _threshold(0.10, 0.0, 1.0, "mask_leak (warning): more than this fraction of the mask outside the box grown by 10%")
    max_attached_leak: float = _threshold(0.03, 0.0, 1.0, "mask_attached_leak (warning): the person's mask grown by more than this fraction of its size where neither the neighbouring frames' masks nor the drawn skeleton are", 0.01)
    head_out_eyes_ears: int = _threshold(2, 1, 4, "mask_head_out (fail): the head outside the mask - the drawn nose, or at least this many of the drawn eyes and ears, inside the frame and outside the (slightly grown) mask. The nose outside always fails. A head the mask leaves out is redrawn by the model from nothing. 1: a single eye or ear outside fails too, also where the mask's outline at the hair leaves one out; 4: only all four, or the nose. A face the pose model draws on the back of a head lies on the head, which the mask covers. Needs pose_data; a frame whose mask is empty is mask_empty's. The default 2 is set on the test clips: a correct (prompt) mask left no head keypoint out on any frame, and on every frame where the keypoint mask lost the head, the nose or two or more of the eyes and ears were out.", 1)


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
