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
class MaskChecksConfig:
    """Thresholds of the mask checks every guard of a mask runs: the Mask Guard and the WanAnimate
    Preprocess Guard, and the SCAIL-2 guard on its driving mask. Which keypoints count is the
    draw_threshold the pose images were drawn with, read from pose_data."""
    min_mask_to_box: float = _threshold(0.15, 0.0, 1.0, "mask_empty: mask area below this fraction of the box area")
    max_mask_outside_box: float = _threshold(0.10, 0.0, 1.0, "mask_leak: more than this fraction of the mask outside the box grown by 10%")
    max_attached_leak: float = _threshold(0.03, 0.0, 1.0, "mask_attached_leak (warning): the person's mask grown by more than this fraction of its size where neither the neighbouring frames' masks nor the drawn skeleton are", 0.01)
    min_keypoint_recall: float = _threshold(0.9, 0.0, 1.0, "mask_missing_keypoints (warning): fewer than this fraction of the drawn keypoints inside the mask")
    max_body_not_drawn: float = _threshold(0.25, 0.0, 1.0, "body_not_drawn (warning): more than this fraction of the person's mask away from the drawn skeleton (a body the pose image does not draw)")
    min_mask_iou: float = _threshold(0.6, 0.0, 1.0, "mask_unstable (warning): mask IoU with the previous frame below this while the box IoU is above 0.7")
    max_mask_loss: float = _threshold(0.0185, 0.0, 1.0, "mask_loss (warning): the mask drops a region for 1 to 8 frames while holding it on the frames before and after, and no limb moved away to account for it; flagged over the whole run when the region is thicker than this on a single frame or twice this over a longer run, from half this where the drawn skeleton crosses it (with pose_data). Such a dropout of large_loss_area of the person's mask or more fails as mask_loss_large. No model reads the raw mask: the Wan Animate workflow grows it into the final mask (GrowMaskWithBlur expand 10, BlockifyMask 32) first, so only the part the final leaves out on every frame of the run counts. Thickness is the radius of the largest disc the region holds, as a fraction of the frame's shorter side: a dropped hand or foot is as thick as it is wide, the slivers a mask's outline jitters by are a few pixels thick. Set on the test clips.", 0.0005)


@dataclass
class MaskGuardConfig(MaskChecksConfig):
    """The Mask Guard's thresholds: the mask checks', then those of its two fails set on the Wan
    Animate keypoint mask, which the SCAIL-2 guard does not run (its node shows MaskChecksConfig).
    A subclass's fields follow its base's: these are the last widgets of the Mask Guard and the
    WanAnimate Preprocess Guard, so the widgets of saved workflows keep their positions."""
    head_out_eyes_ears: int = _threshold(2, 1, 4, "mask_head_out (fail): the head outside the mask - the drawn nose, or at least this many of the drawn eyes and ears, inside the frame and outside the (slightly grown) mask - fails on its frame in place of the mask_missing_keypoints warning; fewer eyes and ears outside only count toward that warning. The nose outside always fails. A head the mask leaves out is redrawn by the model from nothing. 1: a single eye or ear outside fails too, also where the mask's outline at the hair leaves one out; 4: only all four, or the nose. A face the pose model draws on the back of a head lies on the head, which the mask covers. Needs pose_data; a frame whose mask is empty is mask_empty's. The default 2 is set on the test clips: a correct (prompt) mask left no head keypoint out on any frame, and on every frame where the keypoint mask lost the head, the nose or two or more of the eyes and ears were out.", 1)
    large_loss_area: float = _threshold(0.05, 0.0, 1.0, "mask_loss_large (fail): a dropout that reaches the mask_loss warning and drops this share of the person's mask or more (measured on the frame before its run; mask_loss_area in the metrics) fails on the frames of its run in place of the mask_loss warning: a hand or a leg the models cannot restore. A smaller one stays a mask_loss warning; a frame whose mask is empty is mask_empty's. 0: every mask_loss dropout fails; 1: only one that drops the whole mask. The default 0.05 is set on the test clips: a correct (prompt) mask's dropouts stayed under 4% of the person, the hands and legs the keypoint mask dropped measured 5.5-11% on the raw mask.", 0.01)


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
