"""Pose Guard and Mask Guard."""

from ..libs.config_widgets import config_inputs
from .common import PREPROCESS, _config


def _guard_inputs(config_cls, switch, tooltip):
    return {switch: ("BOOLEAN", {"default": True, "tooltip": tooltip}), **config_inputs(config_cls)}


MASK_GUARD_TOOLTIP = "Stop the workflow when a mask check fails: a person the mask leaves empty, a large piece torn off the mask, the head or a whole limb outside the mask, a hand-sized region of her the model loses. Warnings never stop. Off still measures and reports every check."
POSE_DATA_TOOLTIP = "The drawn pose of the same frames at the same size (Pose Detection or WanAnimate Preprocess). Optional, but it gives the best result: without it the guard runs only its pose-free checks (a torn-off piece, a region dropped for a few frames between two that hold it) and cannot catch the head or a limb outside the mask, a loss that runs from the clip's start or to its end, background attached to the body or an empty or leaking mask (box-based), or tell whether a detached piece inside the frame is the person."


class BCVPoseGuard:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines import guard

        return {"required": {"pose_data": ("POSEDATA",), **config_inputs(guard.PoseGuardConfig)}}

    RETURN_TYPES = ("POSEDATA", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_data", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Checks the drawn pose frame by frame and reports warnings, which never stop the workflow: a torso jump, a limb spike, a subject switch. A pose error the diffusion model does not absorb shows in the mask checks. 'metrics' has every measurement per frame and 'timeline' plots them. pose_data passes through."

    def check(self, pose_data, **thresholds):
        from ..pipelines import guard

        return tuple(guard.check_pose(pose_data, _config(guard.PoseGuardConfig, thresholds)))


class BCVMaskGuard:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines import guard

        return {"required": {"mask": ("MASK",), **_guard_inputs(guard.MaskGuardConfig, "mask_guard", MASK_GUARD_TOOLTIP)},
                "optional": {"pose_data": ("POSEDATA", {"tooltip": POSE_DATA_TOOLTIP})}}

    RETURN_TYPES = ("MASK", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Checks the mask frame by frame, against the drawn pose it belongs to when pose_data is connected, and fails only on damage the diffusion model cannot absorb (when mask_guard is on the workflow stops with the report): the mask empty on a frame with a person, a detached piece of 5% of her or more, the head outside the mask (the drawn nose, or head_out_eyes_ears of the drawn eyes and ears, default 2), a whole forearm-and-hand or lower leg outside it, or a hand-sized region of her the model loses - one the mask drops for a run of frames that the final mask (GrowMaskWithBlur expand 10, BlockifyMask 32) leaves out too, a whole block of it. Warnings never stop: a mask leaking outside the box, background attached to the body. 'metrics' has every measurement per frame and 'timeline' plots them. The mask passes through."

    def check(self, mask, mask_guard, pose_data=None, **thresholds):
        from ..pipelines import guard

        return tuple(guard.check_mask(mask, pose_data, _config(guard.MaskGuardConfig, thresholds), enabled=mask_guard))
