"""Pose Guard and Mask Guard."""

from ..libs.config_widgets import config_inputs
from .common import PREPROCESS, _config


def _guard_inputs(config_cls, switch, tooltip):
    return {switch: ("BOOLEAN", {"default": True, "tooltip": tooltip}), **config_inputs(config_cls)}


POSE_GUARD_TOOLTIP = "Stop the workflow when a pose check fails (incomplete skeleton, torso jump, subject switch). Off still measures and reports every check."
MASK_GUARD_TOOLTIP = "Stop the workflow when a mask check fails (empty, leaking, fragmented, keypoints outside, unstable). Off still measures and reports every check."


class BCVPoseGuard:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines import guard

        return {"required": {"pose_data": ("POSEDATA",), **_guard_inputs(guard.PoseGuardConfig, "pose_guard", POSE_GUARD_TOOLTIP)}}

    RETURN_TYPES = ("POSEDATA", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_data", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Checks the drawn pose frame by frame (incomplete skeletons, torso jumps, limb spikes, subject switches; limbs missing for a stretch is a warning). A failed check stops the workflow with the report when pose_guard is on; warnings never stop. 'metrics' has every measurement per frame and 'timeline' plots them. pose_data passes through."

    def check(self, pose_data, pose_guard, **thresholds):
        from ..pipelines import guard

        return tuple(guard.check_pose(pose_data, _config(guard.PoseGuardConfig, thresholds), enabled=pose_guard))


class BCVMaskGuard:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines import guard

        return {"required": {"mask": ("MASK",), "pose_data": ("POSEDATA",),
                             **_guard_inputs(guard.MaskGuardConfig, "mask_guard", MASK_GUARD_TOOLTIP)}}

    RETURN_TYPES = ("MASK", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Checks the mask frame by frame against the drawn pose it belongs to (empty, leaking outside the box, detached pieces, keypoints outside the mask, body the pose does not draw, unstable; small detached specks, background attached to the body and a single limb end outside the mask are warnings). A failed check stops the workflow with the report when mask_guard is on; warnings never stop. 'metrics' has every measurement per frame and 'timeline' plots them. The mask passes through."

    def check(self, mask, pose_data, mask_guard, **thresholds):
        from ..pipelines import guard

        return tuple(guard.check_mask(mask, pose_data, _config(guard.MaskGuardConfig, thresholds), enabled=mask_guard))
