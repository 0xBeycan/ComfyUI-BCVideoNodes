"""Pose Guard and Mask Guard."""

from ..libs.config_widgets import config_inputs
from .common import PREPROCESS, _config


def _guard_inputs(config_cls, switch, tooltip):
    return {switch: ("BOOLEAN", {"default": True, "tooltip": tooltip}), **config_inputs(config_cls)}


POSE_GUARD_TOOLTIP = "Stop the workflow when a pose check fails (torso jump, subject switch). Warnings never stop. Off still measures and reports every check."
MASK_GUARD_TOOLTIP = "Stop the workflow when a mask check fails (empty, leaking, fragmented). Warnings never stop. Off still measures and reports every check."
POSE_DATA_TOOLTIP = "The drawn pose of the same frames at the same size (Pose Detection or WanAnimate Preprocess). Optional, but it gives the best result: without it the guard runs only its pose-free checks (a detached piece, a region dropped for one frame) and cannot catch a limb outside the mask, body the pose does not draw, background attached to the body, an empty, leaking or unstable mask (box-based), or tell whether a detached piece is the person."


class BCVPoseGuard:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines import guard

        return {"required": {"pose_data": ("POSEDATA",), **_guard_inputs(guard.PoseGuardConfig, "pose_guard", POSE_GUARD_TOOLTIP)}}

    RETURN_TYPES = ("POSEDATA", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_data", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Checks the drawn pose frame by frame. A torso jump or a subject switch stops the workflow with the report when pose_guard is on; warnings never stop: an incomplete skeleton, a limb spike, a limb missing for a stretch (a limb out of the shot or hidden by the body is never missing). 'metrics' has every measurement per frame and 'timeline' plots them. pose_data passes through."

    def check(self, pose_data, pose_guard, **thresholds):
        from ..pipelines import guard

        return tuple(guard.check_pose(pose_data, _config(guard.PoseGuardConfig, thresholds), enabled=pose_guard))


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
    DESCRIPTION = "Checks the mask frame by frame, against the drawn pose it belongs to when pose_data is connected. An empty, leaking (outside the box) or fragmented mask stops the workflow with the report when mask_guard is on; warnings never stop: small detached specks, background attached to the body, keypoints outside the mask, a limb end the mask lost, body the pose does not draw, an unstable mask, a region dropped for one frame. 'metrics' has every measurement per frame and 'timeline' plots them. The mask passes through."

    def check(self, mask, mask_guard, pose_data=None, **thresholds):
        from ..pipelines import guard

        return tuple(guard.check_mask(mask, pose_data, _config(guard.MaskGuardConfig, thresholds), enabled=mask_guard))
