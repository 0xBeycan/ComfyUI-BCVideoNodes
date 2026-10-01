"""Pose Guard and Mask Guard."""

from ..libs.config_widgets import config_inputs
from .common import PREPROCESS, _config


def _guard_inputs(config_cls, switch, tooltip):
    return {switch: ("BOOLEAN", {"default": True, "tooltip": tooltip}), **config_inputs(config_cls)}


def _raw_final_inputs():
    """The Mask Guard's grow and block_size widgets: the preprocess's (libs.mask.FinalMaskConfig), its
    block_size also taking 0, a final without blocks, so the guard can judge a raw mask as it is."""
    from ..libs.mask import FinalMaskConfig

    inputs = config_inputs(FinalMaskConfig)
    kind, options = inputs["block_size"]
    inputs["block_size"] = (kind, {**options, "min": 0, "tooltip": options["tooltip"] + RAW_BLOCK_SIZE_TOOLTIP})
    return inputs


def _final_widgets(widgets):
    """check_mask's grow and block_size from the guard's widgets (libs.mask.FinalMaskConfig)."""
    from dataclasses import asdict

    from ..libs.mask import FinalMaskConfig

    return asdict(_config(FinalMaskConfig, widgets))


def _reference_mask(reference_image):
    """The character on `reference_image` as SAM 3.1 Multiplex finds it with the default prompt,
    as SCAIL-2 Preprocess segments its reference; None without a reference image."""
    if reference_image is None:
        return None
    from ..pipelines.sam3_1_multiplex import track as sam3
    from .sam3_1_multiplex import track_reference

    return track_reference(reference_image, sam3.PROMPT)


RAW_BLOCK_SIZE_TOOLTIP = (" Mask Guard only: 0 models a final without blocks, the mask grown by grow and read a pixel at a "
                          "time, so a region it loses needs no whole block; with grow 0 too, the mask is judged as it is.")
REFERENCE_IMAGE_TOOLTIP = "The reference image of the replacement run, the one the Wan Animate node gets (Load Reference Image). Optional: connected, the guard finds the character on it with SAM 3.1 Multiplex (prompt mode, the default prompt), places it as the Wan Animate node places the reference (center crop to the mask's aspect ratio, resized to the mask's size) and warns (reference_misaligned) when it overlaps the mask on frame 0 by an IoU below min_reference_iou: replacement expects the reference posed and placed like the first frame. A warning never stops. Not connected, nothing is checked and no SAM runs."
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
                # the reference check's threshold, the reference image, then the final mask's widgets, last
                "optional": {"pose_data": ("POSEDATA", {"tooltip": POSE_DATA_TOOLTIP}),
                             **config_inputs(guard.ReferenceGuardConfig),
                             "reference_image": ("IMAGE", {"tooltip": REFERENCE_IMAGE_TOOLTIP}),
                             **_raw_final_inputs()}}

    RETURN_TYPES = ("MASK", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Checks the mask frame by frame, against the drawn pose it belongs to when pose_data is connected, and fails only on damage the diffusion model cannot absorb (when mask_guard is on the workflow stops with the report): the mask empty on a frame with a person, a detached piece of 5% of her or more, the head outside the mask (the drawn nose, or head_out_eyes_ears of the drawn eyes and ears, default 2), a whole forearm-and-hand or lower leg outside it, or a hand-sized region of her the model loses - one the mask drops for a run of frames that the final mask (WanAnimate Preprocess's final_mask: the mask grown by grow and blockified by block_size, which must equal the preprocess's) leaves out too, a whole block of it (block_size 0: a final without blocks, any part; with grow 0 too, the mask as it is). Warnings never stop: a mask leaking outside the box, background attached to the body, and with reference_image connected a reference not placed like the first frame (the character SAM 3.1 Multiplex finds on it, placed as the Wan Animate node places the reference, overlapping the mask on frame 0 by an IoU below min_reference_iou). 'metrics' has every measurement per frame and 'timeline' plots them. The mask passes through."

    def check(self, mask, mask_guard, pose_data=None, reference_image=None, **thresholds):
        from ..pipelines import guard

        return tuple(guard.check_mask(mask, pose_data, _config(guard.MaskGuardConfig, thresholds), enabled=mask_guard,
                                      reference=_reference_mask(reference_image),
                                      reference_config=_config(guard.ReferenceGuardConfig, thresholds),
                                      **_final_widgets(thresholds)))
