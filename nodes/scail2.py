"""SCAIL-2 Colored Mask, SCAIL-2 Preprocess and SCAIL-2 Preprocess Guard, the preprocess of the
SCAIL-2 Long Video Sampler. The preprocess wrapper calls the individual nodes, so it computes
exactly what the chained nodes compute."""

from .common import _config
from .guard import _guard_inputs
from .sam3_1_multiplex import BCVSAM3VideoTrack

SCAIL = "BCVideoNodes/SCAIL"


class BCVSCAIL2ColoredMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "driving_mask": ("MASK", {"tooltip": "The person in the driving video, one mask per frame (SAM 3.1 Multiplex Video Track). On above 0.5."}),
                "replacement_mode": ("BOOLEAN", {"default": False, "tooltip": "False: animation mode (driving mask on black, reference mask on white). True: replacement mode (driving mask on white, reference mask on black). Set the sampler's replacement_mode the same way."}),
            },
            "optional": {
                "reference_mask": ("MASK", {"tooltip": "The character on the reference image. Without it the reference mask is the background alone: animation mode can then collapse into replacement behaviour, and replacement mode raises."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("pose_video_mask", "reference_image_mask")
    FUNCTION = "render"
    CATEGORY = SCAIL
    DESCRIPTION = "Renders the person masks as the colored masks SCAIL-2 reads: the person in blue (the first colour of the palette SCAIL-2 was trained on) on the background of the mode. One person; the colours are pure, so core's 28-channel extraction reads them exactly."

    def render(self, driving_mask, replacement_mode, reference_mask=None):
        from ..pipelines import scail2

        return scail2.colored_masks(driving_mask, replacement_mode, reference_mask)


class BCVSCAIL2Preprocess:
    @classmethod
    def INPUT_TYPES(cls):
        sam3_types = BCVSAM3VideoTrack.INPUT_TYPES()
        colored = BCVSCAIL2ColoredMask.INPUT_TYPES()
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The driving video, at the generation size (divisible by 32)."}),
                "reference_image": ("IMAGE", {"tooltip": "The character. In replacement mode SCAIL-2 expects it posed like the first driving frame."}),
                "replacement_mode": colored["required"]["replacement_mode"],
                "prompt": sam3_types["required"]["prompt"],
                "black_background": ("BOOLEAN", {"default": False, "tooltip": "Animation mode only. On: pose_video is the driving video with every pixel outside the person's mask black, as SCAIL-2's training pose videos were (zai-org/SCAIL-2 issue #17; SCAIL-Pose's --crop_e2e_mask), so the driving video's background and camera do not reach the result. Off: the driving video unchanged. On with replacement_mode is an error: replacement mode keeps the driving video's background."}),
            },
            "optional": {
                "reference_mask": ("MASK", {"tooltip": "The character on the reference image. Without it the reference image is segmented with the prompt."}),
                "sam3_config": sam3_types["optional"]["sam3_config"],
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("pose_video", "pose_video_mask", "reference_image_mask", "mask", "reference_mask")
    FUNCTION = "process"
    CATEGORY = SCAIL
    DESCRIPTION = "The SCAIL-2 preprocess in one node, one person: SAM 3.1 Multiplex Video Track (prompt mode) on the whole driving video once, so the mask keeps its shape and colour across the sampler's chunks, and on the reference image unless reference_mask is connected; then SCAIL-2 Colored Mask. pose_video is the driving video, SCAIL-2's end-to-end pose input in both modes: unchanged, or in animation mode with black_background on, with everything outside the person's mask black."

    def process(self, images, reference_image, replacement_mode, prompt, black_background=False, reference_mask=None,
                sam3_config=None):
        from ..pipelines import scail2
        from ..pipelines.sam3_1_multiplex import track as sam3

        scail2.check_black_background(black_background, replacement_mode)
        tracker = BCVSAM3VideoTrack()
        (mask,) = tracker.track(images, sam3.MODE_PROMPT, prompt, 1, -1, sam3_config=sam3_config)
        if reference_mask is None:
            (reference_mask,) = tracker.track(reference_image, sam3.MODE_PROMPT, prompt, 1, -1, sam3_config=sam3_config)
        pose_video_mask, reference_image_mask = BCVSCAIL2ColoredMask().render(mask, replacement_mode, reference_mask=reference_mask)
        pose_video = scail2.driving_on_black(images, mask) if black_background else images
        return (pose_video, pose_video_mask, reference_image_mask, mask, reference_mask)


SCAIL2_GUARD_TOOLTIP = "Stop the workflow when a SCAIL-2 check fails (no driving frame has the person, the reference mask has no character). Warnings never stop. Off still measures and reports every check."


class BCVSCAIL2PreprocessGuard:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines import guard

        return {"required": {
            "pose_video_mask": ("IMAGE", {"tooltip": "The colored driving mask (SCAIL-2 Preprocess or SCAIL-2 Colored Mask), at the generation size."}),
            "reference_image_mask": ("IMAGE", {"tooltip": "The colored reference mask. The mode is read from its border, as the sampler reads it."}),
            **_guard_inputs(guard.SCAIL2GuardConfig, "scail2_guard", SCAIL2_GUARD_TOOLTIP),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_video_mask", "reference_image_mask", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = SCAIL
    DESCRIPTION = "Checks the colored masks of SCAIL-2 Preprocess before the sampler, without a pose (end-to-end SCAIL-2 draws none), on the person as the sampler reads it (blue above 225/255). Stops the workflow when scail2_guard is on and no driving frame has the person or the reference mask has no character. Warnings, which never stop: blank or split-up driving frames (normal when the person leaves the shot or is occluded), a split-up reference mask, a reference character the core node's center crop cuts, and in replacement mode a reference not placed like the first driving frame. 'metrics' has every measurement and 'timeline' plots the driving frames. Both masks pass through."

    def check(self, pose_video_mask, reference_image_mask, scail2_guard, **thresholds):
        from ..pipelines import guard
        from ..pipelines.guard import scail2

        return tuple(scail2.check_scail2(pose_video_mask, reference_image_mask, _config(guard.SCAIL2GuardConfig, thresholds),
                                         enabled=scail2_guard))
