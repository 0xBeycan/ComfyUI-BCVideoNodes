"""SCAIL-2 Colored Mask, SCAIL-2 Preprocess and SCAIL-2 Preprocess Guard, the preprocess of the
SCAIL-2 Long Video Sampler. The preprocess wrapper calls the individual nodes, so it computes
exactly what the chained nodes compute."""

from ..libs import log
from ..libs.config_widgets import config_inputs
from .common import _config
from .guard import _guard_inputs
from .pose import BCVPoseDetection
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


PROMPT_TOOLTIP = "What to segment: the person in the driving video (prompt and prompt_pose modes) and the character on the reference image (every mode, unless reference_mask is connected). The defaults were validated with this prompt."
POSE_CONFIG_TOOLTIP = "[box_keypoint, prompt_pose] Overrides from Pose Config for the Pose Detection the node runs on the driving frames in these modes; the measured defaults without it. Ignored in prompt mode, which runs no pose."


class BCVSCAIL2Preprocess:
    @classmethod
    def INPUT_TYPES(cls):
        sam3_types = BCVSAM3VideoTrack.INPUT_TYPES()
        colored = BCVSCAIL2ColoredMask.INPUT_TYPES()
        prompt_kind, prompt_options = sam3_types["required"]["prompt"]
        pose_config_kind, pose_config_options = BCVPoseDetection.INPUT_TYPES()["optional"]["pose_config"]
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The driving video, at the generation size (divisible by 32)."}),
                "reference_image": ("IMAGE", {"tooltip": "The character. In replacement mode SCAIL-2 expects it posed like the first driving frame."}),
                "replacement_mode": colored["required"]["replacement_mode"],
                "mode": sam3_types["required"]["mode"],
                "prompt": (prompt_kind, {**prompt_options, "tooltip": PROMPT_TOOLTIP}),
                "black_background": ("BOOLEAN", {"default": False, "tooltip": "Animation mode only. On: pose_video is the driving video with every pixel outside the person's mask black, as SCAIL-2's training pose videos were (zai-org/SCAIL-2 issue #17; SCAIL-Pose's --crop_e2e_mask), so the driving video's background and camera do not reach the result. Off: the driving video unchanged. On with replacement_mode is an error: replacement mode keeps the driving video's background."}),
            },
            "optional": {
                "reference_mask": ("MASK", {"tooltip": "The character on the reference image. Without it the reference image is segmented with the prompt, in prompt mode whatever the mode widget says: the pose modes are video modes, and the reference is one image."}),
                "pose_config": (pose_config_kind, {**pose_config_options, "tooltip": POSE_CONFIG_TOOLTIP}),
                "sam3_config": sam3_types["optional"]["sam3_config"],
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("pose_video", "pose_video_mask", "reference_image_mask", "mask", "reference_mask")
    FUNCTION = "process"
    CATEGORY = SCAIL
    DESCRIPTION = "The SCAIL-2 preprocess in one node, one person: SAM 3.1 Multiplex Video Track in the chosen mode on the whole driving video once, so the mask keeps its shape and colour across the sampler's chunks; box_keypoint and prompt_pose read the pose, so in those modes the node first runs Pose Detection on the driving frames (its default widgets, pose_config when connected) and hands its pose_data to the track, while prompt mode runs no pose. The reference image is tracked in prompt mode in every mode, unless reference_mask is connected: the pose modes are video modes, and the reference is one image. Then SCAIL-2 Colored Mask. SCAIL-2 draws no pose; the pose only shapes the mask. pose_video is the driving video, SCAIL-2's end-to-end pose input in animation and replacement mode alike: unchanged, or in animation mode with black_background on, with everything outside the person's mask black."

    def process(self, images, reference_image, replacement_mode, mode, prompt, black_background=False, reference_mask=None,
                pose_config=None, sam3_config=None):
        from ..pipelines import scail2
        from ..pipelines.sam3_1_multiplex import track as sam3

        scail2.check_black_background(black_background, replacement_mode)
        pose_data = None
        if mode != sam3.MODE_PROMPT:
            # SCAIL-2 draws no pose: Pose Detection runs at its default widgets and only its pose_data is
            # used, which carries the default draw_threshold prompt_pose picks its points at
            widgets = {name: options[1]["default"] for name, options in BCVPoseDetection.INPUT_TYPES()["required"].items()
                       if name != "images"}
            _, pose_data, _, _ = BCVPoseDetection().detect(images, **widgets, pose_config=pose_config)
        elif pose_config is not None:
            log.info("prompt mode runs no pose; pose_config not used")
        tracker = BCVSAM3VideoTrack()
        (mask,) = tracker.track(images, mode, prompt, 1, -1, pose_data=pose_data, sam3_config=sam3_config)
        if reference_mask is None:
            (reference_mask,) = tracker.track(reference_image, sam3.MODE_PROMPT, prompt, 1, -1, sam3_config=sam3_config)
        pose_video_mask, reference_image_mask = BCVSCAIL2ColoredMask().render(mask, replacement_mode, reference_mask=reference_mask)
        pose_video = scail2.driving_on_black(images, mask) if black_background else images
        return (pose_video, pose_video_mask, reference_image_mask, mask, reference_mask)


SCAIL2_GUARD_TOOLTIP = "Stop the workflow when a SCAIL-2 check fails (no driving frame has the person, the reference mask has no character; with pose_data also an empty or leaking driving mask). Warnings never stop. Off still measures and reports every check."
POSE_DATA_TOOLTIP = "Pose Detection on the driving frames, at the generation size (the frames SCAIL-2 Preprocess got). Optional, but it gives the best result: with it the driving mask also gets the Mask Guard's pose-based checks, but not its two keypoint-mask fails (the head outside the mask, a large region dropped); without it the guard cannot catch a limb outside the mask, body the pose does not draw, background attached to the body, an empty, leaking or unstable mask (box-based), or tell whether a detached piece is the person."
# SCAIL-2 reads the driving mask on a 16 x 16 px latent grid with 4 frames stacked per latent frame, so
# motion (a limb that moved, an arm-body gap opening for a frame) empties whole cells on correct masks;
# only a hand-sized loss separates from that (measured on the test clips: motion 13-25 px on a single
# frame and 28-35 px on runs, a dropped hand 53 px, on a 704 px short side).
SCAIL2_MAX_MASK_LOSS = 0.05
MASK_LOSS_TOOLTIP = "mask_loss (warning): the driving mask drops a region for 1 to 8 frames while holding it on the frames before and after, and no limb moved away to account for it; flagged over the whole run when the region is thicker than this on a single frame or twice this over a longer run, from half this where the drawn skeleton crosses it (with pose_data). The sampler reads the mask on its latent grid (16 x 16 px cells of the generation, a cell on where the person fills at least half of it, nothing grown), so only the part that grid drops too counts: a hole or a sliver inside a cell never reaches the model. Thickness is the radius of the largest disc the region holds, as a fraction of the frame's shorter side. The default (0.05, higher than the Mask Guard's) counts a hand-sized loss only: on the latent grid, motion empties whole cells on a correct mask too. Set on the test clips."


class BCVSCAIL2PreprocessGuard:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines import guard

        mask = config_inputs(guard.MaskChecksConfig)
        kind, options = mask["max_mask_loss"]
        mask["max_mask_loss"] = (kind, {**options, "default": SCAIL2_MAX_MASK_LOSS, "tooltip": MASK_LOSS_TOOLTIP})
        return {
            "required": {
                "pose_video_mask": ("IMAGE", {"tooltip": "The colored driving mask (SCAIL-2 Preprocess or SCAIL-2 Colored Mask), at the generation size."}),
                "reference_image_mask": ("IMAGE", {"tooltip": "The colored reference mask. The mode is read from its border, as the sampler reads it."}),
                **_guard_inputs(guard.SCAIL2GuardConfig, "scail2_guard", SCAIL2_GUARD_TOOLTIP),
                **mask,
            },
            "optional": {"pose_data": ("POSEDATA", {"tooltip": POSE_DATA_TOOLTIP})},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_video_mask", "reference_image_mask", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = SCAIL
    DESCRIPTION = "Checks the colored masks of SCAIL-2 Preprocess before the sampler, on the person as the sampler reads it (blue above 225/255); end-to-end SCAIL-2 draws no pose, so pose_data is optional. Stops the workflow when scail2_guard is on and no driving frame has the person or the reference mask has no character (with pose_data also on an empty or leaking driving mask). Warnings, which never stop: blank or split-up driving frames (normal when the person leaves the shot or is occluded), a region the driving mask drops for one frame, a split-up reference mask, in replacement mode a reference not placed like the first driving frame, and with pose_data the Mask Guard's pose-based warnings. The driving mask is judged the way the sampler reads it, on its latent grid of 16 x 16 px cells with nothing grown: a dropped region, and a drawn keypoint outside the mask, count only where that grid has them too, so a hole or a sliver inside a cell, or a keypoint in a cell the grid reads as the person, never warns. 'metrics' has every measurement and 'timeline' plots the driving frames. Both masks pass through."

    def check(self, pose_video_mask, reference_image_mask, scail2_guard, pose_data=None, **thresholds):
        from ..pipelines import guard
        from ..pipelines.guard import scail2

        return tuple(scail2.check_scail2(pose_video_mask, reference_image_mask, _config(guard.SCAIL2GuardConfig, thresholds),
                                         enabled=scail2_guard, pose_data=pose_data,
                                         mask_config=_config(guard.MaskChecksConfig, thresholds)))
