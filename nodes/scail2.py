"""SCAIL-2 Colored Mask and SCAIL-2 Preprocess, the preprocess of the SCAIL-2 Long Video
Sampler. The preprocess wrapper calls the individual nodes, so it computes exactly what the
chained nodes compute."""

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
    DESCRIPTION = "The SCAIL-2 preprocess in one node, one person: SAM 3.1 Multiplex Video Track (prompt mode) on the whole driving video once, so the mask keeps its shape and colour across the sampler's chunks, and on the reference image unless reference_mask is connected; then SCAIL-2 Colored Mask. pose_video is the driving video unchanged: SCAIL-2's end-to-end mode reads the raw driving video as its pose input in both modes."

    def process(self, images, reference_image, replacement_mode, prompt, reference_mask=None, sam3_config=None):
        from ..pipelines.sam3_1_multiplex import track as sam3

        tracker = BCVSAM3VideoTrack()
        (mask,) = tracker.track(images, sam3.MODE_PROMPT, prompt, 1, -1, sam3_config=sam3_config)
        if reference_mask is None:
            (reference_mask,) = tracker.track(reference_image, sam3.MODE_PROMPT, prompt, 1, -1, sam3_config=sam3_config)
        pose_video_mask, reference_image_mask = BCVSCAIL2ColoredMask().render(mask, replacement_mode, reference_mask=reference_mask)
        return (images, pose_video_mask, reference_image_mask, mask, reference_mask)
