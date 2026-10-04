"""SCAIL-2 Colored Mask, SCAIL-2 Preprocess and SCAIL-2 Preprocess Guard, the preprocess of the
SCAIL-2 Long Video Sampler. The preprocess wrapper calls the individual nodes, so it computes
exactly what the chained nodes compute."""

from ..libs import log
from ..libs.config_widgets import config_inputs
from .common import _config
from .guard import _guard_inputs
from .pose import VITPOSE, BCVPoseDetection
from .sam3_1_multiplex import BCVSAM3VideoTrack, track_reference
from .unused_outputs import LINK_INPUTS, drop_unwanted, heavy_wanted, wants

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
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("pose_video_mask", "reference_image_mask")
    # not rendered when nothing links it (nodes/unused_outputs.py); reference_image_mask is one frame
    HEAVY_OUTPUTS = ("pose_video_mask",)
    FUNCTION = "render"
    CATEGORY = SCAIL
    DESCRIPTION = "Renders the person masks as the colored masks SCAIL-2 reads: the person in blue (the first colour of the palette SCAIL-2 was trained on) on the background of the mode. One person; the colours are pure, so core's 28-channel extraction reads them exactly."

    def render(self, driving_mask, replacement_mode, reference_mask=None, prompt_graph=None, unique_id=None, *, wanted=None):
        """`wanted`: the heavy outputs a wrapper needs, in place of the link stamp's."""
        from ..pipelines import scail2

        wanted = heavy_wanted(type(self), prompt_graph, unique_id, wanted)
        return drop_unwanted(type(self), scail2.colored_masks(driving_mask, replacement_mode, reference_mask,
                                                              render_driving=wants(wanted, "pose_video_mask")), wanted)


PROMPT_TOOLTIP = "What to segment: the person in the driving video (prompt and prompt_pose modes) and the character on the reference image (every mode, unless reference_mask is connected). The defaults were validated with this prompt."
REFERENCE_SOURCE_TOOLTIP = "face_crop only: the reference image at its own resolution, not resized (Load Reference Image's source_image), which the face close-up is cut from. Needed when face_crop is on; not used when it is off."
FACE_CROP_TOOLTIP = "Off: the primary reference alone. On: a face close-up as a second reference (SCAIL-2 multi-reference). Needs reference_source (Load Reference Image's source_image). Pose Detection with pose_model finds the face on it, in every mode; the crop has the generation's aspect (reference_image's width x height), the face box filling its width (its height when the generation is too wide for the box), centred, the head in the upper part, cut from reference_source and resized to reference_image's size; SAM 3.1 Multiplex finds the character on it with the prompt, and its colored mask is the character in blue on black, in both modes. reference_images and reference_image_mask then have two frames, the primary first: link the sampler's reference_image from reference_images."
POSE_CONFIG_TOOLTIP = "[box_keypoint, prompt_pose] Overrides from Pose Config for the Pose Detection the node runs on the driving frames in these modes; the measured defaults without it. Ignored in prompt mode, which runs no pose."


class BCVSCAIL2Preprocess:
    @classmethod
    def INPUT_TYPES(cls):
        sam3_types = BCVSAM3VideoTrack.INPUT_TYPES()
        colored = BCVSCAIL2ColoredMask.INPUT_TYPES()
        prompt_kind, prompt_options = sam3_types["required"]["prompt"]
        pose = BCVPoseDetection.INPUT_TYPES()["optional"]
        pose_config_kind, pose_config_options = pose["pose_config"]
        pose_model_kind, pose_model_options = pose["pose_model"]
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The driving video, at the generation size (divisible by 32)."}),
                "reference_image": ("IMAGE", {"tooltip": "The character (Load Reference Image's resized_image). In replacement mode SCAIL-2 expects it posed like the first driving frame. With face_crop on, the face close-up gets its size and aspect."}),
                "replacement_mode": colored["required"]["replacement_mode"],
                "mode": sam3_types["required"]["mode"],
                "prompt": (prompt_kind, {**prompt_options, "tooltip": PROMPT_TOOLTIP}),
                "black_background": ("BOOLEAN", {"default": False, "tooltip": "Animation mode only. On: pose_video is the driving video with every pixel outside the person's mask black, as SCAIL-2's training pose videos were (zai-org/SCAIL-2 issue #17; SCAIL-Pose's --crop_e2e_mask), so the driving video's background and camera do not reach the result. Off: the driving video unchanged. On with replacement_mode is an error: replacement mode keeps the driving video's background."}),
                "face_crop": ("BOOLEAN", {"default": False, "tooltip": FACE_CROP_TOOLTIP}),
            },
            "optional": {
                # the first optional input: its socket sits right below reference_image's
                "reference_source": ("IMAGE", {"tooltip": REFERENCE_SOURCE_TOOLTIP}),
                "reference_mask": ("MASK", {"tooltip": "The character on the reference image. Without it the reference image is segmented with the prompt, in prompt mode whatever the mode widget says: the pose modes are video modes, and the reference is one image."}),
                "pose_config": (pose_config_kind, {**pose_config_options, "tooltip": POSE_CONFIG_TOOLTIP}),
                "sam3_config": sam3_types["optional"]["sam3_config"],
                "pose_model": (pose_model_kind, {**pose_model_options, "tooltip": "[box_keypoint, prompt_pose, face_crop] " + pose_model_options["tooltip"] + " With face_crop on it also finds the face on reference_source, in every mode; otherwise ignored in prompt mode, which runs no pose."}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "MASK", "BOOLEAN")
    RETURN_NAMES = ("pose_video", "pose_video_mask", "reference_image_mask", "reference_images", "mask", "reference_mask",
                    "replacement_mode")
    # nodes/unused_outputs.py: pose_video (a new clip with black_background, else the input) and
    # pose_video_mask are not computed when nothing links them; mask, which both are cut by, is
    # dropped at return. The references and their masks are one or two frames.
    HEAVY_OUTPUTS = ("pose_video", "pose_video_mask", "mask")
    FUNCTION = "process"
    CATEGORY = SCAIL
    DESCRIPTION = "The SCAIL-2 preprocess in one node, one person: SAM 3.1 Multiplex Video Track in the chosen mode on the whole driving video once, so the mask keeps its shape and colour across the sampler's chunks; box_keypoint and prompt_pose read the pose, so in those modes the node first runs Pose Detection on the driving frames (its default widgets, pose_config when connected, and pose_model) and hands its pose_data to the track, while prompt mode runs no pose. The reference image is tracked in prompt mode in every mode, unless reference_mask is connected: the pose modes are video modes, and the reference is one image. Then SCAIL-2 Colored Mask. SCAIL-2 draws no pose; the pose only shapes the mask. pose_video is the driving video, SCAIL-2's end-to-end pose input in animation and replacement mode alike: unchanged, or in animation mode with black_background on, with everything outside the person's mask black. face_crop adds a face close-up as a second reference (SCAIL-2 multi-reference), cut from reference_source around the face Pose Detection finds there, its colored mask the character in blue on black; reference_images is the reference_image the sampler takes (the primary alone with face_crop off, the primary and the close-up with it on), paired frame by frame with reference_image_mask."

    def process(self, images, reference_image, replacement_mode, mode, prompt, black_background=False, reference_source=None,
                reference_mask=None, pose_config=None, sam3_config=None, pose_model=VITPOSE, face_crop=False, prompt_graph=None,
                unique_id=None):
        from ..pipelines import scail2
        from ..pipelines.sam3_1_multiplex import track as sam3

        scail2.check_black_background(black_background, replacement_mode)
        scail2.check_face_crop(face_crop, reference_source, reference_image, reference_mask)
        wanted = heavy_wanted(type(self), prompt_graph, unique_id)
        pose_data = None
        if mode != sam3.MODE_PROMPT:
            pose_data = _pose_data(images, pose_config, pose_model)
        else:
            unused = [name for name, changed in (("pose_config", pose_config is not None),
                                                 ("pose_model", pose_model != VITPOSE and not face_crop)) if changed]
            if unused:
                log.info(f"prompt mode runs no pose; {' and '.join(unused)} not used")
        tracker = BCVSAM3VideoTrack()
        (mask,) = tracker.track(images, mode, prompt, 1, -1, pose_data=pose_data, sam3_config=sam3_config)
        if reference_mask is None:
            reference_mask = track_reference(reference_image, prompt, sam3_config)
        pose_video_mask, reference_image_mask = BCVSCAIL2ColoredMask().render(mask, replacement_mode, reference_mask=reference_mask,
                                                                              wanted=wanted)
        reference_images = reference_image
        if face_crop:
            reference_images, reference_image_mask = _with_face_reference(reference_image, reference_image_mask, reference_source,
                                                                          prompt, sam3_config, pose_model)
        pose_video = scail2.driving_on_black(images, mask) if black_background and wants(wanted, "pose_video") else images
        return drop_unwanted(type(self), (pose_video, pose_video_mask, reference_image_mask, reference_images, mask, reference_mask,
                                          bool(replacement_mode)), wanted)


def _pose_data(images, pose_config, pose_model):
    """Pose Detection's pose_data on `images` at its default widgets, pose_config and pose_model. SCAIL-2
    draws no pose: only the pose_data is used, which carries the default draw_threshold prompt_pose picks
    its points at; the pose images are not drawn (wanted: none of its heavy outputs)."""
    widgets = {name: options[1]["default"] for name, options in BCVPoseDetection.INPUT_TYPES()["required"].items()
               if name != "images"}
    return BCVPoseDetection().detect(images, **widgets, pose_config=pose_config, pose_model=pose_model, wanted=set())[1]


def _with_face_reference(reference_image, reference_image_mask, reference_source, prompt, sam3_config, pose_model):
    """(reference_images, reference_image_mask) with the face close-up appended as the second
    reference: Pose Detection (its defaults and pose_model) finds the face on reference_source, the
    close-up is cut in the generation's aspect (reference_image's size), SAM 3.1 Multiplex finds the
    character on it with the prompt, and its colored mask is the character on black."""
    from ..pipelines import scail2

    height, width = reference_image.shape[1:3]
    source_height, source_width = reference_source.shape[1:3]
    box = scail2.face_box(_pose_data(reference_source, None, pose_model), source_width, source_height)
    face = scail2.face_reference(reference_source, box, width, height)
    face_mask = scail2.extra_reference_mask(track_reference(face, prompt, sam3_config))
    return scail2.with_extra_reference(reference_image, reference_image_mask, face, face_mask)


SCAIL2_GUARD_TOOLTIP = "Stop the workflow when a SCAIL-2 check fails: no driving frame has the person, a driving frame's mask is torn (a detached piece of 5% of her or more), a reference mask (the primary or an extra one) has no character; with pose_data also the driving mask empty on a frame with a person, the head or a whole limb outside it, or a hand-sized region of her the latent grid loses. Warnings never stop. Off still measures and reports every check."
POSE_DATA_TOOLTIP = "Pose Detection on the driving frames, at the generation size (the frames SCAIL-2 Preprocess got). Optional, but it gives the best result: with it the driving mask also gets the Mask Guard's pose-based checks; without it the guard cannot catch a region of her the mask drops (on SCAIL-2's latent grid a limb in motion empties cells of a correct mask, which only the pose tells apart), the head or a limb outside the mask, background attached to the body, an empty or leaking mask (box-based), or tell whether a detached piece is the person."


class BCVSCAIL2PreprocessGuard:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines import guard

        return {
            "required": {
                "pose_video_mask": ("IMAGE", {"tooltip": "The colored driving mask (SCAIL-2 Preprocess or SCAIL-2 Colored Mask), at the generation size."}),
                "reference_image_mask": ("IMAGE", {"tooltip": "The colored reference masks, one per reference (with face_crop the primary and the face close-up). The mode is read from the primary's (the first frame's) border, as the sampler reads it; each extra reference is on black in both modes and is checked for the character on its own."}),
                **_guard_inputs(guard.SCAIL2GuardConfig, "scail2_guard", SCAIL2_GUARD_TOOLTIP),
                **config_inputs(guard.MaskGuardConfig),
            },
            "optional": {"pose_data": ("POSEDATA", {"tooltip": POSE_DATA_TOOLTIP})},
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_video_mask", "reference_image_mask", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = SCAIL
    DESCRIPTION = "Checks the colored masks of SCAIL-2 Preprocess before the sampler, on the person as the sampler reads it (blue above 225/255); end-to-end SCAIL-2 draws no pose, so pose_data is optional. Stops the workflow when scail2_guard is on and no driving frame has the person, a driving frame's mask is torn (a detached piece of 5% of her or more) or a reference mask has no character (the primary, or an extra reference such as the face close-up, each named by its index); with pose_data also when the driving mask is empty on a frame with a person, leaves the head or a whole forearm-and-hand or lower leg out, or drops a hand-sized region of her for a run of frames (from the clip's start or to its end too). The driving mask is judged the way the sampler reads it, on its latent grid of 16 x 16 px cells with nothing grown: a dropped region counts only where it empties whole cells, and a drawn keypoint in a cell the grid reads as the person is inside the mask. Warnings, which never stop: without pose_data a blank driving frame (normal when the person leaves the shot), a split-up reference mask (any reference), in replacement mode a primary reference not placed like the first driving frame, and with pose_data a leaking mask or background attached to the body. 'metrics' has every measurement and 'timeline' plots the driving frames. Both masks pass through."

    def check(self, pose_video_mask, reference_image_mask, scail2_guard, pose_data=None, **thresholds):
        from ..pipelines import guard
        from ..pipelines.guard import scail2

        return tuple(scail2.check_scail2(pose_video_mask, reference_image_mask, _config(guard.SCAIL2GuardConfig, thresholds),
                                         enabled=scail2_guard, pose_data=pose_data,
                                         mask_config=_config(guard.MaskGuardConfig, thresholds)))
