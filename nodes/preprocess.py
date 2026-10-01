"""The Wan Animate wrappers: WanAnimate Preprocess and WanAnimate Preprocess Guard. Each one
calls the individual nodes, so it computes exactly what the chained nodes compute."""

from ..libs.config_widgets import config_inputs
from .common import _config
from .face import BCVFaceCrop
from .guard import BCVMaskGuard, BCVPoseGuard, _final_widgets, _reference_mask
from .pose import VITPOSE, BCVPoseDetection
from .sam3_1_multiplex import BCVSAM3VideoTrack
from .unused_outputs import LINK_INPUTS, drop_unwanted, heavy_wanted, wants


class BCVWanAnimatePreprocess:
    @classmethod
    def INPUT_TYPES(cls):
        from ..libs.mask import FinalMaskConfig

        pose_types = BCVPoseDetection.INPUT_TYPES()
        sam3_types = BCVSAM3VideoTrack.INPUT_TYPES()
        pose = pose_types["required"]
        sam3 = sam3_types["required"]
        face = BCVFaceCrop.INPUT_TYPES()["required"]
        return {
            "required": {
                "images": ("IMAGE",),
                **{name: pose[name] for name in ("body_stick_width", "hand_stick_width", "draw_head", "draw_threshold")},
                "face_padding": face["face_padding"],
                **{name: sam3[name] for name in ("mode", "prompt")},
            },
            "optional": {
                "pose_config": pose_types["optional"]["pose_config"],
                "sam3_config": sam3_types["optional"]["sam3_config"],
                # after the widgets a workflow saved before them has, so it keeps their values, runs
                # ViTPose-H and makes the final mask at the defaults
                "pose_model": pose_types["optional"]["pose_model"],
                **config_inputs(FinalMaskConfig),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "POSEDATA", "BBOX", "STRING", "BBOX", "MASK", "IMAGE")
    RETURN_NAMES = ("pose_images", "face_images", "mask", "pose_data", "bboxes", "key_frame_body_points", "face_bboxes",
                    "final_mask", "bg_images")
    # not computed when nothing links them (nodes/unused_outputs.py); mask, which the final mask is made
    # of, and final_mask, which bg_images is cut by, are only dropped at return when another needs them
    HEAVY_OUTPUTS = ("pose_images", "face_images", "mask", "final_mask", "bg_images")
    FUNCTION = "process"
    CATEGORY = "BCVideoNodes/Wan/Animate"
    DESCRIPTION = "The whole WanAnimate preprocess in one node: Pose Detection (with its pose_model), SAM 3.1 Multiplex Video Track and Face Crop chained, computing exactly what the three nodes compute when wired by hand. In prompt mode the mask comes from the text prompt alone (one track, repaired from the mask alone where it lost a part of her for good or dropped one for up to 8 frames); in box_keypoint mode from the pose, with no extra boxes or points; in prompt_pose mode from the text prompt, with the pose's drawn keypoints as points on each frame where the track lost a whole forearm-and-hand or lower leg, and points inside a hand-sized region of her the mask drops for up to 8 frames between two that hold it. final_mask is the mask grown by grow steps of the 3 x 3 cross and cut into blocks of about block_size px (ComfyUI-BCNodes' MaskGrow with blur 0, then BlockifyMask), and bg_images the frames with the final mask painted black (ComfyUI-BCNodes' Draw Mask On Image, colour 0, 0, 0): the character mask and the background video of a replacement run. The models are downloaded on first use. Feed it frames already at the generation size."

    def process(self, images, body_stick_width, hand_stick_width, draw_head, draw_threshold, face_padding,
                mode, prompt, pose_config=None, sam3_config=None, pose_model=VITPOSE, prompt_graph=None, unique_id=None,
                **widgets):
        """`widgets`: grow and block_size (libs.mask.FinalMaskConfig), its defaults when left out."""
        import torch

        from ..libs.mask import final_mask, painted_black

        wanted = heavy_wanted(type(self), prompt_graph, unique_id)
        pose_images, pose_data, bboxes, key_points = BCVPoseDetection().detect(
            images, body_stick_width, hand_stick_width, draw_head, draw_threshold, pose_config=pose_config,
            pose_model=pose_model, wanted=wanted)
        from ..pipelines.sam3_1_multiplex import track as sam3

        finals = wants(wanted, "final_mask") or wants(wanted, "bg_images")
        if wants(wanted, "mask") or finals:
            # prompt mode segments from the text alone; pose_data is connected only where it is read
            reads_pose = mode != sam3.MODE_PROMPT
            (mask,) = BCVSAM3VideoTrack().track(images, mode, prompt, 1, -1, pose_data=pose_data if reads_pose else None,
                                                sam3_config=sam3_config)
        else:
            mask = torch.empty((0, *images.shape[1:3]))  # SAM's float mask, no frame of it tracked
        face_images, face_bboxes = BCVFaceCrop().crop(images, pose_data, face_padding, wanted=wanted)
        final = final_mask(mask, **_final_widgets(widgets)) if finals else torch.empty((0, *images.shape[1:3]))
        background = painted_black(images, final) if wants(wanted, "bg_images") else images.new_empty((0, *images.shape[1:]))
        return drop_unwanted(type(self), (pose_images, face_images, mask, pose_data, bboxes, key_points, face_bboxes, final,
                                          background), wanted)


FINAL_MASK_TOOLTIP = "The final mask the sampler gets: WanAnimate Preprocess's final_mask (its mask grown by grow and blockified by block_size: ComfyUI-BCNodes' MaskGrow with blur 0, then BlockifyMask), at the size the pose was found on; this node's grow and block_size must be the preprocess's. The raw preprocess mask goes to the Mask Guard."


class BCVWanAnimatePreprocessGuard:
    @classmethod
    def INPUT_TYPES(cls):
        from ..libs.mask import FinalMaskConfig
        from ..pipelines import guard

        pose = BCVPoseGuard.INPUT_TYPES()["required"]
        mask_types = BCVMaskGuard.INPUT_TYPES()
        mask = mask_types["required"]
        return {
            "required": {
                "mask": (mask["mask"][0], {"tooltip": FINAL_MASK_TOOLTIP}),
                "pose_data": pose["pose_data"],
                "mask_guard": mask["mask_guard"],
                **{k: v for k, v in pose.items() if k != "pose_data"},
                **{k: v for k, v in mask.items() if k not in ("mask", "mask_guard")},
            },
            # the Mask Guard's reference check, its threshold the final mask's (pose_data is required here),
            # then the final mask's widgets, the preprocess's: the final mask is blockified
            "optional": {**config_inputs(guard.FinalReferenceGuardConfig),
                         "reference_image": mask_types["optional"]["reference_image"],
                         **config_inputs(FinalMaskConfig)},
        }

    RETURN_TYPES = ("MASK", "POSEDATA", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "pose_data", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = "BCVideoNodes/Wan/Animate"
    DESCRIPTION = "Pose Guard and Mask Guard in one node, with one report: checks the pose and the mask of WanAnimate Preprocess frame by frame. Wire it between the preprocess and the sampler, on the final mask the sampler gets (WanAnimate Preprocess's final_mask: the preprocess mask grown by grow and blockified by block_size, which must be the preprocess's): background attached to the body, detached pieces and dropped regions are measured allowing for its blocks; the box-based checks and the keypoints inside the mask read the final as it is. The mask fails when it is empty on a frame with a person, torn (a detached piece of 5% of her or more), leaves the head out (the drawn nose, or head_out_eyes_ears of the drawn eyes and ears, default 2) or a whole forearm-and-hand or lower leg, or drops a hand-sized region of her (a whole block, holding a drawn keypoint on every frame it is gone); a failed check stops the workflow with the report when mask_guard is on. The pose checks are warnings, and warnings never stop; so is, with reference_image connected, a reference not placed like the first frame (the character SAM 3.1 Multiplex finds on it, placed as the Wan Animate node places the reference and grown and blockified as the final mask, overlapping the mask on frame 0 by an IoU below min_reference_iou). 'metrics' has every measurement per frame and 'timeline' plots them."

    def check(self, mask, pose_data, mask_guard, reference_image=None, **thresholds):
        from ..pipelines import guard

        # each group measures without stopping; the combined report decides
        _, _, pose_metrics, _ = guard.check_pose(pose_data, _config(guard.PoseGuardConfig, thresholds), stop_on_fail=False)
        _, _, mask_metrics, _ = guard.check_mask(mask, pose_data, _config(guard.MaskGuardConfig, thresholds), enabled=mask_guard,
                                                 stop_on_fail=False, final=True, reference=_reference_mask(reference_image),
                                                 reference_config=_config(guard.FinalReferenceGuardConfig, thresholds),
                                                 **_final_widgets(thresholds))
        report, metrics, timeline = guard.combine_guards(pose_metrics, mask_metrics)
        return (mask, pose_data, report, metrics, timeline)
