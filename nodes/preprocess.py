"""The Wan Animate wrappers: WanAnimate Preprocess and WanAnimate Preprocess Guard. Each one
calls the individual nodes, so it computes exactly what the chained nodes compute."""

from .common import _config
from .face import BCVFaceCrop
from .guard import BCVMaskGuard, BCVPoseGuard
from .pose import BCVPoseDetection
from .sam3_1_multiplex import BCVSAM3VideoTrack


class BCVWanAnimatePreprocess:
    @classmethod
    def INPUT_TYPES(cls):
        pose_types = BCVPoseDetection.INPUT_TYPES()
        sam3_types = BCVSAM3VideoTrack.INPUT_TYPES()
        pose = pose_types["required"]
        sam3 = sam3_types["required"]
        face = BCVFaceCrop.INPUT_TYPES()["required"]
        return {
            "required": {
                "images": ("IMAGE",),
                **{name: pose[name] for name in ("pose_model", "body_stick_width", "hand_stick_width", "draw_head", "draw_threshold")},
                "face_padding": face["face_padding"],
                **{name: sam3[name] for name in ("mode", "prompt")},
            },
            "optional": {
                "pose_config": pose_types["optional"]["pose_config"],
                "sam3_config": sam3_types["optional"]["sam3_config"],
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "POSEDATA", "BBOX", "STRING", "BBOX")
    RETURN_NAMES = ("pose_images", "face_images", "mask", "pose_data", "bboxes", "key_frame_body_points", "face_bboxes")
    FUNCTION = "process"
    CATEGORY = "BCVideoNodes/Wan/Animate"
    DESCRIPTION = "The whole WanAnimate preprocess in one node: Pose Detection, SAM 3.1 Multiplex Video Track and Face Crop chained, computing exactly what the three nodes compute when wired by hand. In prompt mode the mask comes from the text prompt alone (one track, the union mask); in box_keypoint mode from the pose, with no extra boxes or points. The models are downloaded on first use. Feed it frames already at the generation size."

    def process(self, images, pose_model, body_stick_width, hand_stick_width, draw_head, draw_threshold, face_padding,
                mode, prompt, pose_config=None, sam3_config=None):
        pose_images, pose_data, bboxes, key_points = BCVPoseDetection().detect(
            images, pose_model, body_stick_width, hand_stick_width, draw_head, draw_threshold, pose_config=pose_config)
        from ..pipelines.sam3_1_multiplex import track as sam3

        # prompt mode segments from the text alone; pose_data is connected only where it is read
        box_keypoint = mode == sam3.MODE_BOX_KEYPOINT
        (mask,) = BCVSAM3VideoTrack().track(images, mode, prompt, 1, -1, pose_data=pose_data if box_keypoint else None,
                                            sam3_config=sam3_config)
        face_images, face_bboxes = BCVFaceCrop().crop(images, pose_data, face_padding)
        return (pose_images, face_images, mask, pose_data, bboxes, key_points, face_bboxes)


class BCVWanAnimatePreprocessGuard:
    @classmethod
    def INPUT_TYPES(cls):
        pose = BCVPoseGuard.INPUT_TYPES()["required"]
        mask = BCVMaskGuard.INPUT_TYPES()["required"]
        return {
            "required": {
                "mask": mask["mask"],
                "pose_data": pose["pose_data"],
                "pose_guard": pose["pose_guard"],
                "mask_guard": mask["mask_guard"],
                **{k: v for k, v in pose.items() if k not in ("pose_data", "pose_guard")},
                **{k: v for k, v in mask.items() if k not in ("mask", "pose_data", "mask_guard")},
            },
        }

    RETURN_TYPES = ("MASK", "POSEDATA", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "pose_data", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = "BCVideoNodes/Wan/Animate"
    DESCRIPTION = "Pose Guard and Mask Guard in one node, with one report: checks the pose and the mask of WanAnimate Preprocess frame by frame. Wire it between the preprocess and the sampler: a failed check of an enabled guard stops the workflow with the report; 'metrics' has every measurement per frame and 'timeline' plots them."

    def check(self, mask, pose_data, pose_guard, mask_guard, **thresholds):
        from ..pipelines import guard

        # each group measures without stopping; the combined report decides
        _, _, pose_metrics, _ = guard.check_pose(pose_data, _config(guard.PoseGuardConfig, thresholds), enabled=pose_guard,
                                                 stop_on_fail=False)
        _, _, mask_metrics, _ = guard.check_mask(mask, pose_data, _config(guard.MaskGuardConfig, thresholds), enabled=mask_guard,
                                                 stop_on_fail=False)
        report, metrics, timeline = guard.combine_guards(pose_metrics, mask_metrics)
        return (mask, pose_data, report, metrics, timeline)
