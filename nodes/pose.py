"""Pose Config and Pose Detection."""

from .common import PREPROCESS, _ConfigNode


class BCVPoseConfig(_ConfigNode):
    RETURN_TYPES = ("POSE_CONFIG",)
    RETURN_NAMES = ("pose_config",)
    DESCRIPTION = "Overrides the Pose Detection tunables. Without it Pose Detection runs with the measured defaults, which are the values shown here. min_keypoint_conf is carried in pose_data and is the keypoint threshold SAM 3.1 Multiplex box_keypoint mode uses; the guards count what Pose Detection draws (draw_threshold)."

    @classmethod
    def _config_class(cls):
        from ..pipelines.pose import PoseConfig

        return PoseConfig


class BCVPoseDetection:
    @classmethod
    def INPUT_TYPES(cls):
        from ..models.common import loader

        return {
            "required": {
                "images": ("IMAGE",),
                "pose_model": (loader.pose_model_names(), {"tooltip": "The wholebody pose model (133 keypoints). ViTPose-H for complex motion: turns, back views, motion blur, close-ups. RTMW-l only for simple motion that stays in the frame: it loses the body under blur, draws a face on the back of the head and pushes limbs to the frame edge, and the diffusion follows the pose it is given. Its speed gain is a few seconds per clip; with SAM 3.1 Multiplex box_keypoint mode it can make SAM 3.1 Multiplex much slower (more re-seeds). The file is downloaded into models/detection on first use."}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks in the pose images; 0 leaves the body out, -1 picks it from the frame size"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks in the pose images; 0 leaves the hands out, -1 picks it from the frame size"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw head keypoints"}),
                "draw_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "A limb is drawn when both its ends reach this confidence; key_frame_body_points uses the same threshold. Carried in pose_data: the guards count what is drawn. SAM 3.1 Multiplex box_keypoint mode reads pose_config.min_keypoint_conf instead"}),
            },
            "optional": {
                "bboxes": ("BBOX", {"tooltip": "Person boxes (x1, y1, x2, y2), one per frame or one for all. When connected the detector does not run and pose_config.detection_threshold is ignored; box_window and the edge snap still apply."}),
                "pose_config": ("POSE_CONFIG", {"tooltip": "Overrides from Pose Config; the measured defaults without it. Its min_keypoint_conf travels in pose_data to SAM 3.1 Multiplex."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA", "BBOX", "STRING")
    RETURN_NAMES = ("pose_images", "pose_data", "bboxes", "key_frame_body_points")
    FUNCTION = "detect"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Wholebody pose on every frame: YOLOv10x finds the person (skipped when bboxes are connected), ViTPose-H or RTMW-l gives the 133 keypoints, which are read against the frames around them, and the pose images are drawn at the frame size. key_frame_body_points is frame 0's confident body keypoints as a points JSON string (KJNodes PointsEditor format). The models are downloaded on first use."

    def detect(self, images, pose_model, body_stick_width, hand_stick_width, draw_head, draw_threshold, bboxes=None, pose_config=None):
        from ..models.common import loader

        # supplied boxes skip the detector, so it is not loaded either
        detector, model = loader.load_pose_models(pose_model, detector=bboxes is None)
        from ..pipelines import pose

        return tuple(pose.pose_detection(images, detector, model, bboxes=bboxes, config=pose_config,
                                         body_stick_width=body_stick_width, hand_stick_width=hand_stick_width,
                                         draw_head=draw_head, draw_threshold=draw_threshold))
