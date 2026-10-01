"""Pose Config, Pose Detection and Sapiens2 Pose, and the pose model the preprocess wrappers pick."""

from .common import PREPROCESS, _ConfigNode


class BCVPoseConfig(_ConfigNode):
    RETURN_TYPES = ("POSE_CONFIG",)
    RETURN_NAMES = ("pose_config",)
    DESCRIPTION = "Overrides the Pose Detection tunables. Without it Pose Detection runs with the measured defaults, which are the values shown here. min_keypoint_conf is carried in pose_data and is the keypoint threshold SAM 3.1 Multiplex box_keypoint mode uses; the guards count the keypoints that reach Pose Detection's draw_threshold."

    @classmethod
    def _config_class(cls):
        from ..pipelines.pose import PoseConfig

        return PoseConfig


class BCVPoseDetection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks in the pose images; 0 leaves the body out, -1 picks it from the input frame size: max(int(min(H, W) / 200) - 1, 1), the official Wan width (2 at 720p)"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks in the pose images; 0 leaves the hands out, -1 picks it from the input frame size: half the body's -1 width, at least 1 (1 at 720p)"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw head keypoints"}),
                "draw_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "A limb is drawn when both its ends reach this confidence; key_frame_body_points uses the same threshold. Carried in pose_data: the guards count the keypoints that reach it, and SAM 3.1 Multiplex prompt_pose picks its positive points from the keypoints drawn at it (with the Pose Config draw rules). SAM 3.1 Multiplex box_keypoint mode reads pose_config.min_keypoint_conf instead"}),
            },
            "optional": {
                "bboxes": ("BBOX", {"tooltip": "Person boxes (x1, y1, x2, y2), one per frame or one for all. When connected the detector does not run and pose_config.detection_threshold is ignored; the boxes still go through pose_config.box_window (widened) and pose_config.edge_snap (snapped to a frame edge they nearly touch), both off by default."}),
                "pose_config": ("POSE_CONFIG", {"tooltip": "Overrides from Pose Config; the measured defaults without it. Its min_keypoint_conf travels in pose_data to SAM 3.1 Multiplex."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA", "BBOX", "STRING")
    RETURN_NAMES = ("pose_images", "pose_data", "bboxes", "key_frame_body_points")
    FUNCTION = "detect"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Wholebody pose on every frame: YOLOv10x finds the person (skipped when bboxes are connected), ViTPose-H gives the 133 keypoints, and the pose images are drawn at the frame size. key_frame_body_points is frame 0's confident body keypoints as a points JSON string (KJNodes PointsEditor format). The models are downloaded on first use."

    def detect(self, images, body_stick_width, hand_stick_width, draw_head, draw_threshold, bboxes=None, pose_config=None):
        from ..models.common import loader

        # supplied boxes skip the detector, so it is not loaded either
        detector, model = loader.load_pose_models(detector=bboxes is None)
        from ..pipelines import pose

        return tuple(pose.pose_detection(images, detector, model, bboxes=bboxes, config=pose_config,
                                         body_stick_width=body_stick_width, hand_stick_width=hand_stick_width,
                                         draw_head=draw_head, draw_threshold=draw_threshold))


# The Sapiens2 Pose `model` widget: size and precision, the largest model first. models/sapiens2
# registers each as the pose estimator "Sapiens2 <value>" (not imported here: the lazy rule; the
# tests tie the two lists).
SAPIENS2_MODELS = ("5b int8 convrot", "5b bf16", "1b int8 convrot", "1b bf16", "0.8b int8 convrot", "0.8b bf16",
                   "0.4b int8 convrot", "0.4b bf16")
SAPIENS2 = "Sapiens2 "
VITPOSE = "ViTPose-H"
# the preprocess wrappers' `pose_model` widget: Pose Detection's ViTPose-H, or a Sapiens2 Pose model
POSE_MODELS = (VITPOSE, *(SAPIENS2 + model for model in SAPIENS2_MODELS))
POSE_MODEL_TOOLTIP = ("The pose model: ViTPose-H runs Pose Detection; a Sapiens2 model runs Sapiens2 Pose (body, feet and "
                      "hands from Sapiens2, the 68 face points from ViTPose-H on the same box). int8 convrot is the "
                      "int8 ConvRot quantized file, computing in bf16. The models are downloaded on first use.")


class BCVSapiens2Pose:
    @classmethod
    def INPUT_TYPES(cls):
        pose = BCVPoseDetection.INPUT_TYPES()
        return {
            "required": {
                "images": pose["required"]["images"],
                "model": (list(SAPIENS2_MODELS), {"default": SAPIENS2_MODELS[0], "tooltip": "The Sapiens2 pose model: its size, and bf16 weights or int8 convrot (int8 ConvRot quantized) weights computing in bf16. Downloaded into models/detection on first use, from beycanai/sapiens2-convrot"}),
                **{name: kind for name, kind in pose["required"].items() if name != "images"},
            },
            "optional": pose["optional"],
        }

    RETURN_TYPES = BCVPoseDetection.RETURN_TYPES
    RETURN_NAMES = BCVPoseDetection.RETURN_NAMES
    FUNCTION = "detect"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Pose Detection with Sapiens2 (Meta) for the body, the feet and the hands: YOLOv10x finds the person (skipped when bboxes are connected), Sapiens2 gives the body, feet and hand keypoints on its own 1024x768 crop, and the 68 face keypoints come from ViTPose-H on the same box, so Face Crop and the guards read the face Pose Detection gives. The outputs, pose_config and the drawing are Pose Detection's. The models are downloaded on first use."

    def detect(self, images, model, body_stick_width, hand_stick_width, draw_head, draw_threshold, bboxes=None,
               pose_config=None):
        from ..models.common import loader

        # supplied boxes skip the detector, so it is not loaded either; ViTPose-H gives the face
        detector, vitpose = loader.load_pose_models(detector=bboxes is None)
        sapiens2 = loader.load_pose_estimator(SAPIENS2 + model)
        from ..pipelines import sapiens2_pose

        return tuple(sapiens2_pose.sapiens2_pose(images, detector, sapiens2, vitpose, bboxes=bboxes, config=pose_config,
                                                 body_stick_width=body_stick_width, hand_stick_width=hand_stick_width,
                                                 draw_head=draw_head, draw_threshold=draw_threshold))


def detect_pose(pose_model, images, body_stick_width, hand_stick_width, draw_head, draw_threshold, pose_config=None):
    """What the preprocess wrappers run for their `pose_model` widget (a POSE_MODELS value): Pose
    Detection for ViTPose-H, Sapiens2 Pose with that model otherwise."""
    if pose_model not in POSE_MODELS:
        raise ValueError(f"pose_model {pose_model!r}: expected one of {', '.join(POSE_MODELS)}")
    widgets = dict(body_stick_width=body_stick_width, hand_stick_width=hand_stick_width, draw_head=draw_head,
                   draw_threshold=draw_threshold, pose_config=pose_config)
    if pose_model == VITPOSE:
        return BCVPoseDetection().detect(images, **widgets)
    return BCVSapiens2Pose().detect(images, pose_model[len(SAPIENS2):], **widgets)
