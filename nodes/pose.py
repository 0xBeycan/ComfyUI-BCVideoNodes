"""Pose Config and Pose Detection, and the pose models its `pose_model` widget offers."""

from .common import PREPROCESS, _ConfigNode
from .unused_outputs import LINK_INPUTS, drop_unwanted, heavy_wanted, wants

# Pose Detection's optional width and height: the size the pose images are drawn at.
MAX_SIZE = 16384
SIZE_TOOLTIP = ("Optional, with the other one: the pose images are drawn at width x height directly, the keypoints scaled "
                "to it (a -1 stick width picked from that size); a frame of another aspect is cut centred first, as core's "
                "ControlNet cuts a hint of another aspect ('center'). Not connected: drawn at the frame size. pose_data, "
                "bboxes and key_frame_body_points stay at the frame size.")

# Pose Detection's `pose_model` widget, and the preprocess wrappers': ViTPose-H, or a Sapiens2 model
# by size and precision, the largest first. Each is the name of a pose estimator models/vitpose or
# models/sapiens2 registers (not imported here: the lazy rule; the tests tie the two lists).
VITPOSE = "ViTPose-H"
POSE_MODELS = (VITPOSE, *(f"Sapiens2 {model}" for model in (
    "5b int8 convrot", "5b bf16", "1b int8 convrot", "1b bf16", "0.8b int8 convrot", "0.8b bf16", "0.4b int8 convrot",
    "0.4b bf16")))
POSE_MODEL_TOOLTIP = ("The pose model: ViTPose-H, or a Sapiens2 model (body, feet and hands from Sapiens2 on its own "
                      "1024x768 crop, the 68 face points from ViTPose-H on the same box). int8 convrot is the int8 "
                      "ConvRot quantized file, computing in bf16. The models are downloaded on first use.")


def _draw_size(width, height):
    """The (width, height) the pose images are drawn at from the optional inputs, None (the frame
    size) when neither is connected."""
    if width is None and height is None:
        return None
    if width is None or height is None:
        raise ValueError("The pose images' width and height go together: connect both, or neither to draw the pose "
                         "images at the frame size.")
    if width < 1 or height < 1:
        raise ValueError(f"The pose images cannot be drawn at {width}x{height}; connect a width and a height of 1 or more.")
    return int(width), int(height)


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
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks in the pose images; 0 leaves the body out, -1 picks it from the size the pose images are drawn at: max(int(min(H, W) / 200) - 1, 1), the official Wan width (2 at 720p)"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks in the pose images; 0 leaves the hands out, -1 picks it from the size the pose images are drawn at: half the body's -1 width, at least 1 (1 at 720p)"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw head keypoints"}),
                "draw_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "A limb is drawn when both its ends reach this confidence; key_frame_body_points uses the same threshold. Carried in pose_data: the guards count the keypoints that reach it, and SAM 3.1 Multiplex prompt_pose picks its positive points from the keypoints drawn at it (with the Pose Config draw rules). SAM 3.1 Multiplex box_keypoint mode reads pose_config.min_keypoint_conf instead"}),
            },
            "optional": {
                "bboxes": ("BBOX", {"tooltip": "Person boxes (x1, y1, x2, y2), one per frame or one for all. When connected the detector does not run and pose_config.detection_threshold is ignored; the boxes still go through pose_config.box_window (widened) and pose_config.edge_snap (snapped to a frame edge they nearly touch), both off by default."}),
                "pose_config": ("POSE_CONFIG", {"tooltip": "Overrides from Pose Config; the measured defaults without it. Its min_keypoint_conf travels in pose_data to SAM 3.1 Multiplex."}),
                # sockets, not widgets: a workflow saved before them keeps every widget value
                "width": ("INT", {"forceInput": True, "min": 1, "max": MAX_SIZE, "tooltip": SIZE_TOOLTIP}),
                "height": ("INT", {"forceInput": True, "min": 1, "max": MAX_SIZE, "tooltip": SIZE_TOOLTIP}),
                # the last widget, so a workflow saved before it keeps its widget values and runs ViTPose-H
                "pose_model": (list(POSE_MODELS), {"default": VITPOSE, "tooltip": POSE_MODEL_TOOLTIP}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA", "BBOX", "STRING")
    RETURN_NAMES = ("pose_images", "pose_data", "bboxes", "key_frame_body_points")
    # not drawn when nothing links them (nodes/unused_outputs.py)
    HEAVY_OUTPUTS = ("pose_images",)
    FUNCTION = "detect"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Wholebody pose on every frame: YOLOv10x finds the person (skipped when bboxes are connected) and the pose model gives the 133 keypoints: ViTPose-H, or with a Sapiens2 pose_model Sapiens2 (Meta) for the body, the feet and the hands, on its own 1024x768 crop, with the 68 face keypoints from ViTPose-H on the same box, so Face Crop and the guards read the face ViTPose-H gives. The pose images are drawn at the frame size, or at width x height when both are connected. key_frame_body_points is frame 0's confident body keypoints as a points JSON string (KJNodes PointsEditor format). The models are downloaded on first use."

    def detect(self, images, body_stick_width, hand_stick_width, draw_head, draw_threshold, bboxes=None, pose_config=None,
               width=None, height=None, pose_model=VITPOSE, prompt_graph=None, unique_id=None, *, wanted=None):
        """`wanted`: the heavy outputs a wrapper needs, in place of the link stamp's."""
        from ..models.common import loader

        if pose_model not in POSE_MODELS:
            raise ValueError(f"pose_model {pose_model!r}: expected one of {', '.join(POSE_MODELS)}")
        size = _draw_size(width, height)
        wanted = heavy_wanted(type(self), prompt_graph, unique_id, wanted)
        # supplied boxes skip the detector, so it is not loaded either; ViTPose-H also gives Sapiens2's face
        detector, vitpose = loader.load_pose_models(detector=bboxes is None)
        widgets = dict(bboxes=bboxes, config=pose_config, body_stick_width=body_stick_width,
                       hand_stick_width=hand_stick_width, draw_head=draw_head, draw_threshold=draw_threshold,
                       draw_images=wants(wanted, "pose_images"), size=size)
        if pose_model == VITPOSE:
            from ..pipelines import pose

            outputs = pose.pose_detection(images, detector, vitpose, **widgets)
        else:
            sapiens2 = loader.load_pose_estimator(pose_model)
            from ..pipelines import sapiens2_pose

            outputs = sapiens2_pose.sapiens2_pose(images, detector, sapiens2, vitpose, **widgets)
        return drop_unwanted(type(self), outputs, wanted)
