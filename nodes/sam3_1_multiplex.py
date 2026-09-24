"""SAM 3.1 Multiplex Config and SAM 3.1 Multiplex Video Track."""

from .common import PREPROCESS, _ConfigNode


class BCVSAM3Config(_ConfigNode):
    RETURN_TYPES = ("SAM3_CONFIG",)
    RETURN_NAMES = ("sam3_config",)
    DESCRIPTION = "Overrides the SAM 3.1 Multiplex Video Track tunables. Without it the node runs with the measured defaults, which are the values shown here. Each tooltip starts with the mode that reads the field; a changed field the run does not read is named in the console."

    @classmethod
    def _config_class(cls):
        from ..pipelines.sam3_1_multiplex.track import SAM3_1MultiplexConfig

        return SAM3_1MultiplexConfig


class BCVSAM3VideoTrack:
    @classmethod
    def INPUT_TYPES(cls):
        from ..pipelines.sam3_1_multiplex import track as sam3

        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (list(sam3.MODES), {"default": sam3.MODE_PROMPT, "tooltip": "prompt: SAM 3.1 Multiplex finds the person from the text prompt alone; limits: a limb the frame edge cuts can be left out, thin hair strands are not followed. box_keypoint: the person is described by pose_data's box and body keypoints (needs pose_data), the v1 behaviour; limits: frame 0 can take in background around the person, and objects the arm reaches can be pulled into the mask. Mask Guard reports both (mask_specks, mask_attached_leak, mask_missing_keypoints). Inputs the mode does not read are ignored with one console line, so switching needs no rewiring."}),
                "prompt": ("STRING", {"default": sam3.PROMPT, "tooltip": "[prompt] what to segment. The thresholds were measured with the default. Ignored in box_keypoint mode."}),
                "max_objects": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1, "tooltip": "[prompt] how many tracks may be born and kept; above 1 the [prompt, max_objects > 1] config fields apply. Ignored in box_keypoint mode, which tracks the one person the pose describes."}),
                "object_index": ("INT", {"default": -1, "min": -1, "max": 15, "step": 1, "tooltip": "[prompt] which tracked object the mask is: -1 the union of every tracked object, k object k (numbered from 0). Ignored in box_keypoint mode."}),
            },
            "optional": {
                "pose_data": ("POSEDATA", {"tooltip": "[box_keypoint] the boxes and keypoints the prompt is built from, and the keypoint threshold (Pose Config's min_keypoint_conf). Ignored in prompt mode."}),
                "bboxes": ("BBOX", {"tooltip": "[box_keypoint] replaces pose_data's person boxes, used as given (every frame counts as detected). Ignored in prompt mode. If the boxes come from Pose Detection, connect pose_data instead: its bboxes output marks frames with no detected person as full-frame boxes."}),
                "positive_coords": ("STRING", {"forceInput": True, "tooltip": "[box_keypoint] extra positive points on frame 0, points JSON (KJNodes PointsEditor / easy-sam3). They win: a derived negative within 4% of the box diagonal is dropped. Ignored in prompt mode. In box_keypoint mode, connecting key_frame_body_points here only repeats the automatic points."}),
                "negative_coords": ("STRING", {"forceInput": True, "tooltip": "[box_keypoint] extra negative points on frame 0, same format. They win: a keypoint or body point within 4% of the box diagonal is dropped. Ignored in prompt mode."}),
                "sam3_config": ("SAM3_CONFIG", {"tooltip": "Overrides from SAM 3.1 Multiplex Config; the measured defaults without it. Only the fields tagged with the current mode are read."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "track"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Segments the person on every frame with SAM 3.1 Multiplex and its tracker memory, from a text prompt or from Pose Detection's boxes and keypoints. The mask covers every frame, including the ones before the person was found. Mode limits: prompt can leave out a limb the frame edge cuts and thin hair; box_keypoint can take in background on frame 0 and objects the arm reaches. The checkpoint is ComfyUI's own SAM 3.1 Multiplex, downloaded into models/checkpoints on first use."

    def track(self, images, mode, prompt, max_objects, object_index, pose_data=None, bboxes=None, positive_coords=None,
              negative_coords=None, sam3_config=None):
        from ..pipelines.sam3_1_multiplex import track as sam3

        mask = sam3.track(sam3.load_sam3_1_multiplex(), images, pose_data=pose_data, bboxes=bboxes, positive_coords=positive_coords,
                          negative_coords=negative_coords, mode=mode, prompt=prompt, max_objects=max_objects,
                          object_index=object_index, config=sam3_config)
        return (mask,)
