"""SAM 3.1 Multiplex Config and SAM 3.1 Multiplex Video Track."""

from .common import PREPROCESS, _ConfigNode


class BCVSAM3Config(_ConfigNode):
    RETURN_TYPES = ("SAM3_CONFIG",)
    RETURN_NAMES = ("sam3_config",)
    DESCRIPTION = "Overrides the SAM 3.1 Multiplex Video Track tunables. Without it the node runs with the measured defaults, which are the values shown here. Each tooltip starts with the mode that reads the field; a changed field the run does not read is named in the console. prompt_pose runs prompt mode's track first, so it reads the [prompt] and [prompt, max_objects 1] fields too."

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
                "mode": (list(sam3.MODES), {"default": sam3.MODE_PROMPT, "tooltip": "prompt: SAM 3.1 Multiplex finds the person from the text prompt alone, then repairs its track from the mask alone: where the track lost a part of her for good (a piece of 8% of the mask or more gone, the area never back above 92% of the frame before's to the end of the stretch), the frames from there to the end of the stretch are tracked again from the mask of the frame before; where the mask drops a part of her for 1-8 frames between two frames that hold it (a hand-sized part holding a whole block of the final mask, grow 10 and blockify 32, on both sides), every frame of the run is refined from up to 16 points inside it and shows its mask and the refine's together. A clip where neither is found gets the track's mask exactly. Limits: a limb the frame edge cuts can be left out, thin hair strands are not followed, a part lost for a while that comes back later is left as tracked. box_keypoint: the person is described by pose_data's box and body keypoints (needs pose_data), the v1 behaviour; limits: frame 0 can take in background around the person, and objects the arm reaches can be pulled into the mask. prompt_pose: prompt mode's track, without its repairs; on every frame where the pose shows a whole forearm-and-hand or lower leg dropped from the mask, far outside it, its drawn keypoints go onto that frame as positive points together with the mask the tracker had on that frame (Meta's point refine on the same object), and the frames that refine can reach are tracked again while the rest keep the track's mask (needs pose_data). Each frame is judged on its own, so a drop over several frames, or from the first tracked frame on, is refined on each of them. It also refines every frame of a run of 1-8 frames where the mask drops a hand-sized region of her that it holds on both sides and the pose has her body in (the Mask Guard's mask_loss on the final mask, grow 10 and blockify 32), from up to 16 points inside that region; a frame both pick gets the pose's points first. A clip where no frame needs points is the track's mask exactly. Nothing is removed on pose grounds. Limits: as prompt's track; only whole-limb drops far outside the mask, and regions dropped for up to 8 frames between two frames that hold them, are recovered. The Mask Guard fails a limb or the head the mask leaves out (mask_limb_out, mask_head_out) and warns on background taken in (mask_attached_leak). Inputs the mode does not read are ignored with one console line, so switching needs no rewiring."}),
                "prompt": ("STRING", {"default": sam3.PROMPT, "tooltip": "[prompt, prompt_pose] what to segment. The defaults were validated with this prompt. Ignored in box_keypoint mode."}),
                "max_objects": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1, "tooltip": "[prompt] how many tracks may be born and kept; above 1 the [prompt, max_objects > 1] config fields apply. Ignored in box_keypoint and prompt_pose modes, which track one person."}),
                "object_index": ("INT", {"default": -1, "min": -1, "max": 15, "step": 1, "tooltip": "[prompt] which tracked object the mask is: -1 the union of every tracked object, k object k (numbered from 0). Ignored in box_keypoint and prompt_pose modes."}),
            },
            "optional": {
                "pose_data": ("POSEDATA", {"tooltip": "[box_keypoint, prompt_pose] box_keypoint: the boxes and keypoints the prompt is built from, and the keypoint threshold (Pose Config's min_keypoint_conf). prompt_pose: the drawn keypoints (pose_data's draw_threshold and draw rules) that may become positive points. Ignored in prompt mode."}),
                "bboxes": ("BBOX", {"tooltip": "[box_keypoint] replaces pose_data's person boxes, used as given (every frame counts as detected). Ignored in prompt and prompt_pose modes. If the boxes come from Pose Detection, connect pose_data instead: its bboxes output marks frames with no detected person as full-frame boxes."}),
                "positive_coords": ("STRING", {"forceInput": True, "tooltip": "[box_keypoint] extra positive points on frame 0, points JSON (KJNodes PointsEditor / easy-sam3). They win: a derived negative within 4% of the box diagonal is dropped. Ignored in prompt and prompt_pose modes. In box_keypoint mode, connecting key_frame_body_points here only repeats the automatic points."}),
                "negative_coords": ("STRING", {"forceInput": True, "tooltip": "[box_keypoint] extra negative points on frame 0, same format. They win: a keypoint or body point within 4% of the box diagonal is dropped. Ignored in prompt and prompt_pose modes."}),
                "sam3_config": ("SAM3_CONFIG", {"tooltip": "Overrides from SAM 3.1 Multiplex Config; the measured defaults without it. Only the fields tagged with the current mode are read; prompt_pose also reads the [prompt] ones."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "track"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Segments the person on every frame with SAM 3.1 Multiplex and its tracker memory, from a text prompt (prompt, which then repairs its own track from the mask alone: the frames after a part of her it lost for good are tracked again from the mask before the loss, and a part the mask drops for up to 8 frames is refined from points inside it), from Pose Detection's boxes and keypoints, or from the text prompt with the pose's drawn keypoints as points where the track lost a limb, and points inside a hand-sized region the mask drops for up to 8 frames (prompt_pose, on the track without prompt mode's repairs). The mask covers every frame, including the ones before the person was found. Mode limits: prompt can leave out a limb the frame edge cuts and thin hair, and a part it loses for a while and finds again later; box_keypoint can take in background on frame 0 and objects the arm reaches; prompt_pose is prompt mode's track and recovers only a whole forearm-and-hand or lower leg the track lost, on each frame it is lost on, and a region of her the mask drops for up to 8 frames between two that hold it. The checkpoint is ComfyUI's own SAM 3.1 Multiplex, downloaded into models/checkpoints on first use."

    def track(self, images, mode, prompt, max_objects, object_index, pose_data=None, bboxes=None, positive_coords=None,
              negative_coords=None, sam3_config=None):
        from ..pipelines.sam3_1_multiplex import track as sam3

        mask = sam3.track(sam3.load_sam3_1_multiplex(), images, pose_data=pose_data, bboxes=bboxes, positive_coords=positive_coords,
                          negative_coords=negative_coords, mode=mode, prompt=prompt, max_objects=max_objects,
                          object_index=object_index, config=sam3_config)
        return (mask,)
