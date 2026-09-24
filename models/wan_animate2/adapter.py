"""WanAnimate2ToVideo (Wan Animate 2) in the long-video loop: the overlap is the node's
CONTINUE_MOTION_FRAMES, attn_log_scale installs the seed-frame attention bias (attention.py), and
a connected clip_vision re-encodes the pose CLIP embedding per chunk."""

import logging

from ...libs.chunking import overlap_for_motion_frames
from ..common.animate import AnimateAdapter, check_pose_percents
from ..common.core_nodes import clip_vision_encode, node_class
from .attention import seed_frame_attention_bias


def continue_motion_frames(animate_cls):
    """The frames of continue_motion the core node keeps: its CONTINUE_MOTION_FRAMES, 1 when it has none."""
    return int(getattr(animate_cls, "CONTINUE_MOTION_FRAMES", 1))


class WanAnimate2Adapter(AnimateAdapter):
    ANIMATE_NODE = "WanAnimate2ToVideo"

    def prepare(self, animate_cls, animate_inputs, reference_image, width, height, frames_per_chunk):
        check_pose_percents(animate_inputs["pose_start_percent"], animate_inputs["pose_end_percent"])
        self._log_scale = float(animate_inputs.pop("attn_log_scale", -1.3))
        self._clip_vision = animate_inputs.pop("clip_vision", None)
        if self._clip_vision is not None:
            animate_inputs.pop("clip_vision_output_pose", None)
            logging.info("[%s] clip_vision connected: pose CLIP embedding is re-encoded per chunk.", self.node_name)
        # The node keeps the last CONTINUE_MOTION_FRAMES frames of continue_motion
        # and trims their decoded span back off every chained chunk; that span
        # is the overlap. Read from the class so a core change is picked up.
        return overlap_for_motion_frames(continue_motion_frames(animate_cls))

    def patch_model(self, patched, animate_inputs):
        if self._log_scale == 0.0:
            return patched
        patched = patched.clone()
        patched.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = seed_frame_attention_bias(self._log_scale)
        logging.info("[%s] attn_log_scale %.2f on the seed frame (official distilled config: -1.3).", self.node_name, self._log_scale)
        return patched

    def chunk_inputs(self, index, offset, anchor, pose_video, animate_inputs):
        if self._clip_vision is None:
            return {}
        # the core node moves the offset back by the seed frame before it reads the pose video; encode that same first frame
        seed = 0 if anchor is None else min(int(anchor.shape[0]), continue_motion_frames(node_class(self.ANIMATE_NODE)))
        first = min(max(0, offset - seed), int(pose_video.shape[0]) - 1)
        return {"clip_vision_output_pose": clip_vision_encode(self._clip_vision, pose_video[first:first + 1])}
