"""WanAnimateToVideo (Wan 2.2 Animate) in the long-video loop: the overlap is its
continue_motion_max_frames widget, and its concat mask rows are realigned when a character mask
is connected (mask_repair.py)."""

import logging

from ...libs.chunking import overlap_for_motion_frames
from ..common.animate import AnimateAdapter
from .mask_repair import fix_replacement_mask, replacement_mask_rows


class WanAnimateAdapter(AnimateAdapter):
    ANIMATE_NODE = "WanAnimateToVideo"

    def prepare(self, animate_cls, animate_inputs, reference_image, width, height, frames_per_chunk):
        # The overlap is a widget here (continue_motion_max_frames): the core
        # node keeps that many frames of continue_motion, moves the offset back
        # by the same amount and trims their decoded span off again. Off the
        # 4k+1 grid the trimmed span is shorter than the offset move and a few
        # frames repeat at the seam, so snap before handing it over.
        wanted = int(animate_inputs["continue_motion_max_frames"])
        motion_frames = overlap_for_motion_frames(wanted)
        if motion_frames != wanted:
            logging.info("[%s] continue_motion_max_frames %d is not on the 4k+1 grid; using %d.", self.node_name, wanted, motion_frames)
            animate_inputs["continue_motion_max_frames"] = motion_frames
        if animate_inputs.get("character_mask") is not None:
            logging.info("[%s] character_mask connected: realigning the core node's mask rows (see _fix_replacement_mask).", self.node_name)
        return motion_frames

    def check_videos(self, pose_video, animate_inputs):
        # The mask says where the character goes in each background frame, so a mask video must
        # be as long as the background.
        character_mask, background = animate_inputs.get("character_mask"), animate_inputs.get("background_video")
        if (character_mask is not None and background is not None and character_mask.ndim >= 3
                and character_mask.shape[0] > 1 and character_mask.shape[0] != background.shape[0]):
            raise ValueError("character_mask has {} frames but background_video has {}: the mask marks where the character "
                             "goes in each background frame, so connect the two from the same video.".format(
                                 int(character_mask.shape[0]), int(background.shape[0])))

    def after_animate(self, positive, negative, trim_image, length, offset, animate_inputs):
        character_mask = animate_inputs.get("character_mask")
        if character_mask is None:
            return None  # without a character mask core's rows are already right
        mask = next((entry[1]["concat_mask"] for entry in positive if len(entry) > 1 and isinstance(entry[1], dict) and "concat_mask" in entry[1]), None)
        if mask is None:
            return None
        # core moves the offset back by the seed frames before it seeks the mask
        rows = replacement_mask_rows(character_mask, max(0, offset - trim_image), length, trim_image, mask.shape[-2], mask.shape[-1])
        if rows is None:
            return None  # mask does not reach this window: core leaves its rows alone, so do we
        seen = set()
        fix_replacement_mask(positive, rows, seen)
        fix_replacement_mask(negative, rows, seen)
