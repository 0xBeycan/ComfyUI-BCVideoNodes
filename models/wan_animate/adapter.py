"""WanAnimateToVideo (Wan 2.2 Animate) in the long-video loop: the overlap is its
continue_motion_max_frames widget, and its concat mask rows are realigned when a character mask
is connected (mask_repair.py)."""

import logging

from ...libs.chunking import overlap_for_motion_frames
from ..common.animate import AnimateAdapter
from .mask_repair import fix_replacement_mask, replacement_mask_rows


class WanAnimateAdapter(AnimateAdapter):
    ANIMATE_NODE = "WanAnimateToVideo"

    def prepare(self, animate_cls, animate_inputs):
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
