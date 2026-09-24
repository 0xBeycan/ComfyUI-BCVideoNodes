"""WanSCAILToVideo (Wan 2.1 SCAIL-2) in the long-video loop.

The core node's chaining contract differs from Wan Animate's: it is seeded with previous_frames
(the last previous_frame_count of them, VAE-encoded into the first latent frames and kept clean
by a noise mask) and returns 4 outputs, with no trim values, so the trim is computed here. Like
the Animate nodes it moves video_frame_offset back by the frames it kept and seeks the pose video
and its colored mask by that offset. SCAIL-2 was trained on 65-81 frame segments, so every chunk
runs at the full frames_per_chunk and the output is cut to total_frames. The reference is
CLIP-encoded once per run; in replacement mode on a black background, as the VAE path gets it.
"""

import logging

import torch

from ...libs.chunking import full_chunk_length, overlap_for_motion_frames, snap_down
from ..common.animate import AnimateAdapter, check_pose_percents
from ..common.core_nodes import clip_vision_encode

SIZE_MULTIPLE = 32  # the pose runs at half resolution through the /16 patch grid
TRAINED_CHUNKS = (65, 81)  # the segment lengths SCAIL-2 was trained on (zai-org/SCAIL-2 issue #16)
# core's thresholds (comfy_extras/nodes_scail.py): a mask colour channel is on above 225/255, and
# the replacement-mode reference keeps the pixels whose mask has any channel above 0.1
ON = 225.0 / 255.0
CHARACTER = 0.1
ANIMATION, REPLACEMENT = "animation", "replacement"
# the reference mask's background per mode: white in animation mode, black in replacement mode
BACKGROUND = {ANIMATION: "white", REPLACEMENT: "black"}


def character_on_black(reference, reference_mask):
    """``reference`` [B, H, W, C] with every pixel outside the character black: the pixels whose
    colored mask (resized nearest to the reference) has no channel above 0.1, the rule core's
    WanSCAILToVideo applies to the VAE reference in replacement mode."""
    import comfy.utils

    height, width = reference.shape[1], reference.shape[2]
    mask = comfy.utils.common_upscale(reference_mask[..., :3].movedim(-1, 1), width, height, "nearest-exact", "center").movedim(1, -1)
    mask = mask[[min(i, mask.shape[0] - 1) for i in range(reference.shape[0])]]
    is_character = (mask.max(dim=-1, keepdim=True).values > CHARACTER).to(reference.dtype)
    return reference * is_character


def mask_convention(reference_mask):
    """The mode a colored reference mask was rendered for, read from the border of its first
    frame: ANIMATION when most border pixels are white, REPLACEMENT when most are black, None when
    neither."""
    frame = reference_mask[0, ..., :3].float()
    border = [frame[0], frame[-1]]
    if frame.shape[0] > 2:
        border += [frame[1:-1, 0], frame[1:-1, -1]]
    pixels = torch.cat(border, dim=0)
    white = float((pixels.min(dim=-1).values > ON).float().mean())
    black = float((pixels.max(dim=-1).values <= CHARACTER).float().mean())
    if white > 0.5:
        return ANIMATION
    if black > 0.5:
        return REPLACEMENT
    return None


class WanSCAIL2Adapter(AnimateAdapter):
    ANIMATE_NODE = "WanSCAILToVideo"
    OUTPUTS = 4  # positive, negative, latent, video_frame_offset
    UPDATE_HINT = "Update ComfyUI: this node needs the {} of SCAIL-2 (previous_frames, pose_video_mask)."
    HELD_VIDEOS = ("pose_video", "pose_video_mask")
    OVERSHOOT = "the last chunk runs the full frames_per_chunk"

    def prepare(self, animate_cls, animate_inputs, reference_image, width, height, frames_per_chunk):
        if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE:
            raise ValueError("width and height must be divisible by 32 for SCAIL-2 (the pose runs at half resolution "
                             "through the /16 patch grid); use e.g. 512x896 or 704x1280. Found {}x{}.".format(width, height))
        chunk = snap_down(frames_per_chunk)
        if not TRAINED_CHUNKS[0] <= chunk <= TRAINED_CHUNKS[1]:
            logging.warning("[%s] frames_per_chunk %d is outside %d-%d, the segment lengths SCAIL-2 was trained on.",
                            self.node_name, chunk, *TRAINED_CHUNKS)
        logging.info("[%s] every chunk runs the full %d frames (SCAIL-2 was trained on %d-%d frame segments); "
                     "the output is cut to total_frames.", self.node_name, chunk, *TRAINED_CHUNKS)

        check_pose_percents(animate_inputs["pose_start_percent"], animate_inputs["pose_end_percent"])
        # the core node's names for them
        animate_inputs["pose_start"] = animate_inputs.pop("pose_start_percent")
        animate_inputs["pose_end"] = animate_inputs.pop("pose_end_percent")
        self._check_mode(bool(animate_inputs["replacement_mode"]), animate_inputs["reference_image_mask"])

        # The core node keeps the last previous_frame_count frames, moves the offset back by that
        # many and encodes them into latent frames; off the 4k+1 grid the decoded span is shorter
        # than the offset move and frames repeat at the seam, so snap before handing it over.
        wanted = int(animate_inputs["previous_frame_count"])
        self._previous = overlap_for_motion_frames(wanted)
        if self._previous != wanted:
            logging.info("[%s] previous_frame_count %d is not on the 4k+1 grid; using %d.", self.node_name, wanted, self._previous)
            animate_inputs["previous_frame_count"] = self._previous

        # SCAIL-2 is trained with the reference stretched to CLIP's square, and in replacement
        # mode with the character on black (the authors, zai-org/SCAIL-2 issue #30).
        clip_vision = animate_inputs.pop("clip_vision")
        image = reference_image[:1]
        if animate_inputs["replacement_mode"]:
            image = character_on_black(image, animate_inputs["reference_image_mask"][:1])
        animate_inputs["clip_vision_output"] = clip_vision_encode(clip_vision, image)
        return self._previous

    def _check_mode(self, replacement, reference_mask):
        rendered = mask_convention(reference_mask)
        mode = REPLACEMENT if replacement else ANIMATION
        if rendered is None:
            logging.warning("[%s] reference_image_mask has no clear white or black background; cannot check that it "
                            "was rendered for %s mode.", self.node_name, mode)
        elif rendered != mode:
            raise ValueError("reference_image_mask was rendered for {} mode ({} background) but replacement_mode is {}: "
                             "set replacement_mode to {}, or re-render the masks (SCAIL-2 Colored Mask) with "
                             "replacement_mode {}.".format(rendered, BACKGROUND[rendered], replacement,
                                                           rendered == REPLACEMENT, replacement))

    def chunk_length(self, produced, total_frames, frames_per_chunk, overlap):
        return full_chunk_length(produced, total_frames, frames_per_chunk, overlap)

    def check_videos(self, pose_video, animate_inputs):
        mask = animate_inputs["pose_video_mask"]
        if mask.shape[0] != pose_video.shape[0]:
            raise ValueError("pose_video_mask has {} frames but pose_video has {}: both come from the same driving "
                             "video; re-run the preprocess (SCAIL-2 Preprocess).".format(int(mask.shape[0]), int(pose_video.shape[0])))

    def continuation(self, anchor, offset):
        return {"video_frame_offset": offset, "previous_frames": anchor}

    def unpack(self, outputs, anchor):
        positive, negative, latent, offset = outputs[:self.OUTPUTS]
        # the previous frames come back decoded at the head of the chunk; the references travel
        # as conditioning, so no latent frame is dropped
        trim_image = 0 if anchor is None else min(self._previous, int(anchor.shape[0]))
        return positive, negative, latent, 0, trim_image, offset
