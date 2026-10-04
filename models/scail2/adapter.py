"""WanSCAILToVideo (SCAIL-2) in the long-video loop.

The core node's chaining contract differs from Wan Animate's: it is seeded with previous_frames
(the last previous_frame_count of them, VAE-encoded into the first latent frames, which a noise
mask marks known) and returns 4 outputs, with no trim values, so the trim is computed here. Like
the Animate nodes it moves video_frame_offset back by the frames it kept and seeks the pose video
and its colored mask by that offset. SCAIL-2 was trained on 65-81 frame segments, so the node's
last_chunk defaults to full (nodes/sampler.py): every chunk, the last one included, runs the full
frames_per_chunk and the output is cut to total_frames. The reference is CLIP-encoded once per
run as official SCAIL-2 encodes it: the VAE reference (cropped and resized to the generation
size; in replacement mode on a black background) stretched to CLIP's square without antialias or
8-bit rounding (core_nodes.clip_vision_encode_official).

The pose tokens' RoPE as official SCAIL-2 builds it (rope.py), an object patch on the model
clone, so it holds only while the sampler's clone is loaded.

The history frames as official SCAIL-2 feeds them (zai-org/SCAIL-2 wan/scail.py, wan-scail2
branch): on every chained chunk the model gets the encoded previous frames clean at every step
(`apply_clean_history`) and the 4 mask channels of the video tokens 1 on those latent frames, 0
elsewhere (`history_mask`, concatenated after the 16 latent channels: wan/modules/model_scail2.py).
Core's noise mask alone feeds the model those frames re-noised to the current sigma and the mask
channels all zero (WAN21.concat_cond of a model built with image_to_video=False). The adapter's
APPLY_MODEL wrapper (`_with_history`) writes both into the model call: core's c_concat is those
4 channels, concatenated after the latent's 16 in BaseModel._apply_model. Core's noise mask stays:
it clamps the denoised history frames back to the clean latent, so the chunk ends with them clean.
"""

import logging

import torch

from ...libs.chunking import FULL, overlap_for_motion_frames, snap_down
from ...libs.video import requantized
from ..common.animate import AnimateAdapter, check_pose_percents, core_frame, mask_window
from ..common.core_nodes import clip_vision_encode_official
from .rope import official_pose_rope

SIZE_MULTIPLE = 32  # the pose runs at half resolution through the /16 patch grid
TRAINED_CHUNKS = (65, 81)  # the segment lengths SCAIL-2 was trained on (zai-org/SCAIL-2 issue #16)
# core's thresholds (comfy_extras/nodes_scail.py): a mask colour channel is on above 225/255, and
# the replacement-mode reference keeps the pixels whose mask has any channel above 0.1
ON = 225.0 / 255.0
CHARACTER = 0.1
ANIMATION, REPLACEMENT = "animation", "replacement"
HISTORY_WRAPPER = "bcvideonodes_scail2_history"  # the key of the APPLY_MODEL wrapper patch_model installs
ROPE_ENCODE = "diffusion_model.rope_encode"  # the model object patch_model replaces with official's pose RoPE
# each colored mask's background per mode (SCAIL-Pose's preprocess, core's SCAIL2ColoredMask): the
# reference mask is white in animation mode and black in replacement mode, the driving mask the
# opposite
BACKGROUND = {ANIMATION: "white", REPLACEMENT: "black"}
DRIVING_BACKGROUND = {ANIMATION: "black", REPLACEMENT: "white"}


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


def mask_convention(mask, background=BACKGROUND):
    """The mode a colored mask was rendered for, read from the border of its first frame: the
    mode whose `background` (mode -> "white" / "black": BACKGROUND for the reference mask,
    DRIVING_BACKGROUND for the driving mask) most border pixels have, None when neither colour
    has most of them. A half-precision mask's frame is requantized first (libs/video.requantized)."""
    frame = requantized(mask[0, ..., :3]).float()
    border = [frame[0], frame[-1]]
    if frame.shape[0] > 2:
        border += [frame[1:-1, 0], frame[1:-1, -1]]
    pixels = torch.cat(border, dim=0)
    white = float((pixels.min(dim=-1).values > ON).float().mean())
    black = float((pixels.max(dim=-1).values <= CHARACTER).float().mean())
    for colour, share in (("white", white), ("black", black)):
        if share > 0.5:
            return next(mode for mode, painted in background.items() if painted == colour)
    return None


def check_reference_count(reference_image, reference_image_mask):
    """Raises unless the references and their colored masks are as many frames: core's
    WanSCAILToVideo pairs them by position (SCAIL-2 multi-reference, the first the primary), and
    repeats the last mask for a reference without one."""
    images, masks = int(reference_image.shape[0]), int(reference_image_mask.shape[0])
    if images != masks:
        raise ValueError("reference_image has {} frame(s) but reference_image_mask {}: each reference needs its colored "
                         "mask, paired by position. Link the sampler's reference_image from SCAIL-2 Preprocess's "
                         "reference_images (the primary, and the face close-up with face_crop on) and reference_image_mask "
                         "from its reference_image_mask.".format(images, masks))


class SCAIL2Adapter(AnimateAdapter):
    ANIMATE_NODE = "WanSCAILToVideo"
    OUTPUTS = 4  # positive, negative, latent, video_frame_offset
    UPDATE_HINT = "Update ComfyUI: this node needs the {} of SCAIL-2 (previous_frames, pose_video_mask)."
    HELD_VIDEOS = ("pose_video", "pose_video_mask")

    def prepare(self, animate_cls, animate_inputs, reference_image, width, height, frames_per_chunk):
        if width % SIZE_MULTIPLE or height % SIZE_MULTIPLE:
            raise ValueError("width and height must be divisible by 32 for SCAIL-2 (the pose runs at half resolution "
                             "through the /16 patch grid); use e.g. 512x896 or 704x1280. Found {}x{}.".format(width, height))
        chunk = snap_down(frames_per_chunk)
        if not TRAINED_CHUNKS[0] <= chunk <= TRAINED_CHUNKS[1]:
            logging.warning("[%s] frames_per_chunk %d is outside %d-%d, the segment lengths SCAIL-2 was trained on.",
                            self.node_name, chunk, *TRAINED_CHUNKS)
        if self.last_chunk == FULL:
            logging.info("[%s] every chunk runs the full %d frames (SCAIL-2 was trained on %d-%d frame segments); "
                         "the output is cut to total_frames.", self.node_name, chunk, *TRAINED_CHUNKS)

        check_pose_percents(animate_inputs["pose_start_percent"], animate_inputs["pose_end_percent"])
        # the core node's names for them
        animate_inputs["pose_start"] = animate_inputs.pop("pose_start_percent")
        animate_inputs["pose_end"] = animate_inputs.pop("pose_end_percent")
        replacement = bool(animate_inputs["replacement_mode"])
        check_reference_count(reference_image, animate_inputs["reference_image_mask"])
        self._check_mode(replacement, "reference_image_mask", animate_inputs["reference_image_mask"], BACKGROUND)
        self._check_mode(replacement, "pose_video_mask", animate_inputs["pose_video_mask"], DRIVING_BACKGROUND)

        # The core node keeps the last previous_frame_count frames, moves the offset back by that
        # many and encodes them into latent frames; off the 4k+1 grid the decoded span is shorter
        # than the offset move and frames repeat at the seam, so snap before handing it over.
        wanted = int(animate_inputs["previous_frame_count"])
        self._previous = overlap_for_motion_frames(wanted)
        if self._previous != wanted:
            logging.info("[%s] previous_frame_count %d is not on the 4k+1 grid; using %d.", self.node_name, wanted, self._previous)
            animate_inputs["previous_frame_count"] = self._previous

        # Official SCAIL-2 CLIP-encodes the reference the VAE gets (generate.py, wan/scail.py):
        # center-cropped and resized to the generation size, here as core crops and resizes the
        # VAE reference, and in replacement mode with the character on black (the authors,
        # zai-org/SCAIL-2 issue #30); then stretched to CLIP's square as official does.
        image = core_frame(reference_image, width, height, "bicubic")
        if replacement:
            image = character_on_black(image, animate_inputs["reference_image_mask"][:1])
        animate_inputs["clip_vision_output"] = clip_vision_encode_official(animate_inputs.pop("clip_vision"), image)
        self._history = None  # the chunk's clean history latent frames, as core encoded them; None on the first chunk
        self._frames = 0  # the chunk's latent frames
        self._clean = None  # them in the sampling space, on the model's device, once the first model call needs them
        self._calls = 0  # the model calls they went into, which after_chunk logs
        return self._previous

    def _check_mode(self, replacement, name, mask, background):
        """The colored mask `name` against `replacement`: the mode its border was rendered for
        (`background`, as `mask_convention` takes it) must be the one the sampler runs in."""
        rendered = mask_convention(mask, background)
        mode = REPLACEMENT if replacement else ANIMATION
        if rendered is None:
            logging.warning("[%s] %s has no clear white or black background; cannot check that it "
                            "was rendered for %s mode.", self.node_name, name, mode)
        elif rendered != mode:
            raise ValueError("{} was rendered for {} mode ({} background) but replacement_mode is {}: "
                             "set replacement_mode to {}, or re-render the masks (SCAIL-2 Colored Mask) with "
                             "replacement_mode {}.".format(name, rendered, background[rendered], replacement,
                                                           rendered == REPLACEMENT, replacement))

    def check_videos(self, pose_video, animate_inputs):
        mask = animate_inputs["pose_video_mask"]
        if mask.shape[0] != pose_video.shape[0]:
            raise ValueError("pose_video_mask has {} frames but pose_video has {}: both come from the same driving "
                             "video; re-run the preprocess (SCAIL-2 Preprocess).".format(int(mask.shape[0]), int(pose_video.shape[0])))
        # core crops and resizes each of the two to half the generation size on its own, so a mask
        # of another size would cover another part of the frame than the pose
        if mask.shape[1:3] != pose_video.shape[1:3]:
            raise ValueError("pose_video_mask is {}x{} but pose_video is {}x{}: both come from the same driving video; "
                             "re-run the preprocess (SCAIL-2 Preprocess), or resize the mask to the pose video's "
                             "size.".format(int(mask.shape[2]), int(mask.shape[1]), int(pose_video.shape[2]), int(pose_video.shape[1])))

    def patch_model(self, patched, animate_inputs):
        import comfy.patcher_extension

        patched = patched.clone()
        patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, HISTORY_WRAPPER, self._with_history)
        # an object patch: set on the model only while this clone is loaded, so other workflows on
        # the same model keep core's pose RoPE
        patched.add_object_patch(ROPE_ENCODE, official_pose_rope(patched.get_model_object(ROPE_ENCODE)))
        return patched

    def _with_history(self, executor, x, t, c_concat=None, *args, **kwargs):
        """BaseModel.apply_model with the chunk's history as official SCAIL-2 feeds it: the first
        latent frames of ``x`` the clean history, and the 4 mask channels (``c_concat``) 1 there
        and 0 elsewhere. Without history (the first chunk) the call is passed on untouched."""
        if self._history is None:
            return executor(x, t, c_concat, *args, **kwargs)
        known = int(self._history.shape[2])
        if c_concat is None or c_concat.shape[1] != 4 or x.shape[2] != self._frames:
            raise RuntimeError("[{}] the model call is not the chunk's {} latent frames with 4 mask channels to mark the "
                               "previous frames in (c_concat {}, x {}): the model is not SCAIL-2, a context window "
                               "splits the chunk, or core changed how it feeds them; use a SCAIL-2 model without "
                               "context windows, on an up-to-date ComfyUI.".format(
                                   self.node_name, self._frames, None if c_concat is None else tuple(c_concat.shape), tuple(x.shape)))
        if self._clean is None or self._clean.device != x.device or self._clean.dtype != x.dtype:
            self._clean = executor.class_obj.process_latent_in(self._history.to(x.device)).to(x.dtype)
        x = x.clone()
        x[:, :, :known] = self._clean
        mask = torch.zeros_like(c_concat)
        mask[:, :, :known] = 1.0
        self._calls += 1
        return executor(x, t, mask, *args, **kwargs)

    def continuation(self, anchor, offset):
        return {"video_frame_offset": offset, "previous_frames": anchor}

    def unpack(self, outputs, anchor):
        positive, negative, latent, offset = outputs[:self.OUTPUTS]
        # the history the model call gets (_with_history): the latent frames core's noise mask
        # marks known, as core encoded them
        self._history = self._clean = None
        if anchor is not None:
            if latent.get("noise_mask") is None:
                raise RuntimeError("[{}] {} returned the previous frames without a noise_mask: core changed how it seeds "
                                   "a chained chunk; update ComfyUI-BCVideoNodes.".format(self.node_name, self.ANIMATE_NODE))
            known = int((latent["noise_mask"][0, 0].amax(dim=(1, 2)) == 0).sum())
            self._history = latent["samples"][:, :, :known].clone()
            self._frames = int(latent["samples"].shape[2])
        # the previous frames come back decoded at the head of the chunk; the references travel
        # as conditioning, so no latent frame is dropped
        trim_image = 0 if anchor is None else min(self._previous, int(anchor.shape[0]))
        return positive, negative, latent, 0, trim_image, offset

    def after_chunk(self, index):
        if self._history is None:
            return
        calls, self._calls = self._calls, 0
        logging.info("[%s] chunk %d: %d clean history latent frames, marked in the mask channels, in %d model calls.",
                     self.node_name, index + 1, int(self._history.shape[2]), calls)
        if calls == 0:
            logging.warning("[%s] chunk %d has history frames but no model call went through the history wrapper: the "
                            "model saw them re-noised, without the history mask. Another patch replaced "
                            "BaseModel.apply_model.", self.node_name, index + 1)
        self._history = self._clean = None

    def anchor_region(self, first, length, height, width, animate_inputs):
        if not animate_inputs["replacement_mode"]:
            return None  # animation mode generates the whole frame
        # replacement mode keeps the driving video's background: the character is every pixel of the
        # colored driving mask (held like the pose) that is not its white background
        window = animate_inputs["pose_video_mask"][first:first + length, ..., :3]
        return mask_window((window <= ON).any(dim=-1).float(), 0, length, height, width)
