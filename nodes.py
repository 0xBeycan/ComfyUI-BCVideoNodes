"""Wan Animate long video in one node: chained fixed-size chunks.

One node per core conditioning node: WanAnimateLongVideoSampler wraps
WanAnimateToVideo (Wan 2.2 Animate) and WanAnimate2LongVideoSampler wraps
WanAnimate2ToVideo. Every chunk after the first is seeded with the previous
chunk's last frames (continue_motion) and the driving videos are read from
the returned video_frame_offset, so they stay aligned across the whole run.
The sampling stack (ModelSamplingSD3 -> BasicScheduler -> KSamplerSelect ->
SamplerCustom -> TrimVideoLatent -> VAEDecode) is called node-by-node from
ComfyUI's own registry, so this stays in step with core.
"""

import inspect
import logging

# torch and comfy.* are imported inside the functions that use them, so this
# module (and the package __init__) imports without a ComfyUI install; pytest
# collects the repo root as a package.
if __package__:
    from .chunk_planner import format_plan, next_chunk_length, overlap_for_motion_frames, plan_chunks, produced_frames
else:  # top-level import outside ComfyUI (pytest, tooling)
    from chunk_planner import format_plan, next_chunk_length, overlap_for_motion_frames, plan_chunks, produced_frames

ANIMATE_OUTPUTS = 6  # positive, negative, latent, trim_latent, trim_image, video_frame_offset
UPDATE_HINT = "Update ComfyUI: this node needs the {} that returns trim_latent / trim_image / video_frame_offset."


def _node_class(node_id):
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if cls is None:
        raise RuntimeError("Core node '{}' is not registered. Update ComfyUI.".format(node_id))
    return cls


def _with_schema_defaults(cls, kwargs):
    # The graph executor fills widget defaults before calling a node; calling
    # the class directly we must do the same, or a core update that adds a
    # widget turns into a TypeError here.
    spec = cls.INPUT_TYPES()
    for section in ("required", "optional"):
        for name, definition in (spec.get(section) or {}).items():
            if name in kwargs or not isinstance(definition, (list, tuple)) or len(definition) < 2:
                continue
            options = definition[1]
            if isinstance(options, dict) and "default" in options:
                kwargs[name] = options["default"]
    return kwargs


def _call_node(node_id, **kwargs):
    """Run a core node outside the graph and return its outputs as a tuple.

    V3 nodes (io.ComfyNode) expose FUNCTION as a classmethod and return a
    NodeOutput whose values live in .args; V1 nodes name an instance method
    that returns a tuple, or a dict with a "result" key.
    """
    cls = _node_class(node_id)
    fn = getattr(cls, cls.FUNCTION)
    if not inspect.ismethod(fn):
        fn = getattr(cls(), cls.FUNCTION)
    result = fn(**_with_schema_defaults(cls, dict(kwargs)))
    if hasattr(result, "args"):
        return tuple(result.args)
    if isinstance(result, dict):
        return tuple(result["result"])
    return tuple(result)


def _combo_default(options, preferred):
    return preferred if preferred in options else options[0]


class _StepLogger:
    """Wraps the KSamplerSelect sampler so every denoising step is logged with
    its chunk. SamplerCustom builds its own callback (preview + progress bar)
    and CFGGuider hands it to sampler.sample; that is the one point on the
    core chain where the step is visible without re-implementing SamplerCustom.
    Everything else is delegated to the real sampler."""

    def __init__(self, sampler, log_prefix):
        self._sampler = sampler
        self._log_prefix = log_prefix
        self.label = ""

    def __getattr__(self, name):
        return getattr(self._sampler, name)

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        label = self.label
        prefix = self._log_prefix

        def logged(step, x0, x, total_steps):
            logging.info("%s %s step %d/%d", prefix, label, step + 1, total_steps)
            if callback is not None:
                callback(step, x0, x, total_steps)

        return self._sampler.sample(model_wrap, sigmas, extra_args, logged, noise, latent_image, denoise_mask, disable_pbar)


class _LongVideoSampler:
    """The chunk loop, shared by both nodes.

    A subclass names the core conditioning node it wraps, lists the inputs
    that pass straight through to it (``_animate_inputs``) and says how many
    frames that node trims back off every chained chunk (``_prepare``).
    Everything else - widgets, sampling stack, loop, output - is identical.
    """

    ANIMATE_NODE = ""
    MODEL_TOOLTIP = ""
    DEFAULT_CHUNK = 81
    DEFAULT_SHIFT = 5.0
    DEFAULT_SAMPLER = "lcm"
    DEFAULT_STEPS = 6

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "chunk_plan")
    FUNCTION = "generate"
    CATEGORY = "WanAnimate"

    @classmethod
    def _animate_inputs(cls, max_res):
        """(required, optional) inputs handed to the core node unchanged, in widget order."""
        raise NotImplementedError

    def _prepare(self, animate_cls, animate_inputs):
        """Validate / normalize the pass-through inputs before the loop and
        return the frames the core node trims back off every chained chunk."""
        raise NotImplementedError

    def _after_animate(self, positive, negative, trim_image, animate_inputs):
        """Repairs on the core node's conditioning before it is sampled."""
        return None

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        import nodes as comfy_nodes

        samplers = list(comfy.samplers.SAMPLER_NAMES)
        schedulers = list(comfy.samplers.SCHEDULER_NAMES)
        max_res = comfy_nodes.MAX_RESOLUTION
        required, optional = cls._animate_inputs(max_res)
        return {
            "required": {
                "model": ("MODEL", {"tooltip": cls.MODEL_TOOLTIP}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE", {"tooltip": "The character to animate."}),
                "pose_video": ("IMAGE", {"tooltip": "Driving video. With total_frames = 0 its frame count is the output length."}),
                "width": ("INT", {"default": 720, "min": 16, "max": max_res, "step": 2, "tooltip": "Multiples of 16 are ideal; the VAE crops to a multiple of 8."}),
                "height": ("INT", {"default": 1280, "min": 16, "max": max_res, "step": 2, "tooltip": "Multiples of 16 are ideal; the VAE crops to a multiple of 8."}),
                "frames_per_chunk": ("INT", {"default": cls.DEFAULT_CHUNK, "min": 5, "max": max_res, "step": 4, "tooltip": "Frames sampled per chunk, rounded down to 4k+1. 81 for 24 GB, 49 for 16 GB, 33 for 12 GB are sane starts."}),
                "total_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Exact output length. 0 = the pose video's frame count."}),
                "shift": ("FLOAT", {"default": cls.DEFAULT_SHIFT, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "ModelSamplingSD3 shift, applied to the model before the schedule is built."}),
                "sampler_name": (samplers, {"default": _combo_default(samplers, cls.DEFAULT_SAMPLER)}),
                "scheduler": (schedulers, {"default": _combo_default(schedulers, "simple"), "tooltip": "Ignored when sigmas_override is connected."}),
                "steps": ("INT", {"default": cls.DEFAULT_STEPS, "min": 1, "max": 10000, "tooltip": "Ignored when sigmas_override is connected."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ignored when sigmas_override is connected."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "seed_mode": (["increment", "fixed"], {"default": "increment", "tooltip": "increment: chunk i uses seed + i. fixed: every chunk uses seed."}),
                **required,
            },
            "optional": {
                **optional,
                "sigmas_override": ("SIGMAS", {"tooltip": "Replaces the internal schedule; scheduler, steps and denoise are then ignored. shift still applies to the model."}),
            },
        }

    def generate(
        self,
        model,
        positive,
        negative,
        vae,
        reference_image,
        pose_video,
        width,
        height,
        frames_per_chunk,
        total_frames,
        shift,
        sampler_name,
        scheduler,
        steps,
        denoise,
        cfg,
        seed,
        seed_mode,
        sigmas_override=None,
        **animate_inputs,
    ):
        import torch
        import comfy.model_management
        import comfy.utils

        log_prefix = "[{}]".format(type(self).__name__)
        animate_node = self.ANIMATE_NODE
        update_hint = UPDATE_HINT.format(animate_node)

        animate_cls = _node_class(animate_node)
        if len(animate_cls.RETURN_TYPES) < ANIMATE_OUTPUTS:
            raise RuntimeError("{} returns {} outputs, {} expected. {}".format(animate_node, len(animate_cls.RETURN_TYPES), ANIMATE_OUTPUTS, update_hint))
        overlap = self._prepare(animate_cls, animate_inputs)

        pose_frames = int(pose_video.shape[0])
        if pose_frames < 1:
            raise ValueError("pose_video has no frames.")
        total = int(total_frames) if total_frames > 0 else pose_frames
        if total > pose_frames:
            logging.warning("%s total_frames (%d) exceeds pose_video length (%d): the last pose frame is held for the remaining %d frames.",
                            log_prefix, total, pose_frames, total - pose_frames)
            # The core node holds the last frame within a chunk, but once the
            # offset itself runs past the pose video it errors (Animate 2) or
            # drops the pose entirely (Animate). Pad up front.
            pose_video = torch.cat((pose_video, pose_video[-1:].expand(total - pose_frames, -1, -1, -1)), dim=0)

        plan = plan_chunks(total, frames_per_chunk, overlap)
        logging.info("%s chunk plan: %s", log_prefix, format_plan(plan, produced_frames(plan, overlap), total, pose_frames, overlap))

        patched = _call_node("ModelSamplingSD3", model=model, shift=shift)[0]
        if sigmas_override is not None:
            sigmas = sigmas_override
            logging.info("%s sigmas_override connected: scheduler / steps / denoise widgets are ignored.", log_prefix)
        else:
            sigmas = _call_node("BasicScheduler", model=patched, scheduler=scheduler, steps=steps, denoise=denoise)[0]
        if sigmas.numel() < 2:
            raise ValueError("The sigma schedule is empty (denoise too low, or an empty sigmas_override).")
        sampler = _StepLogger(_call_node("KSamplerSelect", sampler_name=sampler_name)[0], log_prefix)

        progress = comfy.utils.ProgressBar(len(plan))
        chunks = []
        lengths = []
        produced = 0
        anchor = None
        offset = 0
        while produced < total:
            comfy.model_management.throw_exception_if_processing_interrupted()
            index = len(lengths)
            length = next_chunk_length(produced, total, frames_per_chunk, overlap)
            chunk_seed = seed if seed_mode == "fixed" else (seed + index) % (1 << 64)
            pose_offset = offset

            animate = _call_node(
                animate_node,
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                reference_image=reference_image,
                pose_video=pose_video,
                continue_motion=anchor,
                video_frame_offset=offset,
                **animate_inputs,
            )
            if len(animate) < ANIMATE_OUTPUTS:
                raise RuntimeError("{} returned {} outputs, {} expected. {}".format(animate_node, len(animate), ANIMATE_OUTPUTS, update_hint))
            chunk_positive, chunk_negative, latent, trim_latent, trim_image, offset = animate[:ANIMATE_OUTPUTS]
            self._after_animate(chunk_positive, chunk_negative, trim_image, animate_inputs)

            # 1-based frame span this chunk adds to the output, as the plan expects it.
            sampler.label = "chunk {}/{} (frames {}-{}/{})".format(index + 1, len(plan), produced + 1, min(total, produced + length - trim_image), total)
            logging.info("%s %s: length %d, pose offset %d, seed %d", log_prefix, sampler.label, length, pose_offset, chunk_seed)

            sampled = _call_node(
                "SamplerCustom",
                model=patched,
                add_noise=True,
                noise_seed=chunk_seed,
                cfg=cfg,
                positive=chunk_positive,
                negative=chunk_negative,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=latent,
            )[0]
            if trim_latent > 0:
                sampled = _call_node("TrimVideoLatent", samples=sampled, trim_amount=trim_latent)[0]
            images = _call_node("VAEDecode", vae=vae, samples=sampled)[0]
            if trim_image > 0:
                images = images[trim_image:]
            images = images.cpu()
            del sampled, latent

            if images.shape[0] == 0:
                raise RuntimeError("Chunk {} (length {}) contributed no frames after trimming {}; the overlap exceeds the chunk.".format(index, length, trim_image))
            chunks.append(images)
            lengths.append(length)
            produced += int(images.shape[0])
            anchor = images
            if index > 0 and trim_image != overlap:
                logging.warning("%s %s trimmed %d frames, planner assumed %d; using %d from here on.", log_prefix, animate_node, trim_image, overlap, trim_image)
                overlap = trim_image
            logging.info("%s %s done: %d new frames, %d/%d total", log_prefix, sampler.label, int(images.shape[0]), min(produced, total), total)
            progress.update(1)

        images = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        images = images[:total]
        plan_text = format_plan(lengths, produced, total, pose_frames, overlap)
        if lengths != plan:
            logging.info("%s ran: %s", log_prefix, plan_text)
        return (images, int(images.shape[0]), plan_text)


def _fix_replacement_mask(cond, seed_frames, seen):
    """Put WanAnimateToVideo's character_mask rows where the model expects them.

    The concat mask has 4 rows per latent frame; pixel frame 0 fills the 4
    rows of latent 0 and pixel frame f (f >= 1) sits at row f + 3. Core's own
    seed-frame zeroing (ref_motion_latent_length * 4 rows), its other Wan
    nodes (``mask[:, :, :frames + 3]``) and the reference implementation
    (``get_i2v_mask``: repeat_interleave of frame 0) all follow that. The
    character mask alone is written at row f, three rows early: it lands on
    the last three seed rows, so the seed latent is flagged "character
    unknown" over real pixels, and every later frame's mask is shifted by
    three sub-frames. This moves the character rows back by three and
    restores the seed rows (or repeats frame 0 when there is no seed).
    """
    for entry in cond:
        mask = entry[1].get("concat_mask") if len(entry) > 1 and isinstance(entry[1], dict) else None
        if mask is None or id(mask) in seen:
            continue
        seen.add(id(mask))
        height, width = mask.shape[-2], mask.shape[-1]
        rows = mask[:, :, 1:].transpose(1, 2).reshape(1, -1, height, width)  # index 0 is the reference latent
        fixed = rows.clone()
        fixed[:, seed_frames + 3:] = rows[:, seed_frames:rows.shape[1] - 3]
        fixed[:, :seed_frames + 3] = 0.0 if seed_frames > 0 else rows[:, :1]
        mask[:, :, 1:] = fixed.view(1, -1, 4, height, width).transpose(1, 2)


class WanAnimateLongVideoSampler(_LongVideoSampler):
    ANIMATE_NODE = "WanAnimateToVideo"
    MODEL_TOOLTIP = "Wan 2.2 Animate model. LoRA and model patches pass through unchanged; shift is applied here."
    DEFAULT_CHUNK = 77
    DEFAULT_SHIFT = 8.0
    DEFAULT_SAMPLER = "euler"
    DEFAULT_STEPS = 6
    DESCRIPTION = "Generates an arbitrarily long Wan 2.2 Animate video by chaining fixed-size chunks internally. Output length equals total_frames (or the pose video length) exactly."

    @classmethod
    def _animate_inputs(cls, max_res):
        required = {
            "continue_motion_max_frames": ("INT", {"default": 5, "min": 1, "max": max_res, "step": 4, "tooltip": "Frames of the previous chunk that seed the next one and are trimmed back off: the overlap between chunks. Snapped down to the 4k+1 grid; must be smaller than frames_per_chunk."}),
        }
        optional = {
            "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the reference image."}),
            "face_video": ("IMAGE", {"tooltip": "Face crops of the driving video (512x512), read from the same offset as the pose video."}),
            "background_video": ("IMAGE", {"tooltip": "Background to place the character into (replacement mode), read from the same offset as the pose video."}),
            "character_mask": ("MASK", {"tooltip": "Where the character goes in the background video (replacement mode). A single frame is repeated; a video is read from the same offset as the pose video."}),
        }
        return required, optional

    def _prepare(self, animate_cls, animate_inputs):
        # The overlap is a widget here (continue_motion_max_frames): the core
        # node keeps that many frames of continue_motion, moves the offset back
        # by the same amount and trims their decoded span off again. Off the
        # 4k+1 grid the trimmed span is shorter than the offset move and a few
        # frames repeat at the seam, so snap before handing it over.
        wanted = int(animate_inputs["continue_motion_max_frames"])
        motion_frames = overlap_for_motion_frames(wanted)
        if motion_frames != wanted:
            logging.info("[%s] continue_motion_max_frames %d is not on the 4k+1 grid; using %d.", type(self).__name__, wanted, motion_frames)
            animate_inputs["continue_motion_max_frames"] = motion_frames
        if animate_inputs.get("character_mask") is not None:
            logging.info("[%s] character_mask connected: realigning the core node's mask rows (see _fix_replacement_mask).", type(self).__name__)
        return motion_frames

    def _after_animate(self, positive, negative, trim_image, animate_inputs):
        if animate_inputs.get("character_mask") is None:
            return None  # without a character mask core's rows are already right
        seen = set()
        _fix_replacement_mask(positive, trim_image, seen)
        _fix_replacement_mask(negative, trim_image, seen)


class WanAnimate2LongVideoSampler(_LongVideoSampler):
    ANIMATE_NODE = "WanAnimate2ToVideo"
    MODEL_TOOLTIP = "Wan Animate 2 model. LoRA, WanAnimate2Cache and context-window patches pass through unchanged; shift is applied here."
    DEFAULT_CHUNK = 81
    DEFAULT_SHIFT = 5.0
    DEFAULT_SAMPLER = "lcm"
    DEFAULT_STEPS = 6
    DESCRIPTION = "Generates an arbitrarily long Wan Animate 2 video by chaining fixed-size chunks internally. Output length equals total_frames (or the pose video length) exactly."

    @classmethod
    def _animate_inputs(cls, max_res):
        required = {
            "reference_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
        optional = {
            "positive_pose": ("CONDITIONING", {"tooltip": "Prompt for the pose branch. Defaults to positive."}),
            "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the reference image."}),
            "clip_vision_output_pose": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the pose video's first frame. Defaults to clip_vision_output."}),
        }
        return required, optional

    def _prepare(self, animate_cls, animate_inputs):
        start, end = animate_inputs["pose_start_percent"], animate_inputs["pose_end_percent"]
        if start > end:
            raise ValueError("pose_start_percent ({}) must not be greater than pose_end_percent ({}).".format(start, end))
        # The node keeps the last CONTINUE_MOTION_FRAMES frames of continue_motion
        # and trims their decoded span back off every chained chunk; that span
        # is the overlap. Read from the class so a core change is picked up.
        return overlap_for_motion_frames(int(getattr(animate_cls, "CONTINUE_MOTION_FRAMES", 1)))


NODE_CLASS_MAPPINGS = {
    "WanAnimateLongVideoSampler": WanAnimateLongVideoSampler,
    "WanAnimate2LongVideoSampler": WanAnimate2LongVideoSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAnimateLongVideoSampler": "Wan Animate Long Video Sampler",
    "WanAnimate2LongVideoSampler": "Wan Animate 2 Long Video Sampler",
}
