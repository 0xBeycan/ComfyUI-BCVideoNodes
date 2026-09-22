"""Every node of the pack; pure wiring, the work is in chunk_planner.py and preprocess/.

Wan Animate long video in one node: chained fixed-size chunks.

One node per core conditioning node: BCVWanAnimateLongVideoSampler wraps
WanAnimateToVideo (Wan 2.2 Animate) and BCVWanAnimate2LongVideoSampler wraps
WanAnimate2ToVideo. Every chunk after the first is seeded with the previous
chunk's last frames (continue_motion) and the driving videos are read from
the returned video_frame_offset, so they stay aligned across the whole run.
The sampling stack (ModelSamplingSD3 -> BasicScheduler -> KSamplerSelect ->
SamplerCustom -> TrimVideoLatent -> VAEDecode) is called node-by-node from
ComfyUI's own registry, so this stays in step with core.

The preprocess nodes (pose, SAM 3.1 mask, face crop, guards) each call one
function of preprocess/; the two WanAnimate wrappers call the individual
nodes, so a wrapper computes exactly what the chained nodes compute.
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
WAN_BETA = "wan_beta"
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


def wan_beta_sigmas(steps, shift, denoise=1.0, alpha=0.6, beta=0.6):
    """The sigmas WanVideoWrapper's 'euler/beta' scheduler samples with:
    diffusers FlowMatchEulerDiscreteScheduler(shift, use_beta_sigmas=True).

    Not ComfyUI's 'beta' scheduler. diffusers shifts first and then spreads
    Beta(0.6, 0.6) quantiles between the shifted extremes, so shift only
    moves sigma_min (1/1000 shifted twice: once in __init__, once in
    set_timesteps) and the steps stay evenly spread. ComfyUI's beta takes
    the quantiles on the timestep axis and reads them off the shifted table,
    which at shift 5 and 4 steps gives 1 / .959 / .834 / .518 / 0 against
    the wrapper's 1 / .731 / .293 / .024 / 0.
    """
    import numpy
    import scipy.stats
    import torch

    total = int(steps / denoise) if 0.0 < denoise < 1.0 else int(steps)
    sigma_min = 1.0 / 1000
    for _ in range(2):
        sigma_min = shift * sigma_min / (1 + (shift - 1) * sigma_min)
    quantiles = scipy.stats.beta.ppf(1 - numpy.linspace(0, 1, total), alpha, beta)
    sigmas = [float(sigma_min + q * (1.0 - sigma_min)) for q in quantiles] + [0.0]
    return torch.FloatTensor(sigmas[-(int(steps) + 1):])


def _active_bar(total_steps):
    """The console tqdm bar the k-diffusion sampler is driving right now.

    It is created inside the sampler loop, so the only handle is tqdm's own
    registry of live bars; match on the step count. None when tqdm is not
    installed or the bar is disabled."""
    try:
        from tqdm import tqdm
        bars = list(tqdm._instances)
    except (ImportError, AttributeError):
        return None
    for bar in bars:
        if getattr(bar, "total", None) == total_steps and not getattr(bar, "disable", False):
            return bar
    return None


def _log_beside_bar(message, *args):
    """logging.info that does not tear through a live tqdm bar: the bar is
    cleared, the line printed, the bar redrawn on its next update."""
    try:
        from tqdm import tqdm
        context = tqdm.external_write_mode()
    except ImportError:
        logging.info(message, *args)
        return
    with context:
        logging.info(message, *args)


class _StepLogger:
    """Wraps the KSamplerSelect sampler so every denoising step is logged with
    its chunk and the console bar carries the chunk label. SamplerCustom
    builds its own callback (preview + progress bar) and CFGGuider hands it to
    sampler.sample; that is the one point on the core chain where the step is
    visible without re-implementing SamplerCustom. Everything else is
    delegated to the real sampler."""

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
            if step == 0:
                bar = _active_bar(total_steps)
                if bar is not None:
                    bar.set_description(label, refresh=False)
            _log_beside_bar("%s %s step %d/%d", prefix, label, step + 1, total_steps)
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
    DEFAULT_SAMPLER = "euler"
    DEFAULT_SCHEDULER = WAN_BETA
    DEFAULT_STEPS = 6

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "chunk_plan")
    FUNCTION = "generate"
    CATEGORY = "BCVideoNodes/Wan/Animate"

    @classmethod
    def _animate_inputs(cls, max_res):
        """(required, optional) inputs handed to the core node unchanged, in widget order."""
        raise NotImplementedError

    def _prepare(self, animate_cls, animate_inputs):
        """Validate / normalize the pass-through inputs before the loop and
        return the frames the core node trims back off every chained chunk.
        Inputs that are the node's own (not the core node's) are popped here."""
        raise NotImplementedError

    def _patch_model(self, patched, animate_inputs):
        """Model-level patches applied once per run, after ModelSamplingSD3."""
        return patched

    def _chunk_inputs(self, index, offset, anchor, pose_video, animate_inputs):
        """Core-node inputs that change per chunk beyond continue_motion /
        video_frame_offset / length; merged over the pass-through inputs."""
        return {}

    def _after_animate(self, positive, negative, trim_image, length, offset, animate_inputs):
        """Repairs on the core node's conditioning before it is sampled.
        ``offset`` is the video_frame_offset the core node was called with."""
        return None

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        import nodes as comfy_nodes

        samplers = list(comfy.samplers.SAMPLER_NAMES)
        schedulers = list(comfy.samplers.SCHEDULER_NAMES) + [WAN_BETA]
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
                "total_frames": ("INT", {"default": 81, "min": 0, "max": 100000, "tooltip": "Exact output length. 0 = the pose video's frame count."}),
                "shift": ("FLOAT", {"default": cls.DEFAULT_SHIFT, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "ModelSamplingSD3 shift, applied to the model before the schedule is built."}),
                "sampler_name": (samplers, {"default": _combo_default(samplers, cls.DEFAULT_SAMPLER)}),
                "scheduler": (schedulers, {"default": _combo_default(schedulers, cls.DEFAULT_SCHEDULER), "tooltip": "ComfyUI schedulers, plus wan_beta: the sigmas WanVideoWrapper's euler/beta samples with (diffusers beta sigmas), evenly spread with a small last step. Ignored when sigmas_override is connected."}),
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

        patched = self._patch_model(_call_node("ModelSamplingSD3", model=model, shift=shift)[0], animate_inputs)
        if sigmas_override is not None:
            sigmas = sigmas_override
            logging.info("%s sigmas_override connected: scheduler / steps / denoise widgets are ignored.", log_prefix)
        elif scheduler == WAN_BETA:
            sigmas = wan_beta_sigmas(steps, shift, denoise)
        else:
            sigmas = _call_node("BasicScheduler", model=patched, scheduler=scheduler, steps=steps, denoise=denoise)[0]
        if sigmas.numel() < 2:
            raise ValueError("The sigma schedule is empty (denoise too low, or an empty sigmas_override).")
        logging.info("%s sigmas (%s): %s", log_prefix, "override" if sigmas_override is not None else scheduler, ", ".join("{:.4f}".format(float(v)) for v in sigmas))
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
            chunk_inputs = dict(animate_inputs, **self._chunk_inputs(index, offset, anchor, pose_video, animate_inputs))

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
                **chunk_inputs,
            )
            if len(animate) < ANIMATE_OUTPUTS:
                raise RuntimeError("{} returned {} outputs, {} expected. {}".format(animate_node, len(animate), ANIMATE_OUTPUTS, update_hint))
            chunk_positive, chunk_negative, latent, trim_latent, trim_image, offset = animate[:ANIMATE_OUTPUTS]
            self._after_animate(chunk_positive, chunk_negative, trim_image, length, pose_offset, animate_inputs)

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


def _replacement_mask_rows(character_mask, offset, length, seed_frames, lat_h, lat_w):
    """The concat-mask rows for one window, built the way the reference
    implementation (wan/animate.py get_i2v_mask) builds them: one row per
    pixel frame, frame 0 repeated four times, seed frames known (0), frames
    the mask does not cover unknown (1). The pixel mask -> latent grid step
    uses core's filter (nearest-exact). Returns None when the mask does not
    reach this window, which is when core does not apply it either."""
    import torch

    mask = character_mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.shape[0] == 1:
        mask = mask.expand(length, -1, -1)
    elif mask.shape[0] > offset:
        mask = mask[offset:offset + length]
    else:
        return None
    mask = torch.nn.functional.interpolate(mask.unsqueeze(1).float(), size=(lat_h, lat_w), mode="nearest-exact").squeeze(1)
    frames = torch.ones((length, lat_h, lat_w), dtype=mask.dtype, device=mask.device)
    frames[:mask.shape[0]] = mask
    frames[:seed_frames] = 0.0
    return torch.cat((frames[:1].expand(4, -1, -1), frames[1:]), dim=0)


def _fix_replacement_mask(cond, rows, seen):
    """Write ``rows`` (from _replacement_mask_rows) over the video part of
    the concat mask WanAnimateToVideo returns; index 0 stays the reference
    latent.

    Why: the concat mask has 4 rows per latent frame and pixel frame f >= 1
    belongs at row f + 3 (frame 0 fills latent 0). Core's own seed-frame
    zeroing, its other Wan nodes (``mask[:, :, :frames + 3]``) and the
    reference follow that; WanAnimateToVideo writes character_mask at row f,
    three rows early, so the last three seed rows are overwritten and the
    seed latent is flagged "character unknown" over real pixels.
    """
    for entry in cond:
        mask = entry[1].get("concat_mask") if len(entry) > 1 and isinstance(entry[1], dict) else None
        if mask is None or id(mask) in seen:
            continue
        seen.add(id(mask))
        height, width = mask.shape[-2], mask.shape[-1]
        mask[:, :, 1:] = rows.to(mask).view(1, -1, 4, height, width).transpose(1, 2)


class BCVWanAnimateLongVideoSampler(_LongVideoSampler):
    ANIMATE_NODE = "WanAnimateToVideo"
    MODEL_TOOLTIP = "Wan 2.2 Animate model. LoRA and model patches pass through unchanged; shift is applied here."
    DEFAULT_CHUNK = 81
    DEFAULT_SHIFT = 8.0
    DEFAULT_SAMPLER = "euler"
    DEFAULT_SCHEDULER = WAN_BETA
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

    def _after_animate(self, positive, negative, trim_image, length, offset, animate_inputs):
        character_mask = animate_inputs.get("character_mask")
        if character_mask is None:
            return None  # without a character mask core's rows are already right
        mask = next((entry[1]["concat_mask"] for entry in positive if len(entry) > 1 and isinstance(entry[1], dict) and "concat_mask" in entry[1]), None)
        if mask is None:
            return None
        # core moves the offset back by the seed frames before it seeks the mask
        rows = _replacement_mask_rows(character_mask, max(0, offset - trim_image), length, trim_image, mask.shape[-2], mask.shape[-1])
        if rows is None:
            return None  # mask does not reach this window: core leaves its rows alone, so do we
        seen = set()
        _fix_replacement_mask(positive, rows, seen)
        _fix_replacement_mask(negative, rows, seen)


def _seed_frame_attention_bias(log_scale):
    """Wan Animate 2's ``log_scale`` (wanxiang/models/wan_animate_2_model.py
    _score_mod_impl): in every generation self-attention the keys of latent
    frame 1 - the previous chunk's last frame on chained chunks, grey fill on
    the first - get ``log_scale`` added to their logits. The official distilled
    checkpoint runs with -1.3 (infer/wan_animate_2_distillation.yaml), the
    base one with 0.0; core has no equivalent, so distilled weights attend to
    the seed frame e^1.3 = 3.7x harder than they were trained to.

    Installed as transformer_options["optimized_attention_override"]. The
    generation calls are told apart by shape: per frame (q = hw tokens, k =
    all gen tokens, plus that frame's pose tokens) or the whole clip when the
    pose branch is windowed out; cross-attention and the pose branch have
    other shapes and pass through untouched."""

    def override(func, q, k, v, **kwargs):
        options = kwargs.get("transformer_options") or {}
        grid = options.get("grid_sizes")
        if options.get("block_type") == "double" and grid is not None:
            frames, gh, gw = grid
            hw = gh * gw
            tokens = frames * hw
            lq, lk = q.shape[1], k.shape[1]
            if (lq == hw and lk in (tokens, tokens + hw)) or (lq == tokens and lk == tokens):
                import comfy.ldm.modules.attention

                bias = q.new_zeros(1, 1, 1, lk)
                bias[..., hw:2 * hw] = log_scale
                return comfy.ldm.modules.attention.attention_pytorch(q, k, v, mask=bias, **kwargs)
        return func(q, k, v, **kwargs)

    return override


class BCVWanAnimate2LongVideoSampler(_LongVideoSampler):
    ANIMATE_NODE = "WanAnimate2ToVideo"
    MODEL_TOOLTIP = "Wan Animate 2 model. LoRA, WanAnimate2Cache and context-window patches pass through unchanged; shift is applied here."
    DEFAULT_CHUNK = 81
    DEFAULT_SHIFT = 5.0
    DEFAULT_SAMPLER = "euler"
    DEFAULT_SCHEDULER = WAN_BETA
    DEFAULT_STEPS = 10
    DESCRIPTION = "Generates an arbitrarily long Wan Animate 2 video by chaining fixed-size chunks internally. Output length equals total_frames (or the pose video length) exactly."

    @classmethod
    def _animate_inputs(cls, max_res):
        required = {
            "reference_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "attn_log_scale": ("FLOAT", {"default": -1.3, "min": -10.0, "max": 10.0, "step": 0.1, "tooltip": "Logit bias on attention to the seed frame (latent frame 1), the official log_scale. -1.3 is the distilled checkpoint's config; use 0.0 for the base checkpoint. Core has no equivalent (0.0 is core's behaviour)."}),
        }
        optional = {
            "positive_pose": ("CONDITIONING", {"tooltip": "Prompt for the pose branch. Defaults to positive. The official pipeline never leaves it empty (default: 人物动作的参考视频)."}),
            "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the reference image."}),
            "clip_vision_output_pose": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the pose video's first frame, used for every chunk. Defaults to clip_vision_output. Connect clip_vision instead to re-encode per chunk."}),
            "clip_vision": ("CLIP_VISION", {"tooltip": "When connected, the pose CLIP embedding is re-encoded from the first frame of each chunk's pose window, as the official pipeline does; clip_vision_output_pose is then ignored."}),
        }
        return required, optional

    def _prepare(self, animate_cls, animate_inputs):
        start, end = animate_inputs["pose_start_percent"], animate_inputs["pose_end_percent"]
        if start > end:
            raise ValueError("pose_start_percent ({}) must not be greater than pose_end_percent ({}).".format(start, end))
        self._log_scale = float(animate_inputs.pop("attn_log_scale", -1.3))
        self._clip_vision = animate_inputs.pop("clip_vision", None)
        if self._clip_vision is not None:
            animate_inputs.pop("clip_vision_output_pose", None)
            logging.info("[%s] clip_vision connected: pose CLIP embedding is re-encoded per chunk.", type(self).__name__)
        # The node keeps the last CONTINUE_MOTION_FRAMES frames of continue_motion
        # and trims their decoded span back off every chained chunk; that span
        # is the overlap. Read from the class so a core change is picked up.
        return overlap_for_motion_frames(int(getattr(animate_cls, "CONTINUE_MOTION_FRAMES", 1)))

    def _patch_model(self, patched, animate_inputs):
        if self._log_scale == 0.0:
            return patched
        patched = patched.clone()
        patched.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = _seed_frame_attention_bias(self._log_scale)
        logging.info("[%s] attn_log_scale %.2f on the seed frame (official distilled config: -1.3).", type(self).__name__, self._log_scale)
        return patched

    def _chunk_inputs(self, index, offset, anchor, pose_video, animate_inputs):
        if self._clip_vision is None:
            return {}
        # the core node moves the offset back by the seed frame before it reads the pose video; encode that same first frame
        seed = 0 if anchor is None else min(int(anchor.shape[0]), int(getattr(_node_class(self.ANIMATE_NODE), "CONTINUE_MOTION_FRAMES", 1)))
        first = min(max(0, offset - seed), int(pose_video.shape[0]) - 1)
        encoded = _call_node("CLIPVisionEncode", clip_vision=self._clip_vision, image=pose_video[first:first + 1], crop="none")[0]
        return {"clip_vision_output_pose": encoded}


# --- preprocess nodes ----------------------------------------------------------------------
# preprocess/ imports torch, cv2 and comfy at module level, so every node imports it on first
# use, the same way the samplers import torch.

PREPROCESS = "BCVideoNodes"


def _preprocess(module):
    """preprocess.<module>, imported on first use."""
    import importlib

    if __package__:
        return importlib.import_module(".preprocess." + module, __package__)
    return importlib.import_module("preprocess." + module)


_WIDGET_TYPES = {bool: "BOOLEAN", int: "INT", float: "FLOAT", str: "STRING"}


def _config_inputs(config_cls):
    """Widgets for every field of the config dataclass `config_cls`, in field order: the
    field's default, and its range and description from the field metadata ("min", "max",
    "step", "tooltip" or "doc"). A field of any other type than bool / int / float / str
    raises; a str field with "choices" in its metadata is a combo."""
    import dataclasses

    inputs = {}
    for f in dataclasses.fields(config_cls):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            raise TypeError("{}.{} has no default; a config node needs one for every field".format(config_cls.__name__, f.name))
        kind = f.type if isinstance(f.type, type) else {"bool": bool, "int": int, "float": float, "str": str}.get(f.type, type(default))
        if kind not in _WIDGET_TYPES:
            raise TypeError("{}.{} is a {}; a config node can only show bool, int, float and str fields".format(config_cls.__name__, f.name, kind))
        meta = f.metadata
        options = {"default": kind(default)}
        for key in ("min", "max", "step"):
            if key in meta and kind in (int, float):
                options[key] = kind(meta[key])
        tooltip = meta.get("tooltip") or meta.get("doc")
        if tooltip:
            options["tooltip"] = tooltip
        if kind is str and "choices" in meta:
            inputs[f.name] = (list(meta["choices"]), options)
        else:
            inputs[f.name] = (_WIDGET_TYPES[kind], options)
    return inputs


def _config(config_cls, values):
    """`config_cls` built from the widget values that are its fields; the rest are left out."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(config_cls)}
    return config_cls(**{k: v for k, v in values.items() if k in names})


class _ConfigNode:
    """A node that builds a config dataclass from its widgets; the widgets are generated from
    the dataclass, so a new field shows up here without touching this file."""

    MODULE = ""
    CONFIG = ""
    FUNCTION = "build"
    CATEGORY = PREPROCESS

    @classmethod
    def _config_class(cls):
        return getattr(_preprocess(cls.MODULE), cls.CONFIG)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _config_inputs(cls._config_class())}

    def build(self, **values):
        return (self._config_class()(**values),)


class BCVPoseConfig(_ConfigNode):
    MODULE = "pose"
    CONFIG = "PoseConfig"
    RETURN_TYPES = ("POSE_CONFIG",)
    RETURN_NAMES = ("pose_config",)
    DESCRIPTION = "Overrides the Pose Detection tunables. Without it Pose Detection runs with the measured defaults, which are the values shown here. min_keypoint_conf is carried in pose_data and is the only keypoint threshold SAM3 box_keypoint mode and the guards use."


class BCVSAM3Config(_ConfigNode):
    MODULE = "sam3"
    CONFIG = "SAM3Config"
    RETURN_TYPES = ("SAM3_CONFIG",)
    RETURN_NAMES = ("sam3_config",)
    DESCRIPTION = "Overrides the SAM 3.1 Video Track tunables. Without it the node runs with the measured defaults, which are the values shown here. Each tooltip starts with the mode that reads the field; a changed field the run does not read is named in the console."


class BCVPoseDetection:
    @classmethod
    def INPUT_TYPES(cls):
        loader = _preprocess("models.loader")
        return {
            "required": {
                "images": ("IMAGE",),
                "pose_model": (list(loader.POSE_MODELS), {"tooltip": "The wholebody pose model (133 keypoints). Its file is downloaded into models/detection on first use."}),
                "body_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the body sticks in the pose images; 0 leaves the body out, -1 picks it from the frame size"}),
                "hand_stick_width": ("INT", {"default": -1, "min": -1, "max": 20, "step": 1, "tooltip": "Width of the hand sticks in the pose images; 0 leaves the hands out, -1 picks it from the frame size"}),
                "draw_head": ("BOOLEAN", {"default": True, "tooltip": "Whether to draw head keypoints"}),
                "draw_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "A limb is drawn when both its ends reach this confidence; key_frame_body_points uses the same threshold. Drawing only: SAM3 box_keypoint mode and the guards read pose_config.min_keypoint_conf"}),
            },
            "optional": {
                "bboxes": ("BBOX", {"tooltip": "Person boxes (x1, y1, x2, y2), one per frame or one for all. When connected the detector does not run and pose_config.detection_threshold is ignored; box_window and the edge snap still apply."}),
                "pose_config": ("POSE_CONFIG", {"tooltip": "Overrides from Pose Config; the measured defaults without it. Its min_keypoint_conf travels in pose_data to SAM3 and the guards."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "POSEDATA", "BBOX", "STRING")
    RETURN_NAMES = ("pose_images", "pose_data", "bboxes", "key_frame_body_points")
    FUNCTION = "detect"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Wholebody pose on every frame: YOLOv10x finds the person (skipped when bboxes are connected), ViTPose-H or RTMW-l gives the 133 keypoints, which are read against the frames around them, and the pose images are drawn at the frame size. key_frame_body_points is frame 0's confident body keypoints as a points JSON string (KJNodes PointsEditor format). The models are downloaded on first use."

    def detect(self, images, pose_model, body_stick_width, hand_stick_width, draw_head, draw_threshold, bboxes=None, pose_config=None):
        # supplied boxes skip the detector, so it is not loaded either
        detector, model = _preprocess("models.loader").load_pose_models(pose_model, detector=bboxes is None)
        return tuple(_preprocess("pose").pose_detection(images, detector, model, bboxes=bboxes, config=pose_config,
                                                        body_stick_width=body_stick_width, hand_stick_width=hand_stick_width,
                                                        draw_head=draw_head, draw_threshold=draw_threshold))


class BCVSAM3VideoTrack:
    @classmethod
    def INPUT_TYPES(cls):
        sam3 = _preprocess("sam3")
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (list(sam3.MODES), {"default": sam3.MODE_PROMPT, "tooltip": "prompt: SAM 3.1 finds the person from the text prompt alone. box_keypoint: the person is described by pose_data's box and body keypoints (needs pose_data); the v1 behaviour. Inputs the mode does not read are ignored with one console line, so switching needs no rewiring."}),
                "prompt": ("STRING", {"default": sam3.PROMPT, "tooltip": "[prompt] what to segment. The thresholds were measured with the default. Ignored in box_keypoint mode."}),
                "max_objects": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1, "tooltip": "[prompt] how many tracks may be born and kept; above 1 the [prompt, max_objects > 1] config fields apply. Ignored in box_keypoint mode, which tracks the one person the pose describes."}),
                "object_index": ("INT", {"default": -1, "min": -1, "max": 15, "step": 1, "tooltip": "[prompt] which tracked object the mask is: -1 the union of every tracked object, k object k (numbered from 0). Ignored in box_keypoint mode."}),
            },
            "optional": {
                "pose_data": ("POSEDATA", {"tooltip": "[box_keypoint] the boxes and keypoints the prompt is built from, and the keypoint threshold (Pose Config's min_keypoint_conf). Ignored in prompt mode."}),
                "bboxes": ("BBOX", {"tooltip": "[box_keypoint] replaces pose_data's person boxes, used as given (every frame counts as detected). Ignored in prompt mode. If the boxes come from Pose Detection, connect pose_data instead: its bboxes output marks frames with no detected person as full-frame boxes."}),
                "positive_coords": ("STRING", {"forceInput": True, "tooltip": "[box_keypoint] extra positive points on frame 0, points JSON (KJNodes PointsEditor / easy-sam3). They win: a derived negative within 4% of the box diagonal is dropped. Ignored in prompt mode. In box_keypoint mode, connecting key_frame_body_points here only repeats the automatic points."}),
                "negative_coords": ("STRING", {"forceInput": True, "tooltip": "[box_keypoint] extra negative points on frame 0, same format. They win: a keypoint or body point within 4% of the box diagonal is dropped. Ignored in prompt mode."}),
                "sam3_config": ("SAM3_CONFIG", {"tooltip": "Overrides from SAM 3 Config; the measured defaults without it. Only the fields tagged with the current mode are read."}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "track"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Segments the person on every frame with SAM 3.1 and its tracker memory, from a text prompt or from Pose Detection's boxes and keypoints. The mask covers every frame, including the ones before the person was found. The checkpoint is ComfyUI's own SAM 3.1, downloaded into models/checkpoints on first use."

    def track(self, images, mode, prompt, max_objects, object_index, pose_data=None, bboxes=None, positive_coords=None,
              negative_coords=None, sam3_config=None):
        sam3 = _preprocess("sam3")
        mask = sam3.track(sam3.load_sam3(), images, pose_data=pose_data, bboxes=bboxes, positive_coords=positive_coords,
                          negative_coords=negative_coords, mode=mode, prompt=prompt, max_objects=max_objects,
                          object_index=object_index, config=sam3_config)
        return (mask,)


class BCVFaceCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "pose_data": ("POSEDATA",),
                "face_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1, "tooltip": "Pixels added on every side of the face box before it is cut and resized to 512x512. Ignored when face_bboxes is connected"}),
            },
            "optional": {
                "face_bboxes": ("BBOX", {"tooltip": "Face boxes (x1, y1, x2, y2), one per frame or one for all, cut as they are: pose_data's face keypoints and face_padding are then not used."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BBOX")
    RETURN_NAMES = ("face_images", "face_bboxes")
    FUNCTION = "crop"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Cuts the face out of every frame, from the face keypoints in pose_data or from face_bboxes, as 512x512 images (Wan Animate's face_video)."

    def crop(self, images, pose_data, face_padding, face_bboxes=None):
        return tuple(_preprocess("face").crop_faces(images, pose_data, face_padding=face_padding, face_bboxes=face_bboxes))


def _guard_inputs(config_name, switch, tooltip):
    guard = _preprocess("guard")
    return {switch: ("BOOLEAN", {"default": True, "tooltip": tooltip}), **_config_inputs(getattr(guard, config_name))}


POSE_GUARD_TOOLTIP = "Stop the workflow when a pose check fails (incomplete skeleton, torso jump, subject switch). Off still measures and reports every check."
MASK_GUARD_TOOLTIP = "Stop the workflow when a mask check fails (empty, leaking, fragmented, keypoints outside, unstable). Off still measures and reports every check."


class BCVPoseGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"pose_data": ("POSEDATA",), **_guard_inputs("PoseGuardConfig", "pose_guard", POSE_GUARD_TOOLTIP)}}

    RETURN_TYPES = ("POSEDATA", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("pose_data", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Checks the pose frame by frame (missing detections, incomplete skeletons, torso jumps, subject switches, a second person). A failed check stops the workflow with the report when pose_guard is on; 'metrics' has every measurement per frame and 'timeline' plots them. pose_data passes through."

    def check(self, pose_data, pose_guard, **thresholds):
        guard = _preprocess("guard")
        return tuple(guard.check_pose(pose_data, _config(guard.PoseGuardConfig, thresholds), enabled=pose_guard))


class BCVMaskGuard:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"mask": ("MASK",), "pose_data": ("POSEDATA",),
                             **_guard_inputs("MaskGuardConfig", "mask_guard", MASK_GUARD_TOOLTIP)}}

    RETURN_TYPES = ("MASK", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = PREPROCESS
    DESCRIPTION = "Checks the mask frame by frame against the pose it belongs to (empty, leaking, fragmented, keypoints outside the mask, unstable). A failed check stops the workflow with the report when mask_guard is on; 'metrics' has every measurement per frame and 'timeline' plots them. The mask passes through."

    def check(self, mask, pose_data, mask_guard, **thresholds):
        guard = _preprocess("guard")
        return tuple(guard.check_mask(mask, pose_data, _config(guard.MaskGuardConfig, thresholds), enabled=mask_guard))


# --- Wan Animate wrappers ------------------------------------------------------------------


class BCVWanAnimatePreprocess:
    @classmethod
    def INPUT_TYPES(cls):
        pose = BCVPoseDetection.INPUT_TYPES()["required"]
        sam3 = BCVSAM3VideoTrack.INPUT_TYPES()["required"]
        face = BCVFaceCrop.INPUT_TYPES()["required"]
        return {
            "required": {
                "images": ("IMAGE",),
                **{name: pose[name] for name in ("pose_model", "body_stick_width", "hand_stick_width", "draw_head", "draw_threshold")},
                "face_padding": face["face_padding"],
                **{name: sam3[name] for name in ("mode", "prompt")},
            },
            "optional": {
                "pose_config": BCVPoseDetection.INPUT_TYPES()["optional"]["pose_config"],
                "sam3_config": BCVSAM3VideoTrack.INPUT_TYPES()["optional"]["sam3_config"],
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "POSEDATA", "BBOX", "STRING", "BBOX")
    RETURN_NAMES = ("pose_images", "face_images", "mask", "pose_data", "bboxes", "key_frame_body_points", "face_bboxes")
    FUNCTION = "process"
    CATEGORY = "BCVideoNodes/Wan/Animate"
    DESCRIPTION = "The whole WanAnimate preprocess in one node: Pose Detection, SAM 3.1 Video Track and Face Crop chained, computing exactly what the three nodes compute when wired by hand. In prompt mode the mask comes from the text prompt alone (one track, the union mask); in box_keypoint mode from the pose, with no extra boxes or points. The models are downloaded on first use. Feed it frames already at the generation size."

    def process(self, images, pose_model, body_stick_width, hand_stick_width, draw_head, draw_threshold, face_padding,
                mode, prompt, pose_config=None, sam3_config=None):
        pose_images, pose_data, bboxes, key_points = BCVPoseDetection().detect(
            images, pose_model, body_stick_width, hand_stick_width, draw_head, draw_threshold, pose_config=pose_config)
        # prompt mode segments from the text alone; pose_data is connected only where it is read
        box_keypoint = mode == _preprocess("sam3").MODE_BOX_KEYPOINT
        (mask,) = BCVSAM3VideoTrack().track(images, mode, prompt, 1, -1, pose_data=pose_data if box_keypoint else None,
                                            sam3_config=sam3_config)
        face_images, face_bboxes = BCVFaceCrop().crop(images, pose_data, face_padding)
        return (pose_images, face_images, mask, pose_data, bboxes, key_points, face_bboxes)


class BCVWanAnimatePreprocessGuard:
    @classmethod
    def INPUT_TYPES(cls):
        pose = BCVPoseGuard.INPUT_TYPES()["required"]
        mask = BCVMaskGuard.INPUT_TYPES()["required"]
        return {
            "required": {
                "mask": mask["mask"],
                "pose_data": pose["pose_data"],
                "pose_guard": pose["pose_guard"],
                "mask_guard": mask["mask_guard"],
                **{k: v for k, v in pose.items() if k not in ("pose_data", "pose_guard")},
                **{k: v for k, v in mask.items() if k not in ("mask", "pose_data", "mask_guard")},
            },
        }

    RETURN_TYPES = ("MASK", "POSEDATA", "STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("mask", "pose_data", "report", "metrics", "timeline")
    FUNCTION = "check"
    CATEGORY = "BCVideoNodes/Wan/Animate"
    DESCRIPTION = "Pose Guard and Mask Guard in one node, with one report: checks the pose and the mask of WanAnimate Preprocess frame by frame. Wire it between the preprocess and the sampler: a failed check of an enabled guard stops the workflow with the report; 'metrics' has every measurement per frame and 'timeline' plots them."

    def check(self, mask, pose_data, pose_guard, mask_guard, **thresholds):
        guard = _preprocess("guard")
        # each group measures without stopping; the combined report decides
        _, _, pose_metrics, _ = guard.check_pose(pose_data, _config(guard.PoseGuardConfig, thresholds), enabled=pose_guard,
                                                 stop_on_fail=False)
        _, _, mask_metrics, _ = guard.check_mask(mask, pose_data, _config(guard.MaskGuardConfig, thresholds), enabled=mask_guard,
                                                 stop_on_fail=False)
        report, metrics, timeline = guard.combine_guards(pose_metrics, mask_metrics)
        return (mask, pose_data, report, metrics, timeline)


NODE_CLASS_MAPPINGS = {
    "BCVWanAnimateLongVideoSampler": BCVWanAnimateLongVideoSampler,
    "BCVWanAnimate2LongVideoSampler": BCVWanAnimate2LongVideoSampler,
    "BCVPoseDetection": BCVPoseDetection,
    "BCVPoseConfig": BCVPoseConfig,
    "BCVSAM3VideoTrack": BCVSAM3VideoTrack,
    "BCVSAM3Config": BCVSAM3Config,
    "BCVFaceCrop": BCVFaceCrop,
    "BCVPoseGuard": BCVPoseGuard,
    "BCVMaskGuard": BCVMaskGuard,
    "BCVWanAnimatePreprocess": BCVWanAnimatePreprocess,
    "BCVWanAnimatePreprocessGuard": BCVWanAnimatePreprocessGuard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BCVWanAnimateLongVideoSampler": "Wan Animate Long Video Sampler",
    "BCVWanAnimate2LongVideoSampler": "Wan Animate 2 Long Video Sampler",
    "BCVPoseDetection": "Pose Detection",
    "BCVPoseConfig": "Pose Config",
    "BCVSAM3VideoTrack": "SAM 3.1 Video Track",
    "BCVSAM3Config": "SAM 3 Config",
    "BCVFaceCrop": "Face Crop",
    "BCVPoseGuard": "Pose Guard",
    "BCVMaskGuard": "Mask Guard",
    "BCVWanAnimatePreprocess": "WanAnimate Preprocess",
    "BCVWanAnimatePreprocessGuard": "WanAnimate Preprocess Guard",
}
