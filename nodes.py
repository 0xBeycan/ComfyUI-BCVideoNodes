"""Wan Animate 2 long video in one node: chained fixed-size chunks.

Every chunk after the first is seeded with the previous chunk's last frame
(WanAnimate2ToVideo's continue_motion) and the pose video is read from the
returned video_frame_offset, so the pose stays aligned across the whole run.
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

LOG_PREFIX = "[WanAnimate2LongVideoSampler]"
ANIMATE_NODE = "WanAnimate2ToVideo"
ANIMATE_OUTPUTS = 6  # positive, negative, latent, trim_latent, trim_image, video_frame_offset
UPDATE_HINT = "Update ComfyUI: this node needs the WanAnimate2ToVideo that returns trim_latent / trim_image / video_frame_offset."


def _node_class(node_id):
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(node_id)
    if cls is None:
        raise RuntimeError("Core node '{}' is not registered. {}".format(node_id, UPDATE_HINT))
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


class WanAnimate2LongVideoSampler:
    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        import nodes as comfy_nodes

        samplers = list(comfy.samplers.SAMPLER_NAMES)
        schedulers = list(comfy.samplers.SCHEDULER_NAMES)
        max_res = comfy_nodes.MAX_RESOLUTION
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Wan Animate 2 model. LoRA, WanAnimate2Cache and context-window patches pass through unchanged; shift is applied here."}),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE", {"tooltip": "The character to animate."}),
                "pose_video": ("IMAGE", {"tooltip": "Driving video. With total_frames = 0 its frame count is the output length."}),
                "width": ("INT", {"default": 720, "min": 16, "max": max_res, "step": 2, "tooltip": "Multiples of 16 are ideal; the VAE crops to a multiple of 8."}),
                "height": ("INT", {"default": 1280, "min": 16, "max": max_res, "step": 2, "tooltip": "Multiples of 16 are ideal; the VAE crops to a multiple of 8."}),
                "frames_per_chunk": ("INT", {"default": 81, "min": 5, "max": max_res, "step": 4, "tooltip": "Frames sampled per chunk, rounded down to 4k+1. 81 for 24 GB, 49 for 16 GB, 33 for 12 GB are sane starts."}),
                "total_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "tooltip": "Exact output length. 0 = the pose video's frame count."}),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "ModelSamplingSD3 shift, applied to the model before the schedule is built."}),
                "sampler_name": (samplers, {"default": _combo_default(samplers, "lcm")}),
                "scheduler": (schedulers, {"default": _combo_default(schedulers, "simple"), "tooltip": "Ignored when sigmas_override is connected."}),
                "steps": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "Ignored when sigmas_override is connected."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Ignored when sigmas_override is connected."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "seed_mode": (["increment", "fixed"], {"default": "increment", "tooltip": "increment: chunk i uses seed + i. fixed: every chunk uses seed."}),
                "reference_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "positive_pose": ("CONDITIONING", {"tooltip": "Prompt for the pose branch. Defaults to positive."}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the reference image."}),
                "clip_vision_output_pose": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the pose video's first frame. Defaults to clip_vision_output."}),
                "sigmas_override": ("SIGMAS", {"tooltip": "Replaces the internal schedule; scheduler, steps and denoise are then ignored. shift still applies to the model."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("images", "frame_count", "chunk_plan")
    FUNCTION = "generate"
    CATEGORY = "WanAnimate2"
    DESCRIPTION = "Generates an arbitrarily long Wan Animate 2 video by chaining fixed-size chunks internally. Output length equals total_frames (or the pose video length) exactly."

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
        reference_image_strength,
        pose_strength,
        pose_start_percent,
        pose_end_percent,
        positive_pose=None,
        clip_vision_output=None,
        clip_vision_output_pose=None,
        sigmas_override=None,
    ):
        import torch
        import comfy.model_management
        import comfy.utils

        if pose_start_percent > pose_end_percent:
            raise ValueError("pose_start_percent ({}) must not be greater than pose_end_percent ({}).".format(pose_start_percent, pose_end_percent))

        animate_cls = _node_class(ANIMATE_NODE)
        if len(animate_cls.RETURN_TYPES) < ANIMATE_OUTPUTS:
            raise RuntimeError("{} returns {} outputs, {} expected. {}".format(ANIMATE_NODE, len(animate_cls.RETURN_TYPES), ANIMATE_OUTPUTS, UPDATE_HINT))

        # The node keeps the last CONTINUE_MOTION_FRAMES frames of continue_motion
        # and trims their decoded span back off every chained chunk; that span
        # is the overlap. Read from the class so a core change is picked up.
        motion_frames = int(getattr(animate_cls, "CONTINUE_MOTION_FRAMES", 1))
        overlap = overlap_for_motion_frames(motion_frames)

        pose_frames = int(pose_video.shape[0])
        if pose_frames < 1:
            raise ValueError("pose_video has no frames.")
        total = int(total_frames) if total_frames > 0 else pose_frames
        if total > pose_frames:
            logging.warning("%s total_frames (%d) exceeds pose_video length (%d): the last pose frame is held for the remaining %d frames.",
                            LOG_PREFIX, total, pose_frames, total - pose_frames)
            # WanAnimate2ToVideo holds the last frame within a chunk, but errors
            # once the offset itself runs past the pose video. Pad up front.
            pose_video = torch.cat((pose_video, pose_video[-1:].expand(total - pose_frames, -1, -1, -1)), dim=0)

        plan = plan_chunks(total, frames_per_chunk, overlap)
        logging.info("%s chunk plan: %s", LOG_PREFIX, format_plan(plan, produced_frames(plan, overlap), total, pose_frames, overlap))

        patched = _call_node("ModelSamplingSD3", model=model, shift=shift)[0]
        if sigmas_override is not None:
            sigmas = sigmas_override
            logging.info("%s sigmas_override connected: scheduler / steps / denoise widgets are ignored.", LOG_PREFIX)
        else:
            sigmas = _call_node("BasicScheduler", model=patched, scheduler=scheduler, steps=steps, denoise=denoise)[0]
        if sigmas.numel() < 2:
            raise ValueError("The sigma schedule is empty (denoise too low, or an empty sigmas_override).")
        sampler = _call_node("KSamplerSelect", sampler_name=sampler_name)[0]

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

            animate = _call_node(
                ANIMATE_NODE,
                positive=positive,
                negative=negative,
                vae=vae,
                width=width,
                height=height,
                length=length,
                batch_size=1,
                reference_image=reference_image,
                pose_video=pose_video,
                clip_vision_output=clip_vision_output,
                positive_pose=positive_pose,
                clip_vision_output_pose=clip_vision_output_pose,
                continue_motion=anchor,
                video_frame_offset=offset,
                pose_strength=pose_strength,
                pose_start_percent=pose_start_percent,
                pose_end_percent=pose_end_percent,
                reference_image_strength=reference_image_strength,
            )
            if len(animate) < ANIMATE_OUTPUTS:
                raise RuntimeError("{} returned {} outputs, {} expected. {}".format(ANIMATE_NODE, len(animate), ANIMATE_OUTPUTS, UPDATE_HINT))
            chunk_positive, chunk_negative, latent, trim_latent, trim_image, offset = animate[:ANIMATE_OUTPUTS]

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
                logging.warning("%s %s trimmed %d frames, planner assumed %d; using %d from here on.", LOG_PREFIX, ANIMATE_NODE, trim_image, overlap, trim_image)
                overlap = trim_image
            progress.update(1)

        images = torch.cat(chunks, dim=0) if len(chunks) > 1 else chunks[0]
        images = images[:total]
        plan_text = format_plan(lengths, produced, total, pose_frames, overlap)
        if lengths != plan:
            logging.info("%s ran: %s", LOG_PREFIX, plan_text)
        return (images, int(images.shape[0]), plan_text)


NODE_CLASS_MAPPINGS = {
    "WanAnimate2LongVideoSampler": WanAnimate2LongVideoSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanAnimate2LongVideoSampler": "Wan Animate 2 Long Video Sampler",
}
