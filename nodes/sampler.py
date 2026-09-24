"""Wan Animate long video in one node: chained fixed-size chunks.

One node per core conditioning node: BCVWanAnimateLongVideoSampler wraps
WanAnimateToVideo (Wan 2.2 Animate) and BCVWanAnimate2LongVideoSampler wraps
WanAnimate2ToVideo. The chunk loop is pipelines/long_video.py.
"""

# comfy.* is imported inside the functions that use it, so this
# module (and the package __init__) imports without a ComfyUI install.
from ..libs.sigmas import WAN_BETA


def _combo_default(options, preferred):
    return preferred if preferred in options else options[0]


class _LongVideoSampler:
    """The node surface both samplers share; ``generate`` hands the run to the
    chunk loop in pipelines/long_video.py.

    A subclass names the core conditioning node it wraps (``ANIMATE_NODE``)
    and lists the inputs that pass straight through to it
    (``_animate_inputs``). How many frames that node trims back off every
    chained chunk is the ``prepare`` of its adapter
    (models/wan_animate*/adapter.py). Everything else - widgets, sampling
    stack, loop, output - is identical.
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
        from ..pipelines import long_video

        return long_video.generate(self.ANIMATE_NODE, type(self).__name__, model, positive, negative, vae, reference_image,
                                   pose_video, width, height, frames_per_chunk, total_frames, shift, sampler_name, scheduler,
                                   steps, denoise, cfg, seed, seed_mode, sigmas_override, animate_inputs)


class BCVWanAnimateLongVideoSampler(_LongVideoSampler):
    ANIMATE_NODE = "WanAnimateToVideo"
    MODEL_TOOLTIP = "Wan 2.2 Animate model. LoRA and model patches pass through unchanged; shift is applied here."
    DEFAULT_SHIFT = 8.0
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


class BCVWanAnimate2LongVideoSampler(_LongVideoSampler):
    ANIMATE_NODE = "WanAnimate2ToVideo"
    MODEL_TOOLTIP = "Wan Animate 2 model. LoRA, WanAnimate2Cache and context-window patches pass through unchanged; shift is applied here."
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
