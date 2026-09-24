"""Wan Animate / SCAIL-2 long video in one node: chained fixed-size chunks.

One node per core conditioning node: BCVWanAnimateLongVideoSampler wraps
WanAnimateToVideo (Wan 2.2 Animate), BCVWanAnimate2LongVideoSampler wraps
WanAnimate2ToVideo and BCVSCAIL2LongVideoSampler wraps WanSCAILToVideo
(SCAIL-2). The chunk loop is pipelines/long_video.py.
"""

# comfy.* is imported inside the functions that use it, so this
# module (and the package __init__) imports without a ComfyUI install.
from ..libs.chunking import FIT, FULL, LAST_CHUNK
from ..libs.sigmas import WAN_BETA, WAN_DPMPP
from ..libs.video import LAST_FRAME, TAIL_PADDING


def _combo_default(options, preferred):
    return preferred if preferred in options else options[0]


class _LongVideoSampler:
    """The node surface the samplers share; ``generate`` hands the run to the
    chunk loop in pipelines/long_video.py.

    A subclass names the core conditioning node it wraps (``ANIMATE_NODE``)
    and lists the inputs that pass straight through to it
    (``_animate_inputs``). How many frames that node trims back off every
    chained chunk is the ``prepare`` of its adapter
    (models/wan_*/adapter.py). Everything else - widgets, sampling
    stack, loop, output - is identical; the class attributes below set a
    subclass's widget defaults.
    """

    ANIMATE_NODE = ""
    MODEL_TOOLTIP = ""
    DEFAULT_CHUNK = 81
    DEFAULT_SHIFT = 5.0
    DEFAULT_SAMPLER = "euler"
    DEFAULT_SCHEDULER = WAN_BETA
    DEFAULT_STEPS = 6
    DEFAULT_WIDTH = 720
    DEFAULT_HEIGHT = 1280
    DEFAULT_LAST_CHUNK = FIT
    SIZE_MIN = 16
    SIZE_STEP = 2
    SIZE_TOOLTIP = "Multiples of 16 are ideal; the VAE crops to a multiple of 8."

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

        samplers = list(comfy.samplers.SAMPLER_NAMES) + [WAN_DPMPP]
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
                "width": ("INT", {"default": cls.DEFAULT_WIDTH, "min": cls.SIZE_MIN, "max": max_res, "step": cls.SIZE_STEP, "tooltip": cls.SIZE_TOOLTIP}),
                "height": ("INT", {"default": cls.DEFAULT_HEIGHT, "min": cls.SIZE_MIN, "max": max_res, "step": cls.SIZE_STEP, "tooltip": cls.SIZE_TOOLTIP}),
                "frames_per_chunk": ("INT", {"default": cls.DEFAULT_CHUNK, "min": 5, "max": max_res, "step": 4, "tooltip": "Frames sampled per chunk, rounded down to 4k+1. 81 for 24 GB, 49 for 16 GB, 33 for 12 GB are sane starts."}),
                "total_frames": ("INT", {"default": 81, "min": 0, "max": 100000, "tooltip": "Exact output length. 0 = the pose video's frame count."}),
                "shift": ("FLOAT", {"default": cls.DEFAULT_SHIFT, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "ModelSamplingSD3 shift, applied to the model before the schedule is built."}),
                "sampler_name": (samplers, {"default": _combo_default(samplers, cls.DEFAULT_SAMPLER), "tooltip": "ComfyUI's samplers (as KSamplerSelect runs them), plus wan_dpmpp: the DPM-Solver++ 2M the official Wan pipelines sample with (Wan's FlowDPMSolverMultistepScheduler), run as core's dpmpp_2m_sde with eta 0 (no noise) and the midpoint solver. Its official pairing is scheduler simple at the same shift: the native Wan pipelines' sigmas (evenly spaced, then shifted)."}),
                "scheduler": (schedulers, {"default": _combo_default(schedulers, cls.DEFAULT_SCHEDULER), "tooltip": "ComfyUI schedulers, plus wan_beta: the sigmas WanVideoWrapper's euler/beta samples with (diffusers beta sigmas), evenly spread with a small last step. Ignored when sigmas_override is connected."}),
                "steps": ("INT", {"default": cls.DEFAULT_STEPS, "min": 1, "max": 10000, "tooltip": "Sampling steps of the schedule every chunk samples with. Ignored when sigmas_override is connected."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Below 1.0 the schedule is the last steps of a longer one (steps / denoise), as BasicScheduler builds it. Ignored when sigmas_override is connected."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale. At 1.0 the negative prompt is not evaluated: one model pass per step, the setting for distilled checkpoints and distill LoRAs. Above 1.0 every step also runs the negative prompt and pushes away from it, at twice the cost."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Noise seed of the first chunk; seed_mode says which seed each later chunk uses."}),
                "seed_mode": (["increment", "fixed"], {"default": "increment", "tooltip": "increment: chunk i uses seed + i. fixed: every chunk uses seed."}),
                **required,
                # after every node's own widgets: saved workflows store widget values by position
                "last_chunk": (list(LAST_CHUNK), {"default": cls.DEFAULT_LAST_CHUNK, "tooltip": "How long the last chunk runs. fit: it shrinks to the frames still needed (snapped up to 4k+1). full: it runs the full frames_per_chunk. min29: like fit, but never fewer than 29 frames (capped at frames_per_chunk), as the official Wan-Animate-2 pads its last clip. Driving inputs the last chunk runs past are extended as tail_padding says. Either way the output is exactly total_frames; the extra frames are cut."}),
                "tail_padding": (list(TAIL_PADDING), {"default": LAST_FRAME, "tooltip": "How the driving inputs are extended past their last frame when a chunk needs frames beyond them (the fit snap-up, last_chunk full or min29, or total_frames longer than the input). last_frame: the last frame is repeated (the motion stops). ping_pong: the input plays backwards from its end, as the official Wan Animate code pads (the motion continues along the same path in reverse). Padded frames past total_frames are cut from the output; they only affect the real frames of the same chunk. When total_frames is longer than the input, the output past the input's end is the padding."}),
            },
            "optional": {
                **optional,
                "sigmas_override": ("SIGMAS", {"tooltip": "Replaces the internal schedule; scheduler, steps and denoise are then ignored. shift still applies to the model."}),
                # the last widget: saved workflows store widget values by position
                "color_anchor_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 = off. Above 0, every chunk after the first gets one colour transform (per-channel Lab mean and std) that maps its regenerated overlap frames onto the frames it was seeded with, applied to the whole chunk before it is trimmed and carried, so the colours stay anchored to the first chunk; the value blends it in (1 = the full transform). In replacement mode (Wan Animate with background_video and character_mask, SCAIL-2 with replacement_mode) only the character is measured and corrected, since the background comes from the source every chunk."}),
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
        last_chunk,
        tail_padding,
        sigmas_override=None,
        color_anchor_strength=0.0,
        **animate_inputs,
    ):
        from ..pipelines import long_video

        return long_video.generate(self.ANIMATE_NODE, type(self).__name__, model, positive, negative, vae, reference_image,
                                   pose_video, width, height, frames_per_chunk, total_frames, shift, sampler_name, scheduler,
                                   steps, denoise, cfg, seed, seed_mode, last_chunk, tail_padding, sigmas_override,
                                   color_anchor_strength, animate_inputs)


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
            "reference_image_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scales the attention values of the reference image's latent frame in every generation block: how strongly the generated frames draw on the reference. 1.0 as trained; below loosens identity and appearance (lets the prompt restyle), above tightens them against drift."}),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scales the pose branch's attention values in every generation block: how closely the motion follows the pose video. 1.0 as trained; below weakens it, above amplifies it. 0.0 mutes the values, but the pose tokens still take attention, so it is not the same as no pose."}),
            "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Sampling percent (0 = the start, pure noise; 1 = the end) where the pose branch starts. Outside pose_start_percent .. pose_end_percent the model runs without the pose branch, which also makes those steps faster. Must not exceed pose_end_percent."}),
            "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Sampling percent where the pose branch ends. Motion is mostly set early, so e.g. 0.7 loosens fine detail while keeping the choreography."}),
            "attn_log_scale": ("FLOAT", {"default": -1.3, "min": -10.0, "max": 10.0, "step": 0.1, "tooltip": "Logit bias on attention to the seed frame (latent frame 1), the official log_scale. -1.3 is the distilled checkpoint's config; use 0.0 for the base checkpoint. Core has no equivalent (0.0 is core's behaviour)."}),
        }
        optional = {
            "positive_pose": ("CONDITIONING", {"tooltip": "Prompt for the pose branch. Defaults to positive. The official pipeline never leaves it empty (default: 人物动作的参考视频)."}),
            "clip_vision_output": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the reference image."}),
            "clip_vision_output_pose": ("CLIP_VISION_OUTPUT", {"tooltip": "CLIP vision of the pose video's first frame, used for every chunk. Defaults to clip_vision_output. Connect clip_vision instead to re-encode per chunk."}),
            "clip_vision": ("CLIP_VISION", {"tooltip": "When connected, the pose CLIP embedding is re-encoded from the first frame of each chunk's pose window, as the official pipeline does; clip_vision_output_pose is then ignored."}),
        }
        return required, optional


class BCVSCAIL2LongVideoSampler(_LongVideoSampler):
    ANIMATE_NODE = "WanSCAILToVideo"
    MODEL_TOOLTIP = "SCAIL-2 model. LoRA (lightx2v distill, SCAIL-2 DPO / relight) and model patches pass through unchanged; shift is applied here."
    DEFAULT_SHIFT = 8.0
    DEFAULT_SCHEDULER = "simple"
    DEFAULT_LAST_CHUNK = FULL
    DEFAULT_WIDTH = 704
    DEFAULT_HEIGHT = 1280
    SIZE_MIN = 32
    SIZE_STEP = 32
    SIZE_TOOLTIP = "Must be divisible by 32 (the pose runs at half resolution through the /16 patch grid). 704x1280 (the authors: replacement and pose-driven are better at 704p) or 512x896 (less VRAM)."
    CATEGORY = "BCVideoNodes/SCAIL"
    DESCRIPTION = ("Generates an arbitrarily long SCAIL-2 video (animation or replacement mode) by chaining fixed-size "
                   "chunks internally, each seeded with the previous chunk's last previous_frame_count frames. last_chunk defaults "
                   "to full here: every chunk, the last one included, runs the full frames_per_chunk (SCAIL-2 was trained on "
                   "65-81 frame segments); fit shortens the last chunk to the frames still needed. Either way the output is "
                   "cut to total_frames (or the pose video length) exactly. The defaults (shift 8, euler / simple, 6 steps, cfg 1, "
                   "for a distill LoRA) are a starting point, not the SCAIL-2 authors' recipe. They give the sigmas the ComfyUI "
                   "SCAIL-2 templates sample with: the templates set ModelSamplingSD3 to 5, but their BasicScheduler takes the model "
                   "before that node, so their schedule runs at the model's default shift 8 (1, .9757, .9413, .8889, .8005, .616, 0). "
                   "The templates sample euler / simple, 6 steps, cfg 1 with lightx2v 0.8 + the SCAIL-2 DPO LoRA 1.0, and 40 steps, "
                   "cfg 5 without lightx2v (the DPO LoRA stays). Other references: the zai-org/SCAIL-2 README samples uni_pc, 40 "
                   "steps, shift 3, cfg 5, and with the lightx2v LoRA 8 steps, shift 1, cfg 1; its SAT config shift 5, 50 steps, cfg 4.")

    @classmethod
    def _animate_inputs(cls, max_res):
        required = {
            "clip_vision": ("CLIP_VISION", {"tooltip": "CLIP vision model (clip_vision_h). The reference is encoded once per run, stretched (crop none) as SCAIL-2 was trained; in replacement mode with the character on black, as the authors require."}),
            "pose_video_mask": ("IMAGE", {"tooltip": "Colored driving mask from SCAIL-2 Colored Mask / SCAIL-2 Preprocess, as long as pose_video: the character in its identity colour, on black (animation) or white (replacement)."}),
            "reference_image_mask": ("IMAGE", {"tooltip": "Colored reference mask from SCAIL-2 Colored Mask / SCAIL-2 Preprocess: the character in its identity colour, on white (animation) or black (replacement)."}),
            "replacement_mode": ("BOOLEAN", {"default": False, "tooltip": "False: animation mode, the reference character is animated by the driving video. True: replacement mode, the character replaces the person in the driving video. Must match the mode the masks were rendered for; a mismatch is an error."}),
            "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Scales the VAE-encoded pose video the model is conditioned on (core multiplies its latent). 1.0 as trained. The colored driving mask is not scaled."}),
            "pose_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Sampling percent (0 = the start, pure noise; 1 = the end) from which the pose video conditions the model; before it the model samples without the pose video. The colored driving mask conditions every step. Passed as the core node's pose_start; must not exceed pose_end_percent."}),
            "pose_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Sampling percent after which the pose video no longer conditions the model. Passed as the core node's pose_end."}),
            "previous_frame_count": ("INT", {"default": 5, "min": 1, "max": max_res, "step": 4, "tooltip": "Frames of the previous chunk that seed the next one and are trimmed back off: the overlap between chunks. SCAIL-2 was trained with 5. Snapped down to the 4k+1 grid; must be smaller than frames_per_chunk."}),
        }
        return required, {}
