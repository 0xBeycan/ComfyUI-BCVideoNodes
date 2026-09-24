"""The sampler test fakes: stand-ins for the core nodes the chunk loop calls (V3- and
V1-shaped), the concat-mask references, and the `node_module` fixture, which stubs ComfyUI
(comfy.*, nodes), binds the pack as `walong` and returns the `sampler` Names the test bodies
read the sampler through (tests/names.py). torch is real, so the tensor plumbing is exercised.
"""

import importlib
import os
import sys
import types

import pytest

from names import Names, Ref, refs

torch = pytest.importorskip("torch")

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANIMATE1 = "BCVWanAnimateLongVideoSampler"
ANIMATE2 = "BCVWanAnimate2LongVideoSampler"
SCAIL2 = "BCVSCAIL2LongVideoSampler"
CONTINUE_MOTION_FRAMES = 1  # WanAnimate2ToVideo's class constant
LATENT_DOWN = 8

sampler = Names("sampler", {
    **refs("nodes.sampler", "BCVWanAnimateLongVideoSampler", "BCVWanAnimate2LongVideoSampler", "BCVSCAIL2LongVideoSampler",
           "_combo_default"),
    "_StepLogger": Ref("pipelines.long_video", "_StepLogger"),
    "_fix_replacement_mask": Ref("models.wan_animate.mask_repair", "fix_replacement_mask"),
    "_replacement_mask_rows": Ref("models.wan_animate.mask_repair", "replacement_mask_rows"),
    "_seed_frame_attention_bias": Ref("models.wan_animate2.attention", "seed_frame_attention_bias"),
    "_call_node": Ref("models.common.core_nodes", "call_node"),
    "_with_schema_defaults": Ref("models.common.core_nodes", "with_schema_defaults"),
    **refs("libs.sigmas", "wan_beta_sigmas"),
}, alias="walong")


class FakeVAE:
    """Frame-count faithful: 4k+1 pixels <-> k+1 latents. Pixel content is
    the frame index so alignment can be checked end to end."""

    def encode(self, pixels):
        frames = pixels.shape[0]
        latents = ((frames - 1) // 4) + 1
        return torch.zeros(1, 16, latents, pixels.shape[1] // LATENT_DOWN, pixels.shape[2] // LATENT_DOWN)

    def decode(self, latent):
        latents = latent.shape[2]
        frames = (latents - 1) * 4 + 1
        h = latent.shape[3] * LATENT_DOWN
        w = latent.shape[4] * LATENT_DOWN
        return torch.full((1, frames, h, w, 3), 1.5)  # past the valid range on purpose, so the clamp is observable

    # comfy/sd.py:510
    process_output = staticmethod(lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0))


class FakeModel:
    def __init__(self):
        self.model_options = {"transformer_options": {}}

    def clone(self):
        import copy
        clone = FakeModel()
        clone.model_options = copy.deepcopy(self.model_options)
        if hasattr(self, "shift"):
            clone.shift = self.shift
        return clone


class Calls:
    animate = []
    sampler = []


class FakeNodeOutput:
    def __init__(self, *args):
        self.args = args


# --- V3-shaped fakes: FUNCTION resolves to a classmethod, return has .args ---

class FakeWanAnimateToVideo:
    """Mirrors comfy_extras/nodes_wan.py WanAnimateToVideo.execute where the
    loop depends on it: the overlap is the continue_motion_max_frames widget,
    the offset moves back by the kept frames, and a pose video the offset has
    run past is dropped (not an error)."""

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"length": ("INT", {"default": 77}), "batch_size": ("INT", {"default": 1}),
                             "continue_motion_max_frames": ("INT", {"default": 5}), "video_frame_offset": ("INT", {"default": 0})},
                "optional": {"continue_motion": ("IMAGE", {}), "face_video": ("IMAGE", {}), "background_video": ("IMAGE", {}), "character_mask": ("MASK", {})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, positive, negative, vae, width, height, length, batch_size, continue_motion_max_frames, video_frame_offset,
                           reference_image=None, clip_vision_output=None, face_video=None, pose_video=None, continue_motion=None,
                           background_video=None, character_mask=None):
        latent_length = ((length - 1) // 4) + 1
        trim_latent = vae.encode(reference_image[:length]).shape[2]
        ref_motion_latent_length = 0
        if continue_motion is not None:
            continue_motion = continue_motion[-continue_motion_max_frames:]
            video_frame_offset = max(0, video_frame_offset - continue_motion.shape[0])
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1
        if pose_video is not None:
            pose_video = None if pose_video.shape[0] <= video_frame_offset else pose_video[video_frame_offset:]
        Calls.animate.append({
            "length": length, "offset_in": video_frame_offset,
            "continue": None if continue_motion is None else continue_motion.shape[0],
            "continue_max": None if continue_motion is None else float(continue_motion.max()),
            "max_frames": continue_motion_max_frames,
            "pose": None if pose_video is None else float(pose_video[0, 0, 0, 0]),
            "clip": clip_vision_output, "face": face_video, "background": background_video, "mask": character_mask,
        })
        trim_image = max(0, ref_motion_latent_length * 4 - 3)
        if character_mask is not None:  # core: a single frame is repeated, a video is seeked and resized to latent size
            character_mask = character_mask.repeat(length, 1, 1) if character_mask.shape[0] == 1 else character_mask[video_frame_offset:][:length]
            character_mask = fake_common_upscale(character_mask.unsqueeze(1), width // LATENT_DOWN, height // LATENT_DOWN, "nearest-exact", "center").squeeze(1)
        mask = core_concat_mask(latent_length, height // LATENT_DOWN, width // LATENT_DOWN, ref_motion_latent_length, trim_image, character_mask)
        positive = [[c[0], {**c[1], "concat_mask": mask}] for c in positive]
        negative = [[c[0], {**c[1], "concat_mask": mask}] for c in negative]
        latent = {"samples": torch.zeros(batch_size, 16, latent_length + trim_latent, height // LATENT_DOWN, width // LATENT_DOWN)}
        return FakeNodeOutput(positive, negative, latent, trim_latent, trim_image, video_frame_offset + length)


def core_concat_mask(latent_length, lat_h, lat_w, ref_motion_latent_length, ref_images_num, character_mask):
    """comfy_extras/nodes_wan.py WanAnimateToVideo mask construction, verbatim
    apart from the resize (the test masks are already latent sized).
    0 == known; concat_cond inverts it for the model."""
    mask = torch.zeros((1, 4, 1, lat_h, lat_w))
    mask_refmotion = torch.ones((1, 1, latent_length * 4, lat_h, lat_w))
    if ref_motion_latent_length > 0:
        mask_refmotion[:, :, :ref_motion_latent_length * 4] = 0.0
    if character_mask is not None:
        character_mask = character_mask.unsqueeze(1).movedim(0, 1).unsqueeze(1)  # (1, 1, T, h, w)
        if character_mask.shape[2] > ref_images_num:
            mask_refmotion[:, :, ref_images_num:character_mask.shape[2]] = character_mask[:, :, ref_images_num:]
    mask_refmotion = mask_refmotion.view(1, mask_refmotion.shape[2] // 4, 4, lat_h, lat_w).transpose(1, 2)
    return torch.cat((mask, mask_refmotion), dim=2)


def reference_concat_mask(latent_length, lat_h, lat_w, seed_frames, character_mask):
    """Wan2.2 wan/animate.py get_i2v_mask + the ref latent, in core's
    polarity (0 == known). mask_pixel_values there is 1 - character mask."""
    if character_mask is None:
        msk = torch.ones(1, (latent_length - 1) * 4 + 1, lat_h, lat_w)
    else:
        msk = character_mask.unsqueeze(0).clone()
    msk[:, :seed_frames] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)
    return torch.cat((torch.zeros((1, 4, 1, lat_h, lat_w)), msk), dim=2)


class FakeWanAnimate2ToVideo:
    """Mirrors comfy_extras/nodes_wan.py WanAnimate2ToVideo.execute: the
    overlap is the class constant, an offset past the pose video is an error."""

    CONTINUE_MOTION_FRAMES = CONTINUE_MOTION_FRAMES
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT", "INT", "INT")
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"length": ("INT", {"default": 81}), "batch_size": ("INT", {"default": 1})},
                "optional": {"continue_motion": ("IMAGE", {})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, positive, negative, vae, width, height, length, batch_size, video_frame_offset,
                           reference_image=None, pose_video=None, clip_vision_output=None, positive_pose=None,
                           clip_vision_output_pose=None, continue_motion=None, pose_strength=1.0,
                           pose_start_percent=0.0, pose_end_percent=1.0, reference_image_strength=1.0):
        latent_length = ((length - 1) // 4) + 1
        trim_latent = 1
        ref_motion_latent_length = 0
        if continue_motion is not None:
            continue_motion = continue_motion[-cls.CONTINUE_MOTION_FRAMES:]
            video_frame_offset = max(0, video_frame_offset - continue_motion.shape[0])
            ref_motion_latent_length += ((continue_motion.shape[0] - 1) // 4) + 1
        if pose_video is not None:
            if pose_video.shape[0] <= video_frame_offset:
                raise ValueError("pose_video has {} frames but video_frame_offset is {}".format(pose_video.shape[0], video_frame_offset))
        Calls.animate.append({"length": length, "offset_in": video_frame_offset, "continue": None if continue_motion is None else continue_motion.shape[0],
                              "pose_strength": pose_strength, "positive_pose": positive_pose, "clip_pose": clip_vision_output_pose})
        latent = {"samples": torch.zeros(batch_size, 16, latent_length + trim_latent, height // LATENT_DOWN, width // LATENT_DOWN)}
        return FakeNodeOutput(positive, negative, latent, trim_latent, max(0, ref_motion_latent_length * 4 - 3), video_frame_offset + length)


class FakeWanSCAILToVideo:
    """Mirrors comfy_extras/nodes_scail.py WanSCAILToVideo.execute where the loop depends on it:
    4 outputs; the last previous_frame_count of previous_frames are kept, the offset moves back
    by as many; the pose video and its mask are seeked by that offset (dropped once it runs past
    them) and cut jointly to the shorter one on the 4k+1 grid, capped at length; the kept frames
    are VAE-encoded into the first latent frames, which a noise_mask marks known; the returned
    offset is the moved-back offset + length. The pose video goes into the conditioning as
    pose_video_latent, VAE-encoded (without core's half-resolution resize) and scaled by
    pose_strength."""

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "INT")
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"width": ("INT", {"default": 512}), "height": ("INT", {"default": 896}), "length": ("INT", {"default": 81}),
                             "batch_size": ("INT", {"default": 1}), "pose_strength": ("FLOAT", {"default": 1.0}),
                             "pose_start": ("FLOAT", {"default": 0.0}), "pose_end": ("FLOAT", {"default": 1.0}),
                             "video_frame_offset": ("INT", {"default": 0}), "previous_frame_count": ("INT", {"default": 5})},
                "optional": {"pose_video": ("IMAGE", {}), "pose_video_mask": ("IMAGE", {}), "replacement_mode": ("BOOLEAN", {"default": False}),
                             "reference_image": ("IMAGE", {}), "reference_image_mask": ("IMAGE", {}),
                             "clip_vision_output": ("CLIP_VISION_OUTPUT", {}), "previous_frames": ("IMAGE", {})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, positive, negative, vae, width, height, length, batch_size, pose_strength, pose_start, pose_end,
                           video_frame_offset, previous_frame_count, replacement_mode=False, reference_image=None,
                           clip_vision_output=None, pose_video=None, pose_video_mask=None, reference_image_mask=None,
                           previous_frames=None):
        latent = torch.zeros(batch_size, 16, ((length - 1) // 4) + 1, height // LATENT_DOWN, width // LATENT_DOWN)
        pose_in = None if pose_video is None else pose_video.shape[0]
        previous = None
        if previous_frames is not None and previous_frames.shape[0] > 0:
            previous = previous_frames[-previous_frame_count:]
            video_frame_offset = max(0, video_frame_offset - previous.shape[0])
        if pose_video is not None:
            pose_video = None if pose_video.shape[0] <= video_frame_offset else pose_video[video_frame_offset:]
        if pose_video_mask is not None:
            pose_video_mask = None if pose_video_mask.shape[0] <= video_frame_offset else pose_video_mask[video_frame_offset:]
        kept = [v.shape[0] for v in (pose_video, pose_video_mask) if v is not None]
        if kept:
            kept = ((min(min(kept), length) - 1) // 4) * 4 + 1
            pose_video = None if pose_video is None else pose_video[:kept]
            pose_video_mask = None if pose_video_mask is None else pose_video_mask[:kept]
        Calls.animate.append({
            "length": length, "offset_in": video_frame_offset, "width": width, "height": height,
            "previous": None if previous is None else previous.shape[0],
            "previous_first": None if previous is None else float(previous[0, 0, 0, 0]),
            "pose": None if pose_video is None else float(pose_video[0, 0, 0, 0]),
            "pose_frames": None if pose_video is None else pose_video.shape[0],
            "pose_in": pose_in,
            "mask": None if pose_video_mask is None else float(pose_video_mask[0, 0, 0, 0]),
            "mask_frames": None if pose_video_mask is None else pose_video_mask.shape[0],
            "replacement_mode": replacement_mode, "pose_strength": pose_strength, "pose_start": pose_start, "pose_end": pose_end,
            "previous_frame_count": previous_frame_count, "clip": clip_vision_output, "reference_mask": reference_image_mask,
        })
        values = {"ref_mask_flag": not replacement_mode}
        if pose_video is not None:
            values["pose_video_latent"] = vae.encode(pose_video[:, :, :, :3]) * pose_strength
        positive = [[c[0], {**c[1], **values}] for c in positive]
        negative = [[c[0], {**c[1], **values}] for c in negative]
        out = {"samples": latent}
        if previous is not None:
            encoded = vae.encode(previous[:, :, :, :3])
            frames = min(encoded.shape[2], latent.shape[2])
            latent[:, :, :frames] = encoded[:, :, :frames]
            noise_mask = torch.ones(1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1])
            noise_mask[:, :, :frames] = 0.0
            out["noise_mask"] = noise_mask
        return FakeNodeOutput(positive, negative, out, video_frame_offset + length)


class FakeCLIPVisionEncode:
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"crop": (["center", "none"], {"default": "center"})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, clip_vision, image, crop):
        return FakeNodeOutput(("clip", clip_vision, float(image[0, 0, 0, 0]), crop))


class FakeSamplerCustom:
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"add_noise": ("BOOLEAN", {"default": True}), "cfg": ("FLOAT", {"default": 8.0})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        Calls.sampler.append({"seed": noise_seed, "model": model, "sigmas": sigmas, "positive": positive, "negative": negative})
        return FakeNodeOutput(dict(latent_image), dict(latent_image))


class FakeKSamplerSelect:
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"sampler_name": (["lcm", "euler"], {})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, sampler_name):
        return FakeNodeOutput(("sampler", sampler_name))


class FakeBasicScheduler:
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"scheduler": (["simple"], {}), "steps": ("INT", {"default": 20}), "denoise": ("FLOAT", {"default": 1.0})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, model, scheduler, steps, denoise):
        assert getattr(model, "shift", None) is not None, "sigmas must be built from the shift-patched model"
        if denoise <= 0.0:
            return FakeNodeOutput(torch.FloatTensor([]))
        return FakeNodeOutput(torch.linspace(1.0, 0.0, steps + 1))


class FakeTrimVideoLatent:
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"trim_amount": ("INT", {"default": 0})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, samples, trim_amount):
        out = samples.copy()
        out["samples"] = samples["samples"][:, :, trim_amount:]
        return FakeNodeOutput(out)


# --- V1-shaped fakes: FUNCTION names an instance method, returns a tuple ---

class FakeModelSamplingSD3:
    FUNCTION = "patch"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "shift": ("FLOAT", {"default": 3.0})}}

    def patch(self, model, shift, multiplier=1000, sampling="flow"):
        m = model.clone()
        m.shift = shift
        return (m,)


class FakeVAEDecode:
    FUNCTION = "decode"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"samples": ("LATENT",), "vae": ("VAE",)}}

    def decode(self, vae, samples):
        images = vae.process_output(vae.decode(samples["samples"]))
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)


class FakeProgressBar:
    instances = []

    def __init__(self, total, node_id=None):
        self.total = total
        self.current = 0
        FakeProgressBar.instances.append(self)

    def update(self, value):
        self.current += value


def fake_common_upscale(samples, width, height, upscale_method, crop):
    """comfy/utils.py common_upscale for 4-D input: "center" crops to the target aspect first."""
    if crop == "center":
        old_width, old_height = samples.shape[-1], samples.shape[-2]
        old_aspect, new_aspect = old_width / old_height, width / height
        x = y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        samples = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
    return torch.nn.functional.interpolate(samples, size=(height, width), mode=upscale_method)


@pytest.fixture
def node_module(monkeypatch):
    comfy = types.ModuleType("comfy")
    comfy_mm = types.ModuleType("comfy.model_management")
    comfy_mm.throw_exception_if_processing_interrupted = lambda: None
    comfy_samplers = types.ModuleType("comfy.samplers")
    comfy_samplers.SAMPLER_NAMES = ["euler", "lcm"]
    comfy_samplers.SCHEDULER_NAMES = ["normal", "simple", "beta"]
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = FakeProgressBar
    comfy_utils.common_upscale = fake_common_upscale
    comfy.model_management = comfy_mm
    comfy.samplers = comfy_samplers
    comfy.utils = comfy_utils

    core_nodes = types.ModuleType("nodes")
    core_nodes.MAX_RESOLUTION = 16384
    core_nodes.NODE_CLASS_MAPPINGS = {
        "WanAnimateToVideo": FakeWanAnimateToVideo,
        "WanAnimate2ToVideo": FakeWanAnimate2ToVideo,
        "WanSCAILToVideo": FakeWanSCAILToVideo,
        "SamplerCustom": FakeSamplerCustom,
        "CLIPVisionEncode": FakeCLIPVisionEncode,
        "KSamplerSelect": FakeKSamplerSelect,
        "BasicScheduler": FakeBasicScheduler,
        "TrimVideoLatent": FakeTrimVideoLatent,
        "ModelSamplingSD3": FakeModelSamplingSD3,
        "VAEDecode": FakeVAEDecode,
    }

    for name, module in {
        "comfy": comfy,
        "comfy.model_management": comfy_mm,
        "comfy.samplers": comfy_samplers,
        "comfy.utils": comfy_utils,
        "nodes": core_nodes,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    # load the package under a hyphen-free alias so relative imports resolve
    package = types.ModuleType("walong")
    package.__path__ = [PACKAGE_ROOT]
    monkeypatch.setitem(sys.modules, "walong", package)
    for stale in [name for name in sys.modules if name.startswith("walong.")]:
        monkeypatch.delitem(sys.modules, stale, raising=False)
    importlib.import_module("walong.nodes.sampler")

    Calls.animate = []
    Calls.sampler = []
    FakeProgressBar.instances = []
    return sampler


NODE_DEFAULTS = {
    ANIMATE1: dict(continue_motion_max_frames=5),
    ANIMATE2: dict(reference_image_strength=1.0, pose_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0, attn_log_scale=0.0),
    SCAIL2: dict(clip_vision="cv", replacement_mode=False, pose_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0,
                 previous_frame_count=5),
}


def reference_mask(replacement_mode=False, frames=1):
    """A colored reference mask of run()'s 64 x 32 size: a blue person in the middle, on white
    (animation mode) or black (replacement mode)."""
    background = 0.0 if replacement_mode else 1.0
    mask = torch.full((frames, 64, 32, 3), background)
    mask[:, 16:48, 8:24] = torch.tensor([0.0, 0.0, 1.0])
    return mask


def run(module, pose_frames, node=ANIMATE2, total_frames=0, frames_per_chunk=81, seed=7, seed_mode="increment", **overrides):
    kwargs = dict(
        model=FakeModel(),
        positive=[["pos", {}]],
        negative=[["neg", {}]],
        vae=FakeVAE(),
        reference_image=torch.zeros(1, 64, 32, 3),
        pose_video=torch.arange(pose_frames, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 64, 32, 3).contiguous(),
        width=32,
        height=64,
        frames_per_chunk=frames_per_chunk,
        total_frames=total_frames,
        shift=5.0,
        sampler_name="lcm",
        scheduler="simple",
        steps=6,
        denoise=1.0,
        cfg=1.0,
        seed=seed,
        seed_mode=seed_mode,
    )
    kwargs.update(NODE_DEFAULTS[node])
    if node == SCAIL2:
        # the colored driving mask: the frame index as well, so its seek is visible
        kwargs["pose_video_mask"] = kwargs["pose_video"].clone()
        kwargs["reference_image_mask"] = reference_mask(overrides.get("replacement_mode", False))
    kwargs.update(overrides)
    return getattr(module, node)().generate(**kwargs)
