"""Runs the nodes' chunk loop against fake core nodes.

ComfyUI itself is stubbed (comfy.*, nodes, latent_preview); torch is real so
the tensor plumbing (trim, cat, crop, pad) is exercised for real. Skipped when
torch is not installed.
"""

import importlib
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ANIMATE1 = "BCVWanAnimateLongVideoSampler"
ANIMATE2 = "BCVWanAnimate2LongVideoSampler"
CONTINUE_MOTION_FRAMES = 1  # WanAnimate2ToVideo's class constant
LATENT_DOWN = 8

# widget order the released 0.1.0 node had: saved workflows store widget
# values positionally, so this must never change
ANIMATE2_REQUIRED_ORDER = [
    "model", "positive", "negative", "vae", "reference_image", "pose_video",
    "width", "height", "frames_per_chunk", "total_frames",
    "shift", "sampler_name", "scheduler", "steps", "denoise", "cfg", "seed", "seed_mode",
    "reference_image_strength", "pose_strength", "pose_start_percent", "pose_end_percent",
]


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
    for stale in ("walong.nodes", "walong.chunk_planner"):
        monkeypatch.delitem(sys.modules, stale, raising=False)
    module = importlib.import_module("walong.nodes")

    Calls.animate = []
    Calls.sampler = []
    FakeProgressBar.instances = []
    return module


NODE_DEFAULTS = {
    ANIMATE1: dict(continue_motion_max_frames=5),
    ANIMATE2: dict(reference_image_strength=1.0, pose_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0, attn_log_scale=0.0),
}


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
    kwargs.update(overrides)
    return getattr(module, node)().generate(**kwargs)


# --- shared behaviour, both nodes ---

@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_input_types_shared_widgets(node_module, node):
    spec = getattr(node_module, node).INPUT_TYPES()
    required = list(spec["required"])
    assert required[:len(ANIMATE2_REQUIRED_ORDER) - 4] == ANIMATE2_REQUIRED_ORDER[:-4]
    assert spec["required"]["seed"][1]["control_after_generate"] is True
    assert list(spec["optional"])[-1] == "sigmas_override"
    assert spec["required"]["frames_per_chunk"][1]["default"] == 81
    assert spec["required"]["total_frames"][1]["default"] == 81
    assert spec["required"]["total_frames"][1]["min"] == 0  # 0 still means "pose video length"
    assert node_module._combo_default(["a", "b"], "lcm") == "a"


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_fixed_seed_mode(node_module, node):
    run(node_module, pose_frames=200, node=node, seed=3, seed_mode="fixed")
    assert [s["seed"] for s in Calls.sampler] == [3, 3, 3]


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_sigmas_override_skips_scheduler(node_module, caplog, node):
    caplog.set_level("INFO")
    override = torch.linspace(1.0, 0.0, 4)
    run(node_module, pose_frames=81, node=node, sigmas_override=override)
    assert Calls.sampler[0]["sigmas"] is override
    assert "sigmas_override" in caplog.text


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_sampler_uses_shift_patched_model(node_module, node):
    run(node_module, pose_frames=81, node=node, shift=8.0)
    assert Calls.sampler[0]["model"].shift == 8.0


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_empty_schedule_is_an_error(node_module, node):
    with pytest.raises(ValueError):
        run(node_module, pose_frames=81, node=node, denoise=0.0)


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_old_core_is_rejected(node_module, monkeypatch, node):
    animate_node = getattr(node_module, node).ANIMATE_NODE

    class OldAnimate(sys.modules["nodes"].NODE_CLASS_MAPPINGS[animate_node]):
        RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")

    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, animate_node, OldAnimate)
    with pytest.raises(RuntimeError, match="Update ComfyUI.*" + animate_node):
        run(node_module, pose_frames=81, node=node)


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_log_prefix_names_the_node(node_module, caplog, node):
    caplog.set_level("INFO")
    run(node_module, pose_frames=81, node=node)
    assert "[{}] chunk plan:".format(node) in caplog.text


# --- Wan Animate 2 node ---

def test_animate2_input_types(node_module):
    spec = node_module.BCVWanAnimate2LongVideoSampler.INPUT_TYPES()
    assert list(spec["required"]) == ANIMATE2_REQUIRED_ORDER + ["attn_log_scale"]
    assert list(spec["optional"]) == ["positive_pose", "clip_vision_output", "clip_vision_output_pose", "clip_vision", "sigmas_override"]
    assert spec["required"]["frames_per_chunk"][1]["default"] == 81
    assert spec["required"]["shift"][1]["default"] == 5.0
    assert spec["required"]["sampler_name"][1]["default"] == "euler"
    # 10 steps as the official distilled config; wan_beta because it won the user's A/B against beta and simple is one click away
    assert spec["required"]["scheduler"][1]["default"] == "wan_beta"
    assert spec["required"]["scheduler"][0][-1] == "wan_beta"
    assert spec["required"]["steps"][1]["default"] == 10
    assert spec["required"]["attn_log_scale"][1]["default"] == -1.3


def test_animate2_exact_length_from_pose(node_module):
    images, count, plan = run(node_module, pose_frames=360)
    assert images.shape[0] == 360
    assert count == 360
    assert plan == "81 + 81 + 81 + 81 + 41 -> 361 produced -> 360 frames (pose 360, overlap 1)"
    assert [c["length"] for c in Calls.animate] == [81, 81, 81, 81, 41]
    # first chunk has no anchor, every later chunk is seeded with the previous batch (core keeps 1 frame)
    assert Calls.animate[0]["continue"] is None
    assert all(c["continue"] == 1 for c in Calls.animate[1:])
    # offset chain: returned offset feeds the next call, adjusted by the overlap inside the node
    assert [c["offset_in"] for c in Calls.animate] == [0, 80, 160, 240, 320]
    assert [s["seed"] for s in Calls.sampler] == [7, 8, 9, 10, 11]
    assert FakeProgressBar.instances[0].total == 5
    assert FakeProgressBar.instances[0].current == 5


def test_animate2_single_chunk_when_it_fits(node_module):
    images, count, plan = run(node_module, pose_frames=81)
    assert count == 81 and images.shape[0] == 81
    assert len(Calls.animate) == 1


def test_animate2_total_frames_widget_crops_pose(node_module):
    images, count, _ = run(node_module, pose_frames=500, total_frames=100)
    assert count == 100
    assert [c["length"] for c in Calls.animate] == [81, 21]


def test_animate2_total_beyond_pose_holds_last_frame(node_module, caplog):
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=100, total_frames=250, frames_per_chunk=49)
    assert count == 250
    assert "held" in caplog.text
    assert plan.endswith("(pose 100, overlap 1)")


def test_animate2_pass_through_inputs_reach_core(node_module):
    run(node_module, pose_frames=81, pose_strength=0.5, positive_pose=[["motion", {}]])
    assert Calls.animate[0]["pose_strength"] == 0.5
    assert Calls.animate[0]["positive_pose"] == [["motion", {}]]


def test_animate2_pose_clip_reencoded_per_chunk_when_clip_vision_connected(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, clip_vision_output_pose="static", clip_vision="cv")
    # the first frame of each chunk's pose window: 0, then 80k (offset moved back by the 1 seed frame)
    assert [c["clip_pose"] for c in Calls.animate] == [("clip", "cv", 0.0, "none"), ("clip", "cv", 80.0, "none"), ("clip", "cv", 160.0, "none")]
    assert "re-encoded per chunk" in caplog.text

    Calls.animate, Calls.sampler = [], []
    run(node_module, pose_frames=200, clip_vision_output_pose="static")
    assert [c["clip_pose"] for c in Calls.animate] == ["static", "static", "static"]


def test_animate2_attn_log_scale_installs_the_attention_override(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=81)
    assert "optimized_attention_override" not in Calls.sampler[0]["model"].model_options["transformer_options"]

    Calls.animate, Calls.sampler = [], []
    run(node_module, pose_frames=81, attn_log_scale=-1.3, shift=5.0)
    model = Calls.sampler[0]["model"]
    assert callable(model.model_options["transformer_options"]["optimized_attention_override"])
    assert model.shift == 5.0  # the clone keeps ModelSamplingSD3's patch
    assert "attn_log_scale -1.30" in caplog.text


def test_seed_frame_attention_bias_targets_generation_attention_only(node_module, monkeypatch):
    seen = []
    attention = types.ModuleType("comfy.ldm.modules.attention")

    def attention_pytorch(q, k, v, heads, mask=None, **kwargs):
        seen.append(mask.clone())
        return "biased"

    attention.attention_pytorch = attention_pytorch
    ldm = types.ModuleType("comfy.ldm"); modules = types.ModuleType("comfy.ldm.modules")
    for name, module in {"comfy.ldm": ldm, "comfy.ldm.modules": modules, "comfy.ldm.modules.attention": attention}.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["comfy"].ldm = ldm; ldm.modules = modules; modules.attention = attention

    override = node_module._seed_frame_attention_bias(-1.3)
    frames, gh, gw = 4, 3, 2
    hw, tokens = gh * gw, 4 * gh * gw
    options = {"block_type": "double", "grid_sizes": (frames, gh, gw)}
    func = lambda q, k, v, **kw: "plain"
    q = torch.zeros(1, hw, 8, dtype=torch.float16)
    # per-frame generation call: q = one frame, k = all gen tokens + that frame's pose tokens
    assert override(func, q, torch.zeros(1, tokens + hw, 8), torch.zeros(1, tokens + hw, 8), heads=2, transformer_options=options) == "biased"
    mask = seen[-1]
    assert mask.shape == (1, 1, 1, tokens + hw) and mask.dtype == torch.float16
    assert torch.allclose(mask[..., hw:2 * hw].float(), torch.full((hw,), -1.3), atol=1e-3)
    assert (mask[..., :hw] == 0).all() and (mask[..., 2 * hw:] == 0).all()
    # frame 0's call has no pose tail
    assert override(func, q, torch.zeros(1, tokens, 8), torch.zeros(1, tokens, 8), heads=2, transformer_options=options) == "biased"
    # whole-clip self-attention when the pose branch is windowed out
    assert override(func, torch.zeros(1, tokens, 8), torch.zeros(1, tokens, 8), torch.zeros(1, tokens, 8), heads=2, transformer_options=options) == "biased"
    # cross-attention (769 text + image tokens), the pose branch ((f-1)*hw), and blocks without grid info pass through
    assert override(func, torch.zeros(1, tokens, 8), torch.zeros(1, 769, 8), torch.zeros(1, 769, 8), heads=2, transformer_options=options) == "plain"
    assert override(func, torch.zeros(1, 3 * hw, 8), torch.zeros(1, 3 * hw, 8), torch.zeros(1, 3 * hw, 8), heads=2, transformer_options=options) == "plain"
    assert override(func, q, torch.zeros(1, tokens, 8), torch.zeros(1, tokens, 8), heads=2, transformer_options={}) == "plain"
    assert len(seen) == 3


def test_animate2_pose_percent_validation(node_module):
    with pytest.raises(ValueError):
        run(node_module, pose_frames=81, pose_start_percent=0.8, pose_end_percent=0.2)
    assert Calls.animate == []


def test_animate2_overlap_follows_the_core_constant(node_module, monkeypatch):
    # a core that keeps 5 motion frames: the loop must still land exactly
    monkeypatch.setattr(FakeWanAnimate2ToVideo, "CONTINUE_MOTION_FRAMES", 5)
    images, count, plan = run(node_module, pose_frames=200, frames_per_chunk=49)
    assert count == 200
    assert plan.endswith("overlap 5)")
    assert all(c["continue"] == 5 for c in Calls.animate[1:])


def test_animate2_chunk_and_step_logging(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, frames_per_chunk=81, seed=7)
    # 81 + 81 + 41 -> 81 + 80 + 40 = 201 produced, cropped to 200
    assert "chunk 1/3 (frames 1-81/200): length 81, pose offset 0, seed 7" in caplog.text
    assert "chunk 2/3 (frames 82-161/200): length 81, pose offset 81, seed 8" in caplog.text
    assert "chunk 3/3 (frames 162-200/200): length 41, pose offset 161, seed 9" in caplog.text
    assert "chunk 2/3 (frames 82-161/200) done: 80 new frames, 161/200 total" in caplog.text


# --- Wan Animate (1) node ---

def test_animate1_input_types(node_module):
    spec = node_module.BCVWanAnimateLongVideoSampler.INPUT_TYPES()
    assert list(spec["required"]) == ANIMATE2_REQUIRED_ORDER[:-4] + ["continue_motion_max_frames"]
    assert list(spec["optional"]) == ["clip_vision_output", "face_video", "background_video", "character_mask", "sigmas_override"]
    # core node + official template defaults
    assert spec["required"]["frames_per_chunk"][1]["default"] == 81
    assert spec["required"]["continue_motion_max_frames"][1] == {"default": 5, "min": 1, "max": 16384, "step": 4, "tooltip": spec["required"]["continue_motion_max_frames"][1]["tooltip"]}
    assert spec["required"]["shift"][1]["default"] == 8.0
    assert spec["required"]["sampler_name"][1]["default"] == "euler"
    assert spec["required"]["scheduler"][1]["default"] == "wan_beta"
    assert spec["required"]["steps"][1]["default"] == 6
    assert spec["optional"]["character_mask"][0] == "MASK"


def test_animate1_exact_length_from_pose(node_module):
    images, count, plan = run(node_module, pose_frames=360, node=ANIMATE1, frames_per_chunk=77)
    assert images.shape[0] == 360
    assert count == 360
    assert plan == "77 + 77 + 77 + 77 + 73 -> 361 produced -> 360 frames (pose 360, overlap 5)"
    assert [c["length"] for c in Calls.animate] == [77, 77, 77, 77, 73]
    assert Calls.animate[0]["continue"] is None
    assert all(c["continue"] == 5 for c in Calls.animate[1:])
    assert all(c["max_frames"] == 5 for c in Calls.animate)
    # offset chain: each chunk starts 5 frames before the previous one ended
    assert [c["offset_in"] for c in Calls.animate] == [0, 72, 144, 216, 288]
    # and reads the pose from exactly there
    assert [c["pose"] for c in Calls.animate] == [0.0, 72.0, 144.0, 216.0, 288.0]
    assert [s["seed"] for s in Calls.sampler] == [7, 8, 9, 10, 11]
    assert FakeProgressBar.instances[0].total == 5
    assert FakeProgressBar.instances[0].current == 5


def test_animate1_single_chunk_when_it_fits(node_module):
    images, count, plan = run(node_module, pose_frames=77, node=ANIMATE1, frames_per_chunk=77)
    assert count == 77 and images.shape[0] == 77
    assert len(Calls.animate) == 1


def test_animate1_larger_overlap_widget(node_module):
    images, count, plan = run(node_module, pose_frames=200, node=ANIMATE1, frames_per_chunk=77, continue_motion_max_frames=9)
    assert count == 200
    assert plan == "77 + 77 + 65 -> 201 produced -> 200 frames (pose 200, overlap 9)"
    assert [c["offset_in"] for c in Calls.animate] == [0, 68, 136]
    assert all(c["continue"] == 9 for c in Calls.animate[1:])


def test_animate1_overlap_above_half_the_chunk_keeps_the_full_seed(node_module, caplog):
    # a middle chunk adds 17 - 13 = 4 new frames; the next chunk must still be seeded with 13,
    # or core moves the offset back by 4, trims only 1 and the pose falls behind the output
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=60, node=ANIMATE1, frames_per_chunk=17, continue_motion_max_frames=13)
    assert count == 60
    assert all(c["continue"] == 13 for c in Calls.animate[1:])
    assert [c["offset_in"] for c in Calls.animate] == [4 * k for k in range(len(Calls.animate))]
    assert "planner assumed" not in caplog.text


def test_animate1_snaps_continue_motion_max_frames_to_grid(node_module, caplog):
    caplog.set_level("INFO")
    images, count, plan = run(node_module, pose_frames=200, node=ANIMATE1, frames_per_chunk=77, continue_motion_max_frames=7)
    assert count == 200
    assert plan.endswith("overlap 5)")
    assert all(c["max_frames"] == 5 for c in Calls.animate)
    assert all(c["continue"] == 5 for c in Calls.animate[1:])
    assert "continue_motion_max_frames 7 is not on the 4k+1 grid; using 5" in caplog.text


def test_animate1_overlap_must_be_smaller_than_chunk(node_module):
    with pytest.raises(ValueError, match="must exceed the overlap"):
        run(node_module, pose_frames=200, node=ANIMATE1, frames_per_chunk=77, continue_motion_max_frames=77)
    assert Calls.animate == []


def test_animate1_optional_videos_reach_core(node_module):
    face = torch.zeros(81, 512, 512, 3)
    background = torch.zeros(81, 64, 32, 3)
    mask = torch.zeros(1, 64, 32)
    run(node_module, pose_frames=81, node=ANIMATE1, frames_per_chunk=77,
        clip_vision_output="clip", face_video=face, background_video=background, character_mask=mask)
    for call in Calls.animate:
        assert call["clip"] == "clip"
        assert call["face"] is face
        assert call["background"] is background
        assert call["mask"] is mask


def test_animate1_total_beyond_pose_keeps_pose_on_every_chunk(node_module, caplog):
    # WanAnimateToVideo silently drops a pose video the offset has run past;
    # the up-front pad means every chunk still sees one
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=100, node=ANIMATE1, total_frames=250, frames_per_chunk=49)
    assert count == 250
    assert "held" in caplog.text
    assert plan.endswith("(pose 100, overlap 5)")
    assert all(c["pose"] is not None for c in Calls.animate)
    assert Calls.animate[-1]["pose"] == 99.0


def _videos(frames):
    """face / background / multi-frame mask whose content is the frame index, so a held frame is visible."""
    index = torch.arange(frames, dtype=torch.float32)
    return dict(face_video=index.view(-1, 1, 1, 1).expand(-1, 8, 8, 3).contiguous(),
                background_video=index.view(-1, 1, 1, 1).expand(-1, 64, 32, 3).contiguous(),
                character_mask=index.view(-1, 1, 1).expand(-1, 64, 32).contiguous())


@pytest.mark.parametrize("frames, produced", [(150, 153), (152, 153)])
def test_animate1_overshooting_last_chunk_holds_every_video(node_module, caplog, frames, produced):
    # off the 4k+1 grid the last chunk is snapped up and runs past the source: pose, face and
    # background must reach that far, or core zero-pads the face and greys the background. The
    # mask is not held: past its end core leaves its rows unknown
    caplog.set_level("INFO")
    images, count, plan = run(node_module, pose_frames=frames, node=ANIMATE1, frames_per_chunk=81, **_videos(frames))
    assert count == frames
    assert plan.startswith("81 + 77 -> {} produced".format(produced))
    last = Calls.animate[-1]
    assert last["offset_in"] + last["length"] == produced
    for key in ("face", "background"):
        video = last[key]
        assert video.shape[0] == produced
        assert torch.equal(video[:frames], _videos(frames)[{"face": "face_video", "background": "background_video"}[key]])
        assert (video[frames:] == frames - 1).all()  # the last frame, held
    assert last["mask"].shape[0] == frames
    padded = [line for line in caplog.text.splitlines() if "held" in line]
    assert len(padded) == 1 and "character_mask" not in padded[0]
    for name in ("pose_video", "face_video", "background_video"):
        assert "{} +{}".format(name, produced - frames) in padded[0]


def test_animate1_total_beyond_inputs_holds_every_video_but_the_mask(node_module, caplog):
    # total_frames past the input length: once the offset passes a video's end core drops it
    # (face driving and, in replacement mode, the scene), so they are held like the pose; the
    # mask is not, past its end the character may be anywhere
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=100, node=ANIMATE1, total_frames=250, frames_per_chunk=49, **_videos(100))
    assert count == 250
    for call in Calls.animate:
        for key in ("face", "background"):
            assert call[key].shape[0] > call["offset_in"] + call["length"] - 1
            assert (call[key][100:] == 99).all()
        assert call["mask"].shape[0] == 100
    assert "total_frames (250) exceeds" in caplog.text


def test_animate1_single_frame_mask_is_not_padded(node_module):
    mask = torch.ones(1, 64, 32)
    run(node_module, pose_frames=150, node=ANIMATE1, character_mask=mask)
    assert all(call["mask"] is mask for call in Calls.animate)


def test_animate1_grid_inputs_reach_core_unchanged(node_module, caplog):
    # 4k+1 past the seed: the plan ends exactly on the last frame, nothing is padded or logged
    caplog.set_level("INFO")
    videos = _videos(161)
    images, count, plan = run(node_module, pose_frames=161, node=ANIMATE1, frames_per_chunk=81, **videos)
    assert plan.startswith("81 + 81 + 9 -> 161 produced")
    for call in Calls.animate:
        assert call["face"] is videos["face_video"]
        assert call["background"] is videos["background_video"]
        assert call["mask"] is videos["character_mask"]
    assert "held" not in caplog.text


def test_animate2_overshooting_last_chunk_is_logged(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=150)
    padded = [line for line in caplog.text.splitlines() if "held" in line]
    assert len(padded) == 1 and "pose_video +3" in padded[0] and "153" in padded[0]


@pytest.mark.parametrize("seed_frames", [0, 1, 5])
def test_animate1_mask_repair_matches_reference_implementation(node_module, seed_frames):
    torch.manual_seed(0)
    latent_length, lat_h, lat_w = 21, 6, 4
    frames = (latent_length - 1) * 4 + 1
    character_mask = (torch.rand(frames, lat_h, lat_w) > 0.5).float()  # changes every frame, so a 3-row shift is visible
    seed_latents = 0 if seed_frames == 0 else ((seed_frames - 1) // 4) + 1
    core = core_concat_mask(latent_length, lat_h, lat_w, seed_latents, seed_frames, character_mask)
    reference = reference_concat_mask(latent_length, lat_h, lat_w, seed_frames, character_mask)
    if seed_frames > 0:
        assert not torch.equal(core, reference)  # the core bug this repairs
        # the last seed latent has 3 of its 4 rows overwritten by the character mask
        assert core[:, 0, seed_latents].sum() == 0 and core[:, 1:, seed_latents].sum() > 0
        assert reference[:, :, 1:1 + seed_latents].sum() == 0
    rows = node_module._replacement_mask_rows(character_mask, 0, frames, seed_frames, lat_h, lat_w)
    assert rows.shape == (frames + 3, lat_h, lat_w)
    cond = [["c", {"concat_mask": core}], ["c2", {"concat_mask": core}]]
    node_module._fix_replacement_mask(cond, rows, set())
    assert torch.equal(core, reference)


def test_animate1_mask_rows_center_crop_like_core(node_module):
    # a square mask onto a 1:2 latent grid: core crops the sides off before resizing (as it does
    # the pose and background videos), so a stripe in the cropped-off column must not survive
    mask = torch.zeros(5, 8, 8)
    mask[:, :, 2] = 1.0
    rows = node_module._replacement_mask_rows(mask, 0, 5, 0, 4, 2)
    expected = torch.nn.functional.interpolate(mask[:, None, :, 2:6], size=(4, 2), mode="nearest-exact")[:, 0]
    assert torch.equal(rows[3:], expected)


def test_animate1_mask_beyond_offset_is_left_to_core(node_module):
    assert node_module._replacement_mask_rows(torch.zeros(10, 4, 4), 10, 9, 1, 4, 4) is None
    single = node_module._replacement_mask_rows(torch.ones(1, 4, 4), 500, 9, 1, 4, 4)  # one frame is repeated whatever the offset
    assert single.shape == (12, 4, 4) and single[:4].sum() == 0 and single[4:].sum() == 8 * 16


def test_animate1_mask_repair_reaches_sampler_only_with_character_mask(node_module):
    torch.manual_seed(0)
    frames = 200
    # longer than the video so every chunk, including the short last one, is fully covered
    character_mask = (torch.rand(frames + 80, 64 // LATENT_DOWN, 32 // LATENT_DOWN) > 0.5).float()
    run(node_module, pose_frames=frames, node=ANIMATE1, frames_per_chunk=77, character_mask=character_mask, background_video=torch.zeros(frames + 80, 64, 32, 3))
    assert [c["length"] for c in Calls.animate] == [77, 77, 57]
    for call, sampled in zip(Calls.animate, Calls.sampler):
        mask = sampled["positive"][0][1]["concat_mask"]
        offset, length = call["offset_in"], call["length"]
        seed = 0 if call["continue"] is None else call["continue"]
        assert torch.equal(mask, reference_concat_mask((length - 1) // 4 + 1, 8, 4, seed, character_mask[offset:offset + length]))
        assert sampled["negative"][0][1]["concat_mask"] is mask  # shared tensor, repaired once

    Calls.animate, Calls.sampler = [], []
    run(node_module, pose_frames=frames, node=ANIMATE1, frames_per_chunk=77)
    for call, sampled in zip(Calls.animate, Calls.sampler):
        seed = 0 if call["continue"] is None else call["continue"]
        seed_latents = 0 if seed == 0 else ((seed - 1) // 4) + 1
        assert torch.equal(sampled["positive"][0][1]["concat_mask"], core_concat_mask((call["length"] - 1) // 4 + 1, 8, 4, seed_latents, seed, None))

    # the mask marks where the character goes in each background frame: a mask video of
    # another length than the background is an error, before anything is sampled
    Calls.animate, Calls.sampler = [], []
    with pytest.raises(ValueError, match="character_mask has 100 frames but background_video has 200"):
        run(node_module, pose_frames=frames, node=ANIMATE1, frames_per_chunk=77, character_mask=character_mask[:100],
            background_video=torch.zeros(frames, 64, 32, 3))
    assert Calls.animate == []  # it stops before anything is sampled


def test_animate1_chunk_logging(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, node=ANIMATE1, frames_per_chunk=77, seed=7)
    # 77 + 77 + 57 -> 77 + 72 + 52 = 201 produced, cropped to 200
    assert "chunk 1/3 (frames 1-77/200): length 77, pose offset 0, seed 7" in caplog.text
    assert "chunk 2/3 (frames 78-149/200): length 77, pose offset 77, seed 8" in caplog.text
    assert "chunk 3/3 (frames 150-200/200): length 57, pose offset 149, seed 9" in caplog.text
    assert "chunk 2/3 (frames 78-149/200) done: 72 new frames, 149/200 total" in caplog.text


def test_wan_beta_sigmas_match_diffusers(node_module):
    pytest.importorskip("scipy")
    # diffusers FlowMatchEulerDiscreteScheduler(shift, use_beta_sigmas=True).set_timesteps(steps), recorded from diffusers 0.40
    expected = {
        (4, 5.0): [1.0, 0.7313, 0.2931, 0.0244, 0.0],
        (6, 5.0): [1.0, 0.8801, 0.6462, 0.3783, 0.1443, 0.0244, 0.0],
        (4, 8.0): [1.0, 0.7412, 0.3190, 0.0602, 0.0],
        (4, 3.0): [1.0, 0.7270, 0.2819, 0.0089, 0.0],
    }
    for (steps, shift), sigmas in expected.items():
        got = node_module.wan_beta_sigmas(steps, shift).tolist()
        assert [round(v, 4) for v in got] == sigmas, (steps, shift, got)
    # denoise < 1 keeps the tail of a longer schedule, like BasicScheduler
    assert node_module.wan_beta_sigmas(2, 5.0, denoise=0.5).tolist() == node_module.wan_beta_sigmas(4, 5.0).tolist()[-3:]


def test_wan_beta_is_used_without_basic_scheduler(node_module, caplog):
    pytest.importorskip("scipy")
    caplog.set_level("INFO")
    run(node_module, pose_frames=81, scheduler="wan_beta", steps=4)
    sigmas = Calls.sampler[0]["sigmas"].tolist()
    assert [round(v, 4) for v in sigmas] == [1.0, 0.7313, 0.2931, 0.0244, 0.0]
    assert "sigmas (wan_beta): 1.0000, 0.7313, 0.2931, 0.0244, 0.0000" in caplog.text


def test_step_logger_labels_the_live_tqdm_bar(node_module):
    tqdm = pytest.importorskip("tqdm")
    import io

    class Inner:
        def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            # k-diffusion creates the bar inside the loop and drives the callback from it
            with tqdm.tqdm(total=len(sigmas) - 1, file=io.StringIO()) as bar:
                for i in range(len(sigmas) - 1):
                    callback(i, "x0", "x", len(sigmas) - 1)
                    bar.update(1)
                return bar.desc

    logger = node_module._StepLogger(Inner(), "[Node]")
    logger.label = "chunk 2/3 (frames 82-161/200)"
    assert logger.sample("wrap", [3, 2, 1, 0], {}, None, "noise") == "chunk 2/3 (frames 82-161/200): "


def test_step_logger_wraps_callback_and_delegates(node_module, caplog):
    caplog.set_level("INFO")
    seen = []

    class Inner:
        extra_options = {"eta": 1.0}

        def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            for i in range(len(sigmas) - 1):
                callback(i, "x0", "x", len(sigmas) - 1)
            return "samples"

    logger = node_module._StepLogger(Inner(), "[Node]")
    logger.label = "chunk 2/3 (frames 82-161/200)"
    assert logger.extra_options == {"eta": 1.0}
    result = logger.sample("wrap", [3, 2, 1, 0], {}, lambda *a: seen.append(a), "noise")
    assert result == "samples"
    assert seen == [(0, "x0", "x", 3), (1, "x0", "x", 3), (2, "x0", "x", 3)]
    assert "[Node] chunk 2/3 (frames 82-161/200) step 1/3" in caplog.text
    assert "[Node] chunk 2/3 (frames 82-161/200) step 3/3" in caplog.text
