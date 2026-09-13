"""Runs the node's chunk loop against fake core nodes.

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

CONTINUE_MOTION_FRAMES = 1
LATENT_DOWN = 8


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
        return torch.zeros(1, frames, h, w, 3)


class FakeModel:
    def __init__(self):
        self.model_options = {"transformer_options": {}}

    def clone(self):
        return FakeModel()


class Calls:
    animate = []
    sampler = []


class FakeNodeOutput:
    def __init__(self, *args):
        self.args = args


# --- V3-shaped fakes: FUNCTION resolves to a classmethod, return has .args ---

class FakeWanAnimate2ToVideo:
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
        # mirrors comfy_extras/nodes_wan.py WanAnimate2ToVideo.execute
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
        Calls.animate.append({"length": length, "offset_in": video_frame_offset, "continue": None if continue_motion is None else continue_motion.shape[0]})
        latent = {"samples": torch.zeros(batch_size, 16, latent_length + trim_latent, height // LATENT_DOWN, width // LATENT_DOWN)}
        return FakeNodeOutput(positive, negative, latent, trim_latent, max(0, ref_motion_latent_length * 4 - 3), video_frame_offset + length)


class FakeSamplerCustom:
    FUNCTION = "EXECUTE_NORMALIZED"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"add_noise": ("BOOLEAN", {"default": True}), "cfg": ("FLOAT", {"default": 8.0})}}

    @classmethod
    def EXECUTE_NORMALIZED(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        Calls.sampler.append({"seed": noise_seed, "model": model, "sigmas": sigmas})
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
        images = vae.decode(samples["samples"])
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


@pytest.fixture
def node_module(monkeypatch):
    comfy = types.ModuleType("comfy")
    comfy_mm = types.ModuleType("comfy.model_management")
    comfy_mm.throw_exception_if_processing_interrupted = lambda: None
    comfy_samplers = types.ModuleType("comfy.samplers")
    comfy_samplers.SAMPLER_NAMES = ["euler", "lcm"]
    comfy_samplers.SCHEDULER_NAMES = ["normal", "simple"]
    comfy_utils = types.ModuleType("comfy.utils")
    comfy_utils.ProgressBar = FakeProgressBar
    comfy.model_management = comfy_mm
    comfy.samplers = comfy_samplers
    comfy.utils = comfy_utils

    core_nodes = types.ModuleType("nodes")
    core_nodes.MAX_RESOLUTION = 16384
    core_nodes.NODE_CLASS_MAPPINGS = {
        "WanAnimate2ToVideo": FakeWanAnimate2ToVideo,
        "SamplerCustom": FakeSamplerCustom,
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
    package = types.ModuleType("wa2long")
    package.__path__ = [PACKAGE_ROOT]
    monkeypatch.setitem(sys.modules, "wa2long", package)
    for stale in ("wa2long.nodes", "wa2long.chunk_planner"):
        monkeypatch.delitem(sys.modules, stale, raising=False)
    module = importlib.import_module("wa2long.nodes")

    Calls.animate = []
    Calls.sampler = []
    FakeProgressBar.instances = []
    return module


def run(module, pose_frames, total_frames=0, frames_per_chunk=81, seed=7, seed_mode="increment", **overrides):
    node = module.WanAnimate2LongVideoSampler()
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
        reference_image_strength=1.0,
        pose_strength=1.0,
        pose_start_percent=0.0,
        pose_end_percent=1.0,
    )
    kwargs.update(overrides)
    return node.generate(**kwargs)


def test_input_types_defaults_and_fallbacks(node_module):
    spec = node_module.WanAnimate2LongVideoSampler.INPUT_TYPES()
    assert spec["required"]["sampler_name"][1]["default"] == "lcm"
    assert spec["required"]["scheduler"][1]["default"] == "simple"
    assert node_module._combo_default(["a", "b"], "lcm") == "a"
    assert spec["required"]["seed"][1]["control_after_generate"] is True
    assert "sigmas_override" in spec["optional"]


def test_exact_length_from_pose(node_module):
    images, count, plan = run(node_module, pose_frames=360)
    assert images.shape[0] == 360
    assert count == 360
    assert plan == "81 + 81 + 81 + 81 + 41 -> 361 produced -> 360 frames (pose 360, overlap 1)"
    assert [c["length"] for c in Calls.animate] == [81, 81, 81, 81, 41]
    # first chunk has no anchor, every later chunk is seeded with the full previous batch
    assert Calls.animate[0]["continue"] is None
    assert all(c["continue"] == 1 for c in Calls.animate[1:])
    # offset chain: returned offset feeds the next call, adjusted by the overlap inside the node
    assert [c["offset_in"] for c in Calls.animate] == [0, 80, 160, 240, 320]
    assert [s["seed"] for s in Calls.sampler] == [7, 8, 9, 10, 11]
    assert FakeProgressBar.instances[0].total == 5
    assert FakeProgressBar.instances[0].current == 5


def test_single_chunk_when_it_fits(node_module):
    images, count, plan = run(node_module, pose_frames=81)
    assert count == 81 and images.shape[0] == 81
    assert len(Calls.animate) == 1


def test_fixed_seed_mode(node_module):
    run(node_module, pose_frames=200, seed=3, seed_mode="fixed")
    assert [s["seed"] for s in Calls.sampler] == [3, 3, 3]


def test_total_frames_widget_crops_pose(node_module):
    images, count, _ = run(node_module, pose_frames=500, total_frames=100)
    assert count == 100
    assert [c["length"] for c in Calls.animate] == [81, 21]


def test_total_beyond_pose_holds_last_frame(node_module, caplog):
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=100, total_frames=250, frames_per_chunk=49)
    assert count == 250
    assert "held" in caplog.text
    assert plan.endswith("(pose 100, overlap 1)")


def test_sigmas_override_skips_scheduler(node_module, caplog):
    caplog.set_level("INFO")
    override = torch.linspace(1.0, 0.0, 4)
    run(node_module, pose_frames=81, sigmas_override=override)
    assert Calls.sampler[0]["sigmas"] is override
    assert "sigmas_override" in caplog.text


def test_sampler_uses_shift_patched_model(node_module):
    run(node_module, pose_frames=81, shift=8.0)
    assert Calls.sampler[0]["model"].shift == 8.0


def test_pose_percent_validation(node_module):
    with pytest.raises(ValueError):
        run(node_module, pose_frames=81, pose_start_percent=0.8, pose_end_percent=0.2)


def test_empty_schedule_is_an_error(node_module):
    with pytest.raises(ValueError):
        run(node_module, pose_frames=81, denoise=0.0)


def test_old_core_is_rejected(node_module, monkeypatch):
    class OldAnimate(FakeWanAnimate2ToVideo):
        RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")

    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "WanAnimate2ToVideo", OldAnimate)
    with pytest.raises(RuntimeError, match="Update ComfyUI"):
        run(node_module, pose_frames=81)


def test_overlap_follows_the_core_constant(node_module, monkeypatch):
    # a core that keeps 5 motion frames: the loop must still land exactly
    monkeypatch.setattr(FakeWanAnimate2ToVideo, "CONTINUE_MOTION_FRAMES", 5)
    images, count, plan = run(node_module, pose_frames=200, frames_per_chunk=49)
    assert count == 200
    assert plan.endswith("overlap 5)")
    assert all(c["continue"] == 5 for c in Calls.animate[1:])
