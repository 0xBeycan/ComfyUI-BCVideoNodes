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

ANIMATE1 = "WanAnimateLongVideoSampler"
ANIMATE2 = "WanAnimate2LongVideoSampler"
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
            "max_frames": continue_motion_max_frames,
            "pose": None if pose_video is None else float(pose_video[0, 0, 0, 0]),
            "clip": clip_vision_output, "face": face_video, "background": background_video, "mask": character_mask,
        })
        latent = {"samples": torch.zeros(batch_size, 16, latent_length + trim_latent, height // LATENT_DOWN, width // LATENT_DOWN)}
        return FakeNodeOutput(positive, negative, latent, trim_latent, max(0, ref_motion_latent_length * 4 - 3), video_frame_offset + length)


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
                              "pose_strength": pose_strength, "positive_pose": positive_pose})
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
        "WanAnimateToVideo": FakeWanAnimateToVideo,
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
    ANIMATE2: dict(reference_image_strength=1.0, pose_strength=1.0, pose_start_percent=0.0, pose_end_percent=1.0),
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
    assert spec["required"]["scheduler"][1]["default"] == "simple"
    assert spec["required"]["seed"][1]["control_after_generate"] is True
    assert list(spec["optional"])[-1] == "sigmas_override"
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
    spec = node_module.WanAnimate2LongVideoSampler.INPUT_TYPES()
    assert list(spec["required"]) == ANIMATE2_REQUIRED_ORDER
    assert list(spec["optional"]) == ["positive_pose", "clip_vision_output", "clip_vision_output_pose", "sigmas_override"]
    assert spec["required"]["frames_per_chunk"][1]["default"] == 81
    assert spec["required"]["shift"][1]["default"] == 5.0
    assert spec["required"]["sampler_name"][1]["default"] == "lcm"
    assert spec["required"]["steps"][1]["default"] == 6


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
    spec = node_module.WanAnimateLongVideoSampler.INPUT_TYPES()
    assert list(spec["required"]) == ANIMATE2_REQUIRED_ORDER[:-4] + ["continue_motion_max_frames"]
    assert list(spec["optional"]) == ["clip_vision_output", "face_video", "background_video", "character_mask", "sigmas_override"]
    # core node + official template defaults
    assert spec["required"]["frames_per_chunk"][1]["default"] == 77
    assert spec["required"]["continue_motion_max_frames"][1] == {"default": 5, "min": 1, "max": 16384, "step": 4, "tooltip": spec["required"]["continue_motion_max_frames"][1]["tooltip"]}
    assert spec["required"]["shift"][1]["default"] == 8.0
    assert spec["required"]["sampler_name"][1]["default"] == "euler"
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


def test_animate1_chunk_logging(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, node=ANIMATE1, frames_per_chunk=77, seed=7)
    # 77 + 77 + 57 -> 77 + 72 + 52 = 201 produced, cropped to 200
    assert "chunk 1/3 (frames 1-77/200): length 77, pose offset 0, seed 7" in caplog.text
    assert "chunk 2/3 (frames 78-149/200): length 77, pose offset 77, seed 8" in caplog.text
    assert "chunk 3/3 (frames 150-200/200): length 57, pose offset 149, seed 9" in caplog.text
    assert "chunk 2/3 (frames 78-149/200) done: 72 new frames, 149/200 total" in caplog.text


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
