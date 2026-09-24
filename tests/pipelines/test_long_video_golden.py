"""G1 and G11: the chunk loop of the long-video sampler nodes, end to end on the sampler_fakes
core stubs.

Every core call is recorded in order, with its keyword arguments reduced to plain data at the
moment of the call (a tensor becomes (shape, dtype, md5[:8])); the ProgressBar's total and
updates go into the same call log. Two subclassed fakes carry content, so the output images
fingerprint the trims and the chunk order: SamplerCustom adds noise_seed * 1e-6 to the latent,
and the VAE decodes latent frame i to the mean of that frame + i / 1000. A scenario pins
(md5(images), md5(repr(call log)), chunk_plan, str(dtype), md5(log)); an error scenario pins the
error's type and text, and the calls and log lines before it. The log is every record of the
root and BCVideoNodes loggers, as level and message, times normalised.

G11 pins the overlap self-correction and the "ran:" line (plan 12.D F3), which the current core
cannot reach: a WanAnimate2ToVideo that keeps 5 continue_motion frames while its
CONTINUE_MOTION_FRAMES says 1.

Recorded in the ComfyUI venv on CPU (wan_beta runs scipy). ComfyUI itself is stubbed; skipped
when torch is not installed.
"""

import inspect
import logging
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from golden import check, digest, log_text  # noqa: E402
from sampler_fakes import (ANIMATE1, ANIMATE2, LATENT_DOWN, SCAIL2, FakeNodeOutput, FakeProgressBar,  # noqa: E402,F401
                           FakeSamplerCustom, FakeVAE, FakeWanAnimate2ToVideo, node_module, reference_mask, run)

CORE_NODE = {ANIMATE1: "WanAnimateToVideo", ANIMATE2: "WanAnimate2ToVideo", SCAIL2: "WanSCAILToVideo"}
LOGGERS = ("root", "BCVideoNodes")


# --- recording --------------------------------------------------------------------------------

def reduced(value):
    """`value` as plain data with a deterministic repr: a tensor as (shape, dtype, md5[:8]), a
    function as its name and closure, any other object as its type name and attributes."""
    if isinstance(value, torch.Tensor):
        return (tuple(value.shape), str(value.dtype), digest(value)[:8])
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {key: reduced(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(reduced(item) for item in value)
    if isinstance(value, types.FunctionType):
        return ("function", value.__name__, tuple(reduced(cell.cell_contents) for cell in value.__closure__ or ()))
    if hasattr(value, "__dict__"):
        return (type(value).__name__, reduced(vars(value)))
    raise TypeError("no plain-data form for {!r}".format(type(value)))


def recording(node_id, cls, calls):
    """`cls` with its FUNCTION wrapped, classmethod (V3) or instance method (V1) as it is: every
    call appends (node_id, reduced kwargs) to `calls`, then runs the fake."""
    name = cls.FUNCTION
    if isinstance(inspect.getattr_static(cls, name), classmethod):
        def function(owner, **kwargs):
            calls.append((node_id, reduced(kwargs)))
            return getattr(super(subclass, owner), name)(**kwargs)
        function = classmethod(function)
    else:
        def function(self, **kwargs):
            calls.append((node_id, reduced(kwargs)))
            return getattr(super(subclass, self), name)(**kwargs)
    subclass = type(cls.__name__, (cls,), {name: function})
    return subclass


def recording_bar(calls):
    class RecordingBar(FakeProgressBar):
        def __init__(self, total, node_id=None):
            super().__init__(total, node_id)
            calls.append(("ProgressBar", reduced(total), reduced(node_id)))

        def update(self, value):
            super().update(value)
            calls.append(("ProgressBar.update", reduced(value)))

    return RecordingBar


class ContentSamplerCustom(FakeSamplerCustom):
    """Leaves its seed in the latent: noise_seed * 1e-6 is added."""

    @classmethod
    def EXECUTE_NORMALIZED(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        out = dict(latent_image, samples=latent_image["samples"] + noise_seed * 1e-6)
        return FakeNodeOutput(out, dict(out))


class ContentVAE(FakeVAE):
    """Decodes latent frame i to the mean of that frame + i / 1000: one pixel frame for latent 0,
    four for every later one, as FakeVAE counts them."""

    def decode(self, latent):
        latents = latent.shape[2]
        values = latent.mean(dim=(0, 1, 3, 4)) + torch.arange(latents) / 1000
        index = torch.tensor([0] + [i for i in range(1, latents) for _ in range(4)])
        h, w = latent.shape[3] * LATENT_DOWN, latent.shape[4] * LATENT_DOWN
        return values[index].view(1, -1, 1, 1, 1).expand(1, len(index), h, w, 3).clone()


def install(monkeypatch, **replacements):
    """Swap ContentSamplerCustom (and `replacements`) into the stubbed core, and wrap every core
    node and the ProgressBar so they record into the returned list, in call order."""
    calls = []
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    for node_id, cls in dict(mappings, SamplerCustom=ContentSamplerCustom, **replacements).items():
        monkeypatch.setitem(mappings, node_id, recording(node_id, cls, calls))
    monkeypatch.setattr(sys.modules["comfy.utils"], "ProgressBar", recording_bar(calls))
    return calls


def log_lines(caplog):
    records = [record for record in caplog.records if record.name in LOGGERS]
    return "\n".join("{} {}".format(record.levelname, log_text([record])) for record in records)


def outcome(result, calls, caplog):
    images, count, plan_text = result
    assert count == images.shape[0]
    return [digest(images), digest(calls), plan_text, str(images.dtype), digest(log_lines(caplog))]


# --- G1 ---------------------------------------------------------------------------------------

def widget_defaults(module, node):
    """Every widget default of the node, as the graph executor fills them. width and height stay
    run()'s 32x64, so the fake tensors stay small."""
    spec = getattr(module, node).INPUT_TYPES()
    defaults = {name: definition[1]["default"] for section in ("required", "optional")
                for name, definition in spec[section].items()
                if len(definition) > 1 and isinstance(definition[1], dict) and "default" in definition[1]}
    del defaults["width"], defaults["height"]
    return defaults


def seeded(*shape):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(0))


def replacement_inputs(frames):
    """Replacement mode: face, background and a seeded pixel-size mask video, all `frames` long."""
    index = torch.arange(frames, dtype=torch.float32)
    return dict(face_video=index.view(-1, 1, 1, 1).expand(-1, 8, 8, 3).contiguous(),
                background_video=(index / 1000).view(-1, 1, 1, 1).expand(-1, 64, 32, 3).contiguous(),
                character_mask=seeded(frames, 64, 32), clip_vision_output="clip")


SCENARIOS = {
    "a1_default": lambda module: dict(node=ANIMATE1, pose_frames=81, **widget_defaults(module, ANIMATE1)),
    "a2_default": lambda module: dict(node=ANIMATE2, pose_frames=81, **widget_defaults(module, ANIMATE2)),
    "a2_long": lambda module: dict(node=ANIMATE2, pose_frames=360),
    "a2_clip_vision": lambda module: dict(node=ANIMATE2, pose_frames=200, clip_vision_output="clip",
                                          clip_vision_output_pose="static", clip_vision="cv"),
    "a2_hold": lambda module: dict(node=ANIMATE2, pose_frames=100, total_frames=250, frames_per_chunk=49),
    "a1_long": lambda module: dict(node=ANIMATE1, pose_frames=360, frames_per_chunk=77),
    "a1_replacement_mask": lambda module: dict(node=ANIMATE1, pose_frames=150, **replacement_inputs(150)),
    "a1_snap": lambda module: dict(node=ANIMATE1, pose_frames=200, frames_per_chunk=77, continue_motion_max_frames=6),
    "a1_overlap_13_chunk_17": lambda module: dict(node=ANIMATE1, pose_frames=60, frames_per_chunk=17,
                                                  continue_motion_max_frames=13),
    "seed_2_64_minus_1_increment": lambda module: dict(node=ANIMATE2, pose_frames=200, seed=2 ** 64 - 1,
                                                       seed_mode="increment"),
    "sigmas_override": lambda module: dict(node=ANIMATE2, pose_frames=81, sigmas_override=torch.linspace(1.0, 0.0, 4)),
    "s2_default": lambda module: dict(node=SCAIL2, pose_frames=81, **widget_defaults(module, SCAIL2)),
    "s2_long": lambda module: dict(node=SCAIL2, pose_frames=450),
    "s2_replacement": lambda module: dict(node=SCAIL2, pose_frames=200, replacement_mode=True,
                                          reference_image=seeded(1, 64, 32, 3)),
    "s2_hold": lambda module: dict(node=SCAIL2, pose_frames=100, total_frames=250),
    "s2_overlap_9": lambda module: dict(node=SCAIL2, pose_frames=200, previous_frame_count=9),
    # the last_chunk policy other than the node's default, on totals whose last chunk differs
    "a1_last_chunk_full": lambda module: dict(node=ANIMATE1, pose_frames=240, last_chunk="full", **replacement_inputs(240)),
    "a2_last_chunk_full": lambda module: dict(node=ANIMATE2, pose_frames=250, last_chunk="full"),
    "s2_last_chunk_fit": lambda module: dict(node=SCAIL2, pose_frames=240, last_chunk="fit"),
    # tail_padding ping_pong, on a last chunk run past the input (full) and on total_frames past it
    "a1_ping_pong_full": lambda module: dict(node=ANIMATE1, pose_frames=240, last_chunk="full", tail_padding="ping_pong",
                                             **replacement_inputs(240)),
    "a1_ping_pong_hold": lambda module: dict(node=ANIMATE1, pose_frames=100, total_frames=250, tail_padding="ping_pong",
                                             **replacement_inputs(100)),
    "a2_ping_pong_full": lambda module: dict(node=ANIMATE2, pose_frames=250, last_chunk="full", tail_padding="ping_pong"),
    "a2_ping_pong_hold": lambda module: dict(node=ANIMATE2, pose_frames=100, total_frames=250, frames_per_chunk=49,
                                             tail_padding="ping_pong"),
    "s2_ping_pong_full": lambda module: dict(node=SCAIL2, pose_frames=240, tail_padding="ping_pong"),
    "s2_ping_pong_hold": lambda module: dict(node=SCAIL2, pose_frames=100, total_frames=250, tail_padding="ping_pong"),
}

ERRORS = {
    "error_no_pose_frames": lambda module: dict(node=ANIMATE2, pose_frames=0),
    "error_empty_schedule": lambda module: dict(node=ANIMATE2, pose_frames=81, denoise=0.0),
    "error_old_core_a1": lambda module: dict(node=ANIMATE1, pose_frames=81, old_core=True),
    "error_old_core_a2": lambda module: dict(node=ANIMATE2, pose_frames=81, old_core=True),
    "error_pose_start_after_end": lambda module: dict(node=ANIMATE2, pose_frames=81, pose_start_percent=0.8,
                                                      pose_end_percent=0.2),
    "error_overlap_not_below_chunk": lambda module: dict(node=ANIMATE1, pose_frames=200, frames_per_chunk=77,
                                                         continue_motion_max_frames=77),
    "error_mask_background_mismatch": lambda module: dict(node=ANIMATE1, pose_frames=200, frames_per_chunk=77,
                                                          character_mask=seeded(100, 64, 32),
                                                          background_video=torch.zeros(200, 64, 32, 3)),
    "error_old_core_s2": lambda module: dict(node=SCAIL2, pose_frames=81, old_core=True),
    "error_s2_size_not_32": lambda module: dict(node=SCAIL2, pose_frames=81, width=48),
    "error_s2_mask_length": lambda module: dict(node=SCAIL2, pose_frames=100, pose_video_mask=torch.zeros(80, 64, 32, 3)),
    "error_s2_mode_mismatch": lambda module: dict(node=SCAIL2, pose_frames=81, replacement_mode=True,
                                                  reference_image_mask=reference_mask(False)),
    # the driving mask of the other mode: the frame-index mask's first frame is black (animation)
    # and one higher is white (replacement)
    "error_s2_driving_mode_mismatch_replacement": lambda module: dict(
        node=SCAIL2, pose_frames=81, replacement_mode=True,
        pose_video_mask=torch.arange(81, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 64, 32, 3).contiguous()),
    "error_s2_driving_mode_mismatch_animation": lambda module: dict(
        node=SCAIL2, pose_frames=81,
        pose_video_mask=torch.arange(1, 82, dtype=torch.float32).view(-1, 1, 1, 1).expand(-1, 64, 32, 3).contiguous()),
}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_long_video_golden(node_module, monkeypatch, caplog, name):
    calls = install(monkeypatch)
    caplog.set_level(logging.INFO)
    result = run(node_module, vae=ContentVAE(), **SCENARIOS[name](node_module))
    check(__file__, name, outcome(result, calls, caplog))


@pytest.mark.parametrize("name", list(ERRORS))
def test_long_video_error_golden(node_module, monkeypatch, caplog, name):
    calls = install(monkeypatch)
    caplog.set_level(logging.INFO)
    kwargs = ERRORS[name](node_module)
    if kwargs.pop("old_core", False):
        mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
        core = CORE_NODE[kwargs["node"]]
        old = type("OldCore", (mappings[core],), {"RETURN_TYPES": ("CONDITIONING", "CONDITIONING", "LATENT")})
        monkeypatch.setitem(mappings, core, old)
    with pytest.raises(Exception) as error:
        run(node_module, vae=ContentVAE(), **kwargs)
    check(__file__, name, [type(error.value).__name__, str(error.value), digest(calls), digest(log_lines(caplog))])


# --- G11 --------------------------------------------------------------------------------------

class _KeepsFive(FakeWanAnimate2ToVideo):
    CONTINUE_MOTION_FRAMES = 5


class KeepsFiveFrames(FakeWanAnimate2ToVideo):
    """Keeps 5 continue_motion frames while its constant says 1: the planner assumes an overlap
    of 1, the node trims 5."""

    CONTINUE_MOTION_FRAMES = 1

    @classmethod
    def EXECUTE_NORMALIZED(cls, **kwargs):
        return _KeepsFive.EXECUTE_NORMALIZED(**kwargs)


def test_overlap_self_correction_golden(node_module, monkeypatch, caplog):
    calls = install(monkeypatch, WanAnimate2ToVideo=KeepsFiveFrames)
    caplog.set_level(logging.INFO)
    result = run(node_module, pose_frames=200, frames_per_chunk=49, vae=ContentVAE())
    assert "planner assumed 1; using 5" in caplog.text and " ran: " in caplog.text  # the path G11 is about
    check(__file__, "overlap_self_correction", outcome(result, calls, caplog))
