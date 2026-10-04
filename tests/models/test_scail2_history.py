"""SCAIL-2's history frames as official SCAIL-2 feeds them (zai-org/SCAIL-2 wan/scail.py,
wan-scail2 branch): on every chained chunk the model gets the encoded previous frames clean at
every step, and the 4 mask channels of the video tokens 1 on those latent frames, 0 elsewhere.

HistorySampler stands in for SamplerCustom and calls the patched model's APPLY_MODEL wrappers as
core does at every sigma: the latent in the sampling space (process_latent_in), the known frames
re-noised (KSamplerX0Inpaint with CONST.noise_scaling), the 4 mask channels all zero (WAN21's
concat_cond of a model built with image_to_video=False), cond and uncond in one batch.
ComfyUI itself is stubbed; skipped when torch is not installed.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (SCAIL2, Calls, FakeModel, FakeWanSCAILToVideo, IndexVAE, PoseFollowingSampler, aligned,  # noqa: E402,F401
                           node_module, run)

SIGMAS = 6  # run()'s steps: the model calls per chunk
LATENT_FRAMES, LATENT_H, LATENT_W = 21, 8, 4  # an 81-frame chunk of run()'s 64 x 32 frames


class BaseModel:
    """The BaseModel a wrapper reaches as executor.class_obj; its process_latent_in (a latent
    format's process_in) here 2 * latent + 1, so a latent that skipped it shows."""

    @staticmethod
    def process_latent_in(latent):
        return latent * 2.0 + 1.0


class Executor:
    """comfy/patcher_extension.py WrapperExecutor: a wrapper gets the executor of the ones after
    it and calls it to go on; the last calls the original. class_obj is the BaseModel."""

    class_obj = BaseModel()

    def __init__(self, wrappers, original):
        self.wrappers, self.original = wrappers, original

    def __call__(self, *args, **kwargs):
        if not self.wrappers:
            return self.original(*args, **kwargs)
        return self.wrappers[0](Executor(self.wrappers[1:], self.original), *args, **kwargs)


class HistorySampler(PoseFollowingSampler):
    """Records, per chunk and sigma, what the wrappers were handed and what reached the model."""

    views = []  # per chunk: [{"x", "c_concat" (handed in), "seen_x", "seen_c" (reached the model)}] per sigma
    c_concat = True  # False: the model call has no mask channels

    @classmethod
    def EXECUTE_NORMALIZED(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        wrappers = [w for keyed in model.wrappers.get("apply_model", {}).values() for w in keyed]
        samples = BaseModel.process_latent_in(latent_image["samples"])
        known = latent_image.get("noise_mask")
        denoise = torch.ones_like(samples) if known is None else known.expand_as(samples)
        noise = torch.randn(samples.shape, generator=torch.Generator().manual_seed(noise_seed))
        chunk = []
        for sigma in sigmas[:-1].tolist():
            state = sigma * noise + (1.0 - sigma) * samples.flip(2)  # any sampler state; the known frames are replaced below
            x = state * denoise + (sigma * noise + (1.0 - sigma) * samples) * (1.0 - denoise)
            x = torch.cat((x, x))  # cond and uncond
            c_concat = torch.zeros(2, 4, *x.shape[2:]) if cls.c_concat else None
            view = {"x": x, "x_before": x.clone(), "c_concat": c_concat}

            def model_call(seen_x, t, seen_c, c_crossattn, control, transformer_options, view=view):
                view.update(seen_x=seen_x, seen_c=seen_c)
                return seen_x

            Executor(wrappers, model_call)(x, sigma, c_concat, None, None, {})
            chunk.append(view)
        cls.views.append(chunk)
        return PoseFollowingSampler.EXECUTE_NORMALIZED(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)


@pytest.fixture
def history(aligned, monkeypatch):
    """The sampler Names, with HistorySampler as SamplerCustom."""
    HistorySampler.views = []
    HistorySampler.c_concat = True
    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "SamplerCustom", HistorySampler)
    return aligned


def clean_history(values):
    """The history latent the model must see, written out: IndexVAE puts pixel frame 0 in channel
    0 of latent frame 0 and frame 4k - 3 + j in channel j of latent frame k, everything else 0;
    then process_latent_in (2 * latent + 1)."""
    latent = torch.zeros(1, 16, (len(values) - 1) // 4 + 1, LATENT_H, LATENT_W)
    latent[0, 0, 0] = values[0]
    for f, value in enumerate(values[1:], start=1):
        latent[0, (f - 1) % 4, (f - 1) // 4 + 1] = value
    return latent * 2.0 + 1.0


# previous_frame_count -> (the run's frames: two chunks, the previous frames' driving frames, their latent frames)
HISTORY = {5: (157, [76.0, 77.0, 78.0, 79.0, 80.0], 2), 1: (161, [80.0], 1)}


@pytest.mark.parametrize("previous_frame_count", [5, 1])
def test_the_model_sees_the_clean_history_and_its_mask_at_every_sigma(history, previous_frame_count):
    frames, values, known = HISTORY[previous_frame_count]
    images, count, plan = run(history, pose_frames=frames, node=SCAIL2, vae=IndexVAE(), previous_frame_count=previous_frame_count)
    assert [c["previous"] for c in Calls.animate] == [None, previous_frame_count]
    chunk = HistorySampler.views[1]
    assert len(chunk) == SIGMAS
    mask = torch.zeros(2, 4, LATENT_FRAMES, LATENT_H, LATENT_W)
    mask[:, :, :known] = 1.0
    expected = clean_history(values).expand(2, -1, -1, -1, -1)
    for view in chunk:
        assert torch.equal(view["seen_x"][:, :, :known], expected)  # clean, not re-noised
        assert torch.equal(view["seen_x"][:, :, known:], view["x"][:, :, known:])  # the generated frames untouched
        assert torch.equal(view["seen_c"], mask)
        assert torch.equal(view["x"], view["x_before"])  # the sampler's tensor is not written into
    # the frames stay aligned, the history trimmed back off
    assert images[:, 0, 0, 0].tolist() == [float(i) for i in range(frames)]


def test_the_first_chunk_reaches_the_model_untouched(history):
    run(history, pose_frames=81, node=SCAIL2, vae=IndexVAE())
    assert len(HistorySampler.views) == 1 and len(HistorySampler.views[0]) == SIGMAS
    for view in HistorySampler.views[0]:
        # the same tensors: bit-identical to a run without the wrapper
        assert view["seen_x"] is view["x"] and view["seen_c"] is view["c_concat"]


def test_every_chunk_after_the_first_gets_its_own_history(history):
    run(history, pose_frames=240, node=SCAIL2, vae=IndexVAE())
    assert len(HistorySampler.views) == 4
    assert all(view["seen_x"] is view["x"] for view in HistorySampler.views[0])
    for chunk, first in zip(HistorySampler.views[1:], (76.0, 152.0, 228.0)):
        expected = clean_history([first + i for i in range(5)]).expand(2, -1, -1, -1, -1)
        assert all(torch.equal(view["seen_x"][:, :, :2], expected) for view in chunk)


def test_the_wrapper_is_installed_once_on_a_clone(history):
    model = FakeModel()
    run(history, pose_frames=81, node=SCAIL2, model=model)
    patched = Calls.sampler[0]["model"]
    assert list(patched.wrappers) == ["apply_model"]
    assert list(patched.wrappers["apply_model"]) == [history.HISTORY_WRAPPER]
    assert len(patched.wrappers["apply_model"][history.HISTORY_WRAPPER]) == 1
    assert model.wrappers == {}  # the model the node was handed is not patched


def test_the_history_calls_are_logged_per_chained_chunk(history, caplog):
    caplog.set_level("INFO")
    run(history, pose_frames=157, node=SCAIL2, vae=IndexVAE())
    lines = [line for line in caplog.text.splitlines() if "clean history" in line]
    assert len(lines) == 1
    assert "chunk 2: 2 clean history latent frames, marked in the mask channels, in 6 model calls." in lines[0]


def test_a_model_call_without_mask_channels_is_an_error(history):
    HistorySampler.c_concat = False
    with pytest.raises(RuntimeError, match="not the chunk's 21 latent frames with 4 mask channels"):
        run(history, pose_frames=157, node=SCAIL2, vae=IndexVAE())


def test_previous_frames_without_a_noise_mask_are_an_error(history, monkeypatch):
    class NoNoiseMask(FakeWanSCAILToVideo):
        @classmethod
        def EXECUTE_NORMALIZED(cls, **kwargs):
            output = FakeWanSCAILToVideo.EXECUTE_NORMALIZED(**kwargs)
            output.args[2].pop("noise_mask", None)
            return output

    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "WanSCAILToVideo", NoNoiseMask)
    with pytest.raises(RuntimeError, match="returned the previous frames without a noise_mask"):
        run(history, pose_frames=157, node=SCAIL2, vae=IndexVAE())
