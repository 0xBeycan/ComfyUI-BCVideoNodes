"""The SCAIL-2 nodes: SCAIL-2 Preprocess computes exactly what SAM 3.1 Multiplex Video Track and
SCAIL-2 Colored Mask compute when chained, and the Wan SCAIL-2 Long Video Sampler's widget order
and defaults (read under the sampler_fakes stubs), whose shift / scheduler / steps give the sigmas
the official ComfyUI SCAIL-2 template samples with, computed by ComfyUI itself. Fake SAM model, synthetic frames. Runs where
ComfyUI is importable (with the ComfyUI root on PYTHONPATH):

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_nodes_scail2.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("folder_paths")

from names import nodes  # noqa: E402
from sam3_1_multiplex_fakes import sam3  # noqa: E402
from sampler_fakes import SCAIL2, node_module  # noqa: E402,F401
from scail2_fakes import scail2  # noqa: E402
from test_nodes_wiring import fake_models, same  # noqa: E402,F401

# the template's BasicScheduler("simple", 6) on the model's own shift 8 (report: its scheduler
# bypasses ModelSamplingSD3(5))
TEMPLATE_SIGMAS = [1.0, 0.9757, 0.9413, 0.8889, 0.8005, 0.616, 0.0]


def clip_frames(n=4, seed=0):
    return torch.rand(n, 64, 32, 3, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_the_preprocess_wrapper_is_the_two_nodes_chained(fake_models, replacement_mode):
    images, reference = clip_frames(), clip_frames(1, seed=1)
    wrapped = nodes.BCVSCAIL2Preprocess().process(images, reference, replacement_mode, "person")

    (mask,) = nodes.BCVSAM3VideoTrack().track(images, sam3.MODE_PROMPT, "person", 1, -1)
    (reference_mask,) = nodes.BCVSAM3VideoTrack().track(reference, sam3.MODE_PROMPT, "person", 1, -1)
    pose_video_mask, reference_image_mask = nodes.BCVSCAIL2ColoredMask().render(mask, replacement_mode, reference_mask)
    chained = (images, pose_video_mask, reference_image_mask, mask, reference_mask)

    assert len(wrapped) == len(nodes.BCVSCAIL2Preprocess.RETURN_NAMES)
    for name, a, b in zip(nodes.BCVSCAIL2Preprocess.RETURN_NAMES, wrapped, chained):
        assert same(a, b), name
    assert wrapped[0] is images  # the driving video is the pose input, unchanged
    # the whole driving clip once, then the reference, both from the prompt alone
    calls = fake_models.calls[:2]
    assert [(c["mode"], c["prompt"], c["pose_data"]) for c in calls] == [(sam3.MODE_PROMPT, "person", False)] * 2


def test_a_connected_reference_mask_is_not_tracked(fake_models):
    images, reference = clip_frames(), clip_frames(1, seed=1)
    reference_mask = torch.zeros(1, 64, 32)
    reference_mask[:, 10:40, 5:20] = 1.0
    config = sam3.SAM3Config()
    out = nodes.BCVSCAIL2Preprocess().process(images, reference, False, "person", reference_mask=reference_mask, sam3_config=config)
    assert len(fake_models.calls) == 1 and fake_models.calls[0]["config"] is config
    assert out[4] is reference_mask
    assert same(out[2], nodes.BCVSCAIL2ColoredMask().render(out[3], False, reference_mask)[1])


def test_the_colored_mask_node_is_the_pipeline():
    mask = torch.zeros(3, 64, 32)
    mask[:, 20:40, 10:20] = 1.0
    assert same(nodes.BCVSCAIL2ColoredMask().render(mask, True, reference_mask=mask[:1]),
                scail2.colored_masks(mask, True, mask[:1]))


# --- the sampler node --------------------------------------------------------------------------

SHARED = ["model", "positive", "negative", "vae", "reference_image", "pose_video", "width", "height", "frames_per_chunk",
          "total_frames", "shift", "sampler_name", "scheduler", "steps", "denoise", "cfg", "seed", "seed_mode"]


def test_the_sampler_widgets_and_defaults(node_module):
    spec = getattr(node_module, SCAIL2).INPUT_TYPES()
    required = spec["required"]
    assert list(required) == SHARED + ["clip_vision", "pose_video_mask", "reference_image_mask", "replacement_mode",
                                       "pose_strength", "pose_start_percent", "pose_end_percent", "previous_frame_count"]
    assert list(spec["optional"]) == ["sigmas_override"]
    defaults = {name: required[name][1]["default"] for name in ("width", "height", "frames_per_chunk", "shift", "sampler_name",
                                                                "scheduler", "steps", "cfg", "seed_mode", "replacement_mode",
                                                                "pose_strength", "pose_start_percent", "pose_end_percent",
                                                                "previous_frame_count")}
    assert defaults == dict(width=704, height=1280, frames_per_chunk=81, shift=8.0, sampler_name="euler", scheduler="simple",
                            steps=6, cfg=1.0, seed_mode="increment", replacement_mode=False, pose_strength=1.0,
                            pose_start_percent=0.0, pose_end_percent=1.0, previous_frame_count=5)
    assert required["width"][1]["step"] == required["height"][1]["step"] == 32
    assert required["previous_frame_count"][1]["step"] == 4


def test_the_default_schedule_is_the_template_s():
    import comfy.model_sampling
    import comfy.samplers

    node = scail2.BCVWanSCAIL2LongVideoSampler

    # what the loop builds: ModelSamplingSD3(shift) on the model, then BasicScheduler on it
    class Sampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    sampling = Sampling()
    sampling.set_parameters(shift=node.DEFAULT_SHIFT, multiplier=1000)
    sigmas = comfy.samplers.calculate_sigmas(sampling, node.DEFAULT_SCHEDULER, node.DEFAULT_STEPS)
    assert sigmas.tolist() == pytest.approx(TEMPLATE_SIGMAS, abs=1e-3)
