"""The SCAIL-2 nodes: SCAIL-2 Preprocess computes exactly what Pose Detection (in the modes that
read the pose), SAM 3.1 Multiplex Video Track and SCAIL-2 Colored Mask compute when chained, and
the SCAIL-2 Long Video Sampler's widget order
and defaults (read under the sampler_fakes stubs), whose shift / scheduler / steps give the sigmas
the official ComfyUI SCAIL-2 template samples with, computed by ComfyUI itself. Fake SAM model, synthetic frames. Runs where
ComfyUI is importable (with the ComfyUI root on PYTHONPATH):

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_nodes_scail2.py
"""
import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("folder_paths")

from names import nodes  # noqa: E402
from pose_fakes import pose  # noqa: E402
from sam3_1_multiplex_fakes import sam3  # noqa: E402
from sampler_fakes import SCAIL2, node_module  # noqa: E402,F401
from scail2_fakes import scail2  # noqa: E402
from test_nodes_wiring import fake_models, same  # noqa: E402,F401

# the template's BasicScheduler("simple", 6) on the model's own shift 8 (report: its scheduler
# bypasses ModelSamplingSD3(5))
TEMPLATE_SIGMAS = [1.0, 0.9757, 0.9413, 0.8889, 0.8005, 0.616, 0.0]


def clip_frames(n=4, seed=0):
    return torch.rand(n, 64, 32, 3, generator=torch.Generator().manual_seed(seed))


def spy(monkeypatch, cls, name):
    """Records each call of `cls.name`, its arguments by name (defaults filled in) and what it
    returned, and calls through."""
    real = getattr(cls, name)
    calls = []

    def recorded(self, *args, **kwargs):
        bound = inspect.signature(real).bind(self, *args, **kwargs)
        bound.apply_defaults()
        out = real(self, *args, **kwargs)
        calls.append(({k: v for k, v in bound.arguments.items() if k != "self"}, out))
        return out

    monkeypatch.setattr(cls, name, recorded)
    return calls


@pytest.mark.parametrize("mode", ["prompt", "box_keypoint", "prompt_pose"])
@pytest.mark.parametrize("replacement_mode", [False, True])
def test_the_preprocess_wrapper_is_the_nodes_chained(fake_models, replacement_mode, mode):
    images, reference = clip_frames(), clip_frames(1, seed=1)
    wrapped = nodes.BCVSCAIL2Preprocess().process(images, reference, replacement_mode, mode, "person")

    # Pose Detection at its default widgets, only where the mode reads the pose
    pose_data = None if mode == "prompt" else nodes.BCVPoseDetection().detect(images, -1, -1, True, 0.5)[1]
    (mask,) = nodes.BCVSAM3VideoTrack().track(images, mode, "person", 1, -1, pose_data=pose_data)
    (reference_mask,) = nodes.BCVSAM3VideoTrack().track(reference, "prompt", "person", 1, -1)
    pose_video_mask, reference_image_mask = nodes.BCVSCAIL2ColoredMask().render(mask, replacement_mode, reference_mask)
    chained = (images, pose_video_mask, reference_image_mask, mask, reference_mask)

    assert len(wrapped) == len(nodes.BCVSCAIL2Preprocess.RETURN_NAMES)
    for name, a, b in zip(nodes.BCVSCAIL2Preprocess.RETURN_NAMES, wrapped, chained):
        assert same(a, b), name
    assert wrapped[0] is images  # the driving video is the pose input, unchanged
    # the whole driving clip once in the mode, with the pose where the mode reads it; then the
    # reference from the prompt alone in every mode
    calls = fake_models.calls[:2]
    assert [(c["mode"], c["prompt"], c["pose_data"]) for c in calls] == [(mode, "person", mode != "prompt"),
                                                                          ("prompt", "person", False)]


def test_prompt_mode_runs_no_pose(fake_models, monkeypatch, caplog):
    detections = spy(monkeypatch, nodes.BCVPoseDetection, "detect")
    tracks = spy(monkeypatch, nodes.BCVSAM3VideoTrack, "track")
    caplog.set_level("INFO")
    images, reference = clip_frames(), clip_frames(1, seed=1)
    nodes.BCVSCAIL2Preprocess().process(images, reference, False, "prompt", "person")
    assert detections == []
    assert [(args["mode"], args["pose_data"]) for args, _ in tracks] == [("prompt", None)] * 2
    assert "pose_config not used" not in caplog.text
    # a connected pose_config is not read, and the console says so
    nodes.BCVSCAIL2Preprocess().process(images, reference, False, "prompt", "person", pose_config=pose.PoseConfig())
    assert detections == []
    assert "prompt mode runs no pose; pose_config not used" in caplog.text


@pytest.mark.parametrize("mode", ["box_keypoint", "prompt_pose"])
def test_the_pose_modes_detect_the_pose_once_on_the_driving_frames(fake_models, monkeypatch, mode):
    detections = spy(monkeypatch, nodes.BCVPoseDetection, "detect")
    tracks = spy(monkeypatch, nodes.BCVSAM3VideoTrack, "track")
    images, reference = clip_frames(), clip_frames(1, seed=1)
    config = pose.PoseConfig(detection_threshold=0.3)
    nodes.BCVSCAIL2Preprocess().process(images, reference, False, mode, "person", pose_config=config)

    [(detected, (_, pose_data, _, _))] = detections
    assert detected["images"] is images and detected["pose_config"] is config and detected["bboxes"] is None
    # SCAIL-2 draws no pose: Pose Detection's default widgets
    assert {k: detected[k] for k in ("body_stick_width", "hand_stick_width", "draw_head", "draw_threshold")} == \
        {"body_stick_width": -1, "hand_stick_width": -1, "draw_head": True, "draw_threshold": 0.5}
    [(driving, _), (ref, _)] = tracks
    assert driving["images"] is images and driving["mode"] == mode and driving["pose_data"] is pose_data
    assert ref["images"] is reference and ref["mode"] == "prompt" and ref["pose_data"] is None


def test_a_connected_reference_mask_is_not_tracked(fake_models):
    images, reference = clip_frames(), clip_frames(1, seed=1)
    reference_mask = torch.zeros(1, 64, 32)
    reference_mask[:, 10:40, 5:20] = 1.0
    config = sam3.SAM3Config()
    out = nodes.BCVSCAIL2Preprocess().process(images, reference, False, "prompt", "person", reference_mask=reference_mask,
                                              sam3_config=config)
    assert len(fake_models.calls) == 1 and fake_models.calls[0]["config"] is config
    assert out[4] is reference_mask
    assert same(out[2], nodes.BCVSCAIL2ColoredMask().render(out[3], False, reference_mask)[1])


def test_black_background_blacks_out_the_driving_video_around_the_tracked_person(fake_models):
    images, reference = clip_frames(), clip_frames(1, seed=1)
    out = nodes.BCVSCAIL2Preprocess().process(images, reference, False, "prompt", "person", black_background=True)
    plain = nodes.BCVSCAIL2Preprocess().process(images, reference, False, "prompt", "person")
    assert same(out[0], scail2.driving_on_black(images, out[3]))
    assert not same(out[0], images)
    for name, a, b in zip(nodes.BCVSCAIL2Preprocess.RETURN_NAMES[1:], out[1:], plain[1:]):
        assert same(a, b), name  # the masks do not change


def test_black_background_in_replacement_mode_raises_before_any_tracking(fake_models):
    with pytest.raises(ValueError, match="black_background is for animation mode"):
        nodes.BCVSCAIL2Preprocess().process(clip_frames(), clip_frames(1, seed=1), True, "prompt", "person", black_background=True)
    assert fake_models.calls == []


def test_the_preprocess_widgets_mode_before_prompt_and_black_background_last():
    spec = nodes.BCVSCAIL2Preprocess.INPUT_TYPES()
    required, optional = spec["required"], spec["optional"]
    assert list(required) == ["images", "reference_image", "replacement_mode", "mode", "prompt", "black_background"]
    assert list(optional) == ["reference_mask", "pose_config", "sam3_config"]
    # mode is SAM 3.1 Multiplex Video Track's widget, as on WanAnimate Preprocess
    track = nodes.BCVSAM3VideoTrack.INPUT_TYPES()["required"]
    assert required["mode"] == track["mode"]
    assert required["mode"][0] == ["prompt", "box_keypoint", "prompt_pose"] and required["mode"][1]["default"] == "prompt"
    # the prompt segments the reference in every mode, and its tooltip says so; the widget is the Video Track's
    assert {**required["prompt"][1], "tooltip": None} == {**track["prompt"][1], "tooltip": None}
    assert "the character on the reference image (every mode" in required["prompt"][1]["tooltip"]
    assert "Ignored in box_keypoint mode" not in required["prompt"][1]["tooltip"]
    assert optional["pose_config"][0] == "POSE_CONFIG"
    assert optional["pose_config"][1]["tooltip"].startswith("[box_keypoint, prompt_pose] ")
    assert "Ignored in prompt mode, which runs no pose" in optional["pose_config"][1]["tooltip"]
    assert "The reference image is tracked in prompt mode in every mode" in nodes.BCVSCAIL2Preprocess.DESCRIPTION
    assert required["black_background"] == ("BOOLEAN", required["black_background"][1])
    assert required["black_background"][1]["default"] is False


def widget_defaults(config_cls):
    return {name: options[1]["default"] for name, options in nodes._config_inputs(config_cls).items()}


@pytest.mark.parametrize("with_pose", [False, True])
def test_the_guard_node_passes_the_masks_through_and_is_the_pipeline(with_pose):
    from guard_fakes import clip

    masks, pose_data = clip()
    pose_video_mask, reference_image_mask = scail2.colored_masks(masks, False, masks[:1])
    guard_values, mask_values = widget_defaults(scail2.SCAIL2GuardConfig), widget_defaults(scail2.MaskChecksConfig)
    pose_data = pose_data if with_pose else None
    out = nodes.BCVSCAIL2PreprocessGuard().check(pose_video_mask, reference_image_mask, True, pose_data=pose_data,
                                                 **guard_values, **mask_values)
    assert len(out) == len(nodes.BCVSCAIL2PreprocessGuard.RETURN_NAMES)
    assert out[0] is pose_video_mask and out[1] is reference_image_mask
    expected = scail2.check_scail2(pose_video_mask, reference_image_mask, scail2.SCAIL2GuardConfig(**guard_values),
                                   pose_data=pose_data, mask_config=scail2.MaskChecksConfig(**mask_values))
    assert out[2:4] == expected[2:4] and same(out[4], expected[4])


def test_the_guard_node_widgets():
    spec = nodes.BCVSCAIL2PreprocessGuard.INPUT_TYPES()
    required = spec["required"]
    assert list(required) == ["pose_video_mask", "reference_image_mask", "scail2_guard", "min_reference_iou",
                              *widget_defaults(scail2.MaskChecksConfig)]
    # the SCAIL-2 guard does not run the Mask Guard's two keypoint-mask fails, so it shows no widget for them
    assert not {"head_out_eyes_ears", "large_loss_area"} & set(required)
    assert required["scail2_guard"][1]["default"] is True
    assert "Uncalibrated" in required["min_reference_iou"][1]["tooltip"]
    # mask_loss is judged on the sampler's latent grid, and its tooltip says so; the widget is the Mask
    # Guard's (type, range) with a hand-sized default: on the latent grid motion empties whole cells
    ours, mask_guard = required["max_mask_loss"], nodes._config_inputs(scail2.MaskChecksConfig)["max_mask_loss"]
    assert ours[0] == mask_guard[0] and "latent grid" in ours[1]["tooltip"]
    assert ours[1]["default"] == 0.05 and mask_guard[1]["default"] == 0.0185
    assert {**ours[1], "tooltip": None, "default": None} == {**mask_guard[1], "tooltip": None, "default": None}
    assert list(spec["optional"]) == ["pose_data"] and spec["optional"]["pose_data"][0] == "POSEDATA"
    assert "Optional, but it gives the best result" in spec["optional"]["pose_data"][1]["tooltip"]


def test_the_guard_node_off_never_stops():
    pose_video_mask, reference_image_mask = scail2.colored_masks(torch.zeros(3, 64, 32), False, torch.zeros(1, 64, 32))
    report = nodes.BCVSCAIL2PreprocessGuard().check(pose_video_mask, reference_image_mask, False)[2]
    assert report.startswith("SCAIL-2 guard: passed") and "(off)" in report


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
                                       "pose_strength", "pose_start_percent", "pose_end_percent", "previous_frame_count",
                                       "last_chunk", "tail_padding"]
    assert list(spec["optional"]) == ["sigmas_override", "color_anchor_strength"]
    assert spec["optional"]["color_anchor_strength"][1]["default"] == 0.0
    defaults = {name: required[name][1]["default"] for name in ("width", "height", "frames_per_chunk", "shift", "sampler_name",
                                                                "scheduler", "steps", "cfg", "seed_mode", "replacement_mode",
                                                                "pose_strength", "pose_start_percent", "pose_end_percent",
                                                                "previous_frame_count", "last_chunk", "tail_padding")}
    assert defaults == dict(width=704, height=1280, frames_per_chunk=81, shift=8.0, sampler_name="euler", scheduler="simple",
                            steps=6, cfg=1.0, seed_mode="increment", replacement_mode=False, pose_strength=1.0,
                            pose_start_percent=0.0, pose_end_percent=1.0, previous_frame_count=5, last_chunk="full",
                            tail_padding="last_frame")
    assert required["width"][1]["step"] == required["height"][1]["step"] == 32
    assert required["tail_padding"][0] == ["last_frame", "ping_pong"]
    assert required["previous_frame_count"][1]["step"] == 4


def test_the_default_schedule_is_the_template_s():
    import comfy.model_sampling
    import comfy.samplers

    node = scail2.BCVSCAIL2LongVideoSampler

    # what the loop builds: ModelSamplingSD3(shift) on the model, then BasicScheduler on it
    class Sampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    sampling = Sampling()
    sampling.set_parameters(shift=node.DEFAULT_SHIFT, multiplier=1000)
    sigmas = comfy.samplers.calculate_sigmas(sampling, node.DEFAULT_SCHEDULER, node.DEFAULT_STEPS)
    assert sigmas.tolist() == pytest.approx(TEMPLATE_SIGMAS, abs=1e-3)
