"""The SAM 3.1 Multiplex nodes' prompt_pose surface: the third `mode` value after the two there
were, prompt still the default, and the two SAM 3.1 Multiplex Config fields it adds last - the
point distance, and the one experimental switch, marked the way Pose Config marks its own. The
node loads models/sam3_1_multiplex/loader.py, which imports folder_paths at its top, so this runs
where ComfyUI is importable (with the ComfyUI root on PYTHONPATH) and is skipped elsewhere:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_nodes_prompt_pose.py
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("folder_paths")

from names import spec  # noqa: E402

EXPERIMENTAL_TOOLTIP = ("Experimental: off by default; Meta decodes the first refine of a frame from its points "
                        "alone, and this variant is not yet compared on real clips. ")


def test_the_video_track_mode_gains_prompt_pose_and_stays_prompt_by_default():
    choices, options = spec("BCVSAM3VideoTrack")["required"]["mode"]
    assert choices == ["prompt", "box_keypoint", "prompt_pose"] and options["default"] == "prompt"
    assert "prompt_pose: prompt mode's track;" in options["tooltip"]
    assert "Mask Guard reports all three" in options["tooltip"]
    optional = spec("BCVSAM3VideoTrack")["optional"]
    assert optional["pose_data"][1]["tooltip"].startswith("[box_keypoint, prompt_pose] ")
    for name in ("bboxes", "positive_coords", "negative_coords"):
        assert "Ignored in prompt and prompt_pose modes" in optional[name][1]["tooltip"], name


def test_the_config_node_shows_the_prompt_pose_fields_last():
    inputs = spec("BCVSAM3Config")["required"]
    assert list(inputs)[-2:] == ["pose_point_distance", "pose_refine_with_mask"]
    kind, options = inputs["pose_point_distance"]
    assert (kind, {k: options[k] for k in ("default", "min", "max", "step")}) == (
        "FLOAT", {"default": 0.07, "min": 0.0, "max": 0.5, "step": 0.005})
    assert options["tooltip"].startswith("[prompt_pose] how far outside the tracked mask")
    assert "display_name" not in options


def test_pose_refine_with_mask_is_off_and_the_one_experimental_switch_of_the_config_node():
    inputs = spec("BCVSAM3Config")["required"]
    kind, options = inputs["pose_refine_with_mask"]
    assert (kind, options["default"], set(options)) == ("BOOLEAN", False, {"default", "display_name", "tooltip"})
    assert options["display_name"] == "pose_refine_with_mask (experimental)"
    assert options["tooltip"].startswith(EXPERIMENTAL_TOOLTIP + "[prompt_pose] what a frame given points is decoded from")
    assert all(part in options["tooltip"] for part in ("off (default): the points alone, as Meta does",
                                                       "on: the points plus the mask the first pass had"))
    assert [name for name, (_, opts) in inputs.items() if "display_name" in opts] == ["pose_refine_with_mask"]
