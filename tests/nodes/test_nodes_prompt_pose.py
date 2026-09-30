"""The SAM 3.1 Multiplex nodes' prompt_pose surface: the third `mode` value after the two there
were, prompt still the default, its refine described as points plus the tracker's mask, and the
one SAM 3.1 Multiplex Config field it adds last, the point distance; no field of that node is
marked experimental. The node loads models/sam3_1_multiplex/loader.py, which imports folder_paths
at its top, so this runs where ComfyUI is importable (with the ComfyUI root on PYTHONPATH) and is
skipped elsewhere:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_nodes_prompt_pose.py
"""
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("folder_paths")

from names import spec  # noqa: E402


def test_the_video_track_mode_gains_prompt_pose_and_stays_prompt_by_default():
    choices, options = spec("BCVSAM3VideoTrack")["required"]["mode"]
    assert choices == ["prompt", "box_keypoint", "prompt_pose"] and options["default"] == "prompt"
    assert "prompt_pose: prompt mode's track, without its repairs;" in options["tooltip"]
    assert "as positive points together with the mask the tracker had on that frame" in options["tooltip"]
    assert "The Mask Guard fails a limb or the head the mask leaves out" in options["tooltip"]
    optional = spec("BCVSAM3VideoTrack")["optional"]
    assert optional["pose_data"][1]["tooltip"].startswith("[box_keypoint, prompt_pose] ")
    for name in ("bboxes", "positive_coords", "negative_coords"):
        assert "Ignored in prompt and prompt_pose modes" in optional[name][1]["tooltip"], name


def test_the_config_node_shows_the_prompt_pose_field_last_and_marks_nothing_experimental():
    inputs = spec("BCVSAM3Config")["required"]
    assert list(inputs)[-1] == "pose_point_distance"
    assert [name for name, (_, opts) in inputs.items() if opts["tooltip"].startswith("[prompt_pose]")] == \
        ["pose_point_distance"]
    kind, options = inputs["pose_point_distance"]
    assert (kind, {k: options[k] for k in ("default", "min", "max", "step")}) == (
        "FLOAT", {"default": 0.07, "min": 0.0, "max": 0.5, "step": 0.005})
    assert options["tooltip"].startswith("[prompt_pose] how far outside the tracked mask")
    assert [name for name, (_, opts) in inputs.items() if "display_name" in opts] == []
