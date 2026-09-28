"""The Pose Detection node: supplied boxes that never build the detector; the Pose Config node: the
forearm_limit, limb_dedup and back_view_face widgets, off and marked experimental, and the edge_snap
widget, on and not experimental. Fake models,
synthetic frames. The node loads models/common/download.py, which imports folder_paths at its top,
so this runs where ComfyUI is importable (with the ComfyUI root on PYTHONPATH) and is skipped
elsewhere:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_nodes_pose.py
"""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("folder_paths")

from names import nodes  # noqa: E402
from pose_fakes import FakeDetector, FakePose, frames, loader, pose  # noqa: E402

EXPERIMENTAL = ("forearm_limit", "limb_dedup", "back_view_face")
EXPERIMENTAL_TOOLTIP = ("Experimental: off by default; in a diffusion comparison it gave no clear gain and can "
                        "remove a correct part. ")


def test_supplied_boxes_never_build_the_detector(monkeypatch):
    monkeypatch.setattr(pose, "_to_device", lambda *models: None)
    built = []

    def build(cls, filename):
        built.append(filename)
        return FakePose() if cls is not loader.Yolo else FakeDetector()

    monkeypatch.setattr(loader, "_load", build)
    out = nodes.BCVPoseDetection().detect(frames(), -1, -1, True, 0.5, bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert built == [loader.POSE_FILE] and len(out[2]) == len(frames())
    built.clear()
    nodes.BCVPoseDetection().detect(frames(), -1, -1, True, 0.5)
    assert built == [loader.DETECTOR_FILE, loader.POSE_FILE]


def test_pose_config_shows_the_forearm_limit_off():
    from names import spec

    inputs = spec("BCVPoseConfig")["required"]
    assert list(inputs) == [f.name for f in pose.PoseConfig.__dataclass_fields__.values()]
    kind, options = inputs["forearm_limit"]
    assert (kind, {k: options[k] for k in ("default", "min", "max", "step")}) == (
        "FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1})
    assert "0 is off" in options["tooltip"]
    assert options["tooltip"].startswith(EXPERIMENTAL_TOOLTIP + "Leave out of the pose images a wrist")


def test_pose_config_shows_the_limb_dedup_off():
    from names import spec

    inputs = spec("BCVPoseConfig")["required"]
    kind, options = inputs["limb_dedup"]
    assert (kind, options["default"], set(options)) == ("BOOLEAN", False, {"default", "display_name", "tooltip"})
    tooltip = options["tooltip"]
    assert tooltip.startswith(EXPERIMENTAL_TOOLTIP + "Leave out of the pose images a hand or an arm")
    assert all(part in tooltip for part in ("a hand on the other hand", "a whole arm (elbow, wrist and hand) along "
                                            "the other arm", "pose_data keeps the keypoints"))
    # the one switch: the separate hand and arm switches are gone
    assert "hand_dedup" not in inputs and "mirror_rule" not in inputs


def test_pose_config_shows_the_back_view_face_off():
    from names import spec

    inputs = spec("BCVPoseConfig")["required"]
    assert list(inputs)[-2] == "back_view_face"
    kind, options = inputs["back_view_face"]
    assert (kind, options["default"], set(options)) == ("BOOLEAN", False, {"default", "display_name", "tooltip"})
    tooltip = options["tooltip"]
    assert tooltip.startswith(EXPERIMENTAL_TOOLTIP + "Leave the nose and both eyes out of the pose images")
    assert all(part in tooltip for part in (
        "Leave the nose and both eyes out of the pose images", "seen from behind",
        "the mean confidence of the 17 jaw-line face keypoints under 0.835",
        "ViTPose invents a nose and eyes on the back of the head, up to 0.99 confident",
        "flips side", "The ears keep the head's place", "pose_data keeps the keypoints"))


def test_pose_config_shows_the_edge_snap_on():
    from names import spec

    inputs = spec("BCVPoseConfig")["required"]
    assert list(inputs)[-1] == "edge_snap"
    kind, options = inputs["edge_snap"]
    assert (kind, options["default"], set(options)) == ("BOOLEAN", True, {"default", "tooltip"})
    tooltip = options["tooltip"]
    assert all(part in tooltip for part in (
        "within 15% of the box's size from a frame edge", "SAM 3.1 Multiplex's box prompt",
        "clothing at the frame edge", "The same box also cuts the pose crop", "nearly every frame",
        "Supplied bboxes are snapped too", "An on/off comparison is planned"))


def test_pose_config_marks_the_three_draw_rules_experimental_and_nothing_else():
    # the label is the input's display_name, which the frontend shows as the widget's label; the
    # input name saved workflows use stays the field name
    from names import spec

    inputs = spec("BCVPoseConfig")["required"]
    for name, (_, options) in inputs.items():
        if name in EXPERIMENTAL:
            assert options["display_name"] == f"{name} (experimental)"
            assert options["tooltip"].startswith(EXPERIMENTAL_TOOLTIP)
        else:
            assert "display_name" not in options and "xperimental" not in options["tooltip"], name
    assert [name for name, (_, options) in inputs.items() if "display_name" in options] == list(EXPERIMENTAL)
    assert {"edge_snap", "detection_threshold", "box_window", "min_keypoint_conf"} <= set(inputs) - set(EXPERIMENTAL)
