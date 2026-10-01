"""The Pose Detection node: supplied boxes that never build the detector, the optional width and
height the pose images are drawn at; the Pose Config node: the
box_window and edge_snap widgets and the forearm_limit, limb_dedup and back_view_face widgets, off and
marked experimental, and nothing else marked. Fake models, synthetic frames. The node loads models/common/download.py, which imports folder_paths at its top,
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
BOX_SWITCHES = ("box_window", "edge_snap")
BOX_SWITCH_TOOLTIP = ("Experimental: off by default; in a pose comparison it gave no net gain (the errors moved "
                      "between frames rather than going away). ")


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


def test_width_and_height_are_optional_sockets_after_the_others():
    from names import spec

    optional = spec("BCVPoseDetection")["optional"]
    # pose_model, the widget after them, is in test_nodes_sapiens2.py
    assert list(optional) == ["bboxes", "pose_config", "width", "height", "pose_model"]
    for name in ("width", "height"):
        kind, options = optional[name]
        assert kind == "INT" and options["forceInput"] is True and options["min"] == 1, name
        assert "drawn at width x height" in options["tooltip"]
    # the wrappers draw at the frame size, as before
    for key in ("BCVWanAnimatePreprocess", "BCVSCAIL2Preprocess"):
        assert not {"width", "height"} & (set(spec(key)["required"]) | set(spec(key)["optional"])), key


def test_width_and_height_draw_the_pose_images_at_that_size(monkeypatch):
    from test_nodes_wiring import same

    monkeypatch.setattr(pose, "_to_device", lambda *models: None)
    monkeypatch.setattr(loader, "load_pose_models", lambda detector=True: (FakeDetector() if detector else None, FakePose()))
    images = frames()
    plain = nodes.BCVPoseDetection().detect(images, -1, -1, True, 0.5)
    sized = nodes.BCVPoseDetection().detect(images, -1, -1, True, 0.5, width=60, height=60)
    assert sized[0].shape == (len(images), 60, 60, 3)
    assert torch.equal(sized[0], pose.draw(plain[1], size=(60, 60)))
    # pose_data, the boxes and the key frame's points stay at the frame size
    assert same(sized[1:], plain[1:])
    for width, height in ((60, None), (None, 60)):
        with pytest.raises(ValueError, match="connect both, or neither"):
            nodes.BCVPoseDetection().detect(images, -1, -1, True, 0.5, width=width, height=height)


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


def test_pose_config_shows_the_box_window_off():
    from names import spec

    inputs = spec("BCVPoseConfig")["required"]
    kind, options = inputs["box_window"]
    assert (kind, {k: options[k] for k in ("default", "min", "max", "step")}) == (
        "INT", {"default": 0, "min": 0, "max": 30, "step": 1})
    tooltip = options["tooltip"]
    assert tooltip.startswith(BOX_SWITCH_TOOLTIP + "Frames either side whose person boxes each frame's box is widened")
    assert all(part in tooltip for part in ("supplied bboxes are widened too", "0 is off"))


def test_pose_config_shows_the_edge_snap_off():
    from names import spec

    inputs = spec("BCVPoseConfig")["required"]
    assert list(inputs)[-1] == "edge_snap"
    kind, options = inputs["edge_snap"]
    assert (kind, options["default"], set(options)) == ("BOOLEAN", False, {"default", "display_name", "tooltip"})
    tooltip = options["tooltip"]
    assert tooltip.startswith(BOX_SWITCH_TOOLTIP + "Extend a person box edge")
    assert all(part in tooltip for part in (
        "within 15% of the box's size from a frame edge", "SAM 3.1 Multiplex's box prompt",
        "clothing at the frame edge", "The same box also cuts the pose crop", "nearly every frame",
        "Supplied bboxes are snapped too"))


def test_pose_config_marks_the_box_switches_and_the_draw_rules_experimental_and_nothing_else():
    # the label is the input's display_name, which the frontend shows as the widget's label; the
    # input name saved workflows use stays the field name
    from names import spec

    inputs = spec("BCVPoseConfig")["required"]
    for name, (_, options) in inputs.items():
        if name in EXPERIMENTAL + BOX_SWITCHES:
            assert options["display_name"] == f"{name} (experimental)"
            assert options["tooltip"].startswith(EXPERIMENTAL_TOOLTIP if name in EXPERIMENTAL else BOX_SWITCH_TOOLTIP)
        else:
            assert "display_name" not in options and "xperimental" not in options["tooltip"], name
    assert [name for name, (_, options) in inputs.items() if "display_name" in options] == [
        "box_window", "forearm_limit", "limb_dedup", "back_view_face", "edge_snap"]
    assert {"detection_threshold", "min_keypoint_conf"} <= set(inputs) - set(EXPERIMENTAL + BOX_SWITCHES)
