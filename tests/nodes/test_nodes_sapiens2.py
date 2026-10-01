"""The Sapiens2 Pose node and the preprocess wrappers' pose_model widget: the node's surface (Pose
Detection's inputs and outputs plus the `model` combo), the models it builds, and the wrappers
computing exactly what Sapiens2 Pose chained with the other nodes computes. The pose_model widget is
the wrappers' last widget, with ViTPose-H as its default, so a workflow saved before it keeps its
values and its pose. Fake models, synthetic frames. Runs where ComfyUI is importable (with the ComfyUI
root on PYTHONPATH):

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_nodes_sapiens2.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("folder_paths")

from names import nodes, spec  # noqa: E402
from pose_fakes import FakeDetector, FakePose, frames, loader, pose  # noqa: E402
from sam3_1_multiplex_fakes import sam3  # noqa: E402
from sapiens2_fakes import FakeSapiens2, sapiens2  # noqa: E402
from test_nodes_wiring import fake_models, same  # noqa: E402,F401

MODELS = ["5b int8 convrot", "5b bf16", "1b int8 convrot", "1b bf16", "0.8b int8 convrot", "0.8b bf16",
          "0.4b int8 convrot", "0.4b bf16"]
POSE_MODELS = ["ViTPose-H"] + [f"Sapiens2 {m}" for m in MODELS]
WIDGET_TYPES = ("INT", "FLOAT", "BOOLEAN", "STRING")


def widgets(node_id):
    """The node's widgets in the order the frontend creates them (required, then optional), which is
    the order a saved workflow stores their values in: every input that is a combo or a widget type."""
    types = spec(node_id)
    return [name for group in ("required", "optional") for name, options in types.get(group, {}).items()
            if isinstance(options[0], list) or options[0] in WIDGET_TYPES]


def test_the_node_is_pose_detection_with_a_model_combo():
    ours, theirs = spec("BCVSapiens2Pose"), spec("BCVPoseDetection")
    assert list(ours["required"]) == ["images", "model", "body_stick_width", "hand_stick_width", "draw_head",
                                      "draw_threshold"]
    assert list(ours["optional"]) == ["bboxes", "pose_config"]
    assert {k: v for k, v in ours["required"].items() if k != "model"} == theirs["required"]
    assert ours["optional"] == theirs["optional"]
    values, options = ours["required"]["model"]
    assert values == MODELS and options["default"] == "5b int8 convrot"
    assert "downloaded" in options["tooltip"].lower()
    node = nodes.NODE_CLASS_MAPPINGS["BCVSapiens2Pose"]
    assert (node.RETURN_TYPES, node.RETURN_NAMES) == (nodes.BCVPoseDetection.RETURN_TYPES, nodes.BCVPoseDetection.RETURN_NAMES)
    assert node.CATEGORY == nodes.BCVPoseDetection.CATEGORY
    assert "ViTPose-H" in node.DESCRIPTION and "downloaded on first use" in node.DESCRIPTION


@pytest.mark.parametrize("node_id", ["BCVWanAnimatePreprocess", "BCVSCAIL2Preprocess"])
def test_the_wrappers_pose_model_is_their_last_widget_with_vitpose_as_default(node_id):
    assert widgets(node_id)[-1] == "pose_model"
    values, options = spec(node_id)["optional"]["pose_model"]
    assert values == POSE_MODELS and options["default"] == "ViTPose-H"


def test_the_guard_wrappers_run_no_pose_and_have_no_pose_model():
    for node_id in ("BCVWanAnimatePreprocessGuard", "BCVSCAIL2PreprocessGuard"):
        types = spec(node_id)
        assert "pose_model" not in {**types["required"], **types.get("optional", {})}, node_id


def test_the_node_builds_vitpose_the_detector_and_the_chosen_sapiens2_file(monkeypatch):
    built = []

    def build(cls, filename, repo=None):
        built.append((filename, repo))
        return FakeDetector() if cls is loader.Yolo else FakeSapiens2() if repo else FakePose()

    monkeypatch.setattr(loader, "_load", build)
    monkeypatch.setattr(pose, "_to_device", lambda *models: None)
    out = nodes.NODE_CLASS_MAPPINGS["BCVSapiens2Pose"]().detect(frames(), "0.8b bf16", -1, -1, True, 0.5)
    assert built == [(loader.DETECTOR_FILE, None), (loader.POSE_FILE, None),
                     ("sapiens2_pose_0.8b_bf16.safetensors", "beycanai/sapiens2-convrot")]
    assert len(out) == 4 and len(out[2]) == len(frames())
    built.clear()
    nodes.NODE_CLASS_MAPPINGS["BCVSapiens2Pose"]().detect(frames(), "5b int8 convrot", -1, -1, True, 0.5,
                                                          bboxes=[(30.0, 20.0, 90.0, 140.0)])
    assert built == [(loader.POSE_FILE, None), ("sapiens2_pose_5b_int8_convrot.safetensors", "beycanai/sapiens2-convrot")]


@pytest.fixture
def fake_sapiens2(fake_models, monkeypatch):
    models = {}
    monkeypatch.setattr(sapiens2, "load_pose_estimator", lambda name: models.setdefault(name, FakeSapiens2()))
    return models


@pytest.mark.parametrize("mode", list(sam3.MODES))
def test_the_preprocess_wrapper_with_sapiens2_is_sapiens2_pose_chained(fake_sapiens2, mode):
    images = frames()
    widget_values = dict(body_stick_width=-1, hand_stick_width=-1, draw_head=True, draw_threshold=0.5)
    wrapped = nodes.BCVWanAnimatePreprocess().process(images, face_padding=8, mode=mode, prompt=sam3.PROMPT,
                                                      pose_model="Sapiens2 1b int8 convrot", **widget_values)
    assert list(fake_sapiens2) == ["Sapiens2 1b int8 convrot"]
    pose_images, pose_data, bboxes, key_points = nodes.BCVSapiens2Pose().detect(images, "1b int8 convrot", **widget_values)
    (mask,) = nodes.BCVSAM3VideoTrack().track(images, mode, sam3.PROMPT, 1, -1,
                                             pose_data=pose_data if mode != sam3.MODE_PROMPT else None)
    face_images, face_bboxes = nodes.BCVFaceCrop().crop(images, pose_data, 8)
    chained = (pose_images, face_images, mask, pose_data, bboxes, key_points, face_bboxes)
    for name, a, b in zip(nodes.BCVWanAnimatePreprocess.RETURN_NAMES, wrapped, chained):
        assert same(a, b), name
    # and not what Pose Detection gives: the body is Sapiens2's
    vitpose = nodes.BCVPoseDetection().detect(images, **widget_values)[1]
    assert not same(pose_data["pose_metas_original"], vitpose["pose_metas_original"])


@pytest.mark.parametrize("mode", ["box_keypoint", "prompt_pose"])
def test_the_scail2_wrapper_with_sapiens2_tracks_on_sapiens2_pose(fake_sapiens2, mode):
    images = frames()[:4]
    reference = frames()[:1]
    wrapped = nodes.BCVSCAIL2Preprocess().process(images, reference, False, mode, "person", pose_model="Sapiens2 0.4b bf16")
    pose_data = nodes.BCVSapiens2Pose().detect(images, "0.4b bf16", -1, -1, True, 0.5)[1]
    (mask,) = nodes.BCVSAM3VideoTrack().track(images, mode, "person", 1, -1, pose_data=pose_data)
    assert same(wrapped[3], mask)
    assert list(fake_sapiens2) == ["Sapiens2 0.4b bf16"]


def test_scail2_prompt_mode_names_an_unused_pose_model(fake_sapiens2, caplog):
    caplog.set_level("INFO")
    nodes.BCVSCAIL2Preprocess().process(frames()[:4], frames()[:1], False, "prompt", "person",
                                        pose_model="Sapiens2 0.4b bf16")
    assert fake_sapiens2 == {}
    assert "[BCVideoNodes] prompt mode runs no pose; pose_model not used" in [r.getMessage() for r in caplog.records]


def test_an_unknown_pose_model_raises():
    with pytest.raises(ValueError, match="pose_model 'Sapiens2 2b bf16': expected one of ViTPose-H, Sapiens2 5b"):
        sapiens2.detect_pose("Sapiens2 2b bf16", frames(), -1, -1, True, 0.5)
