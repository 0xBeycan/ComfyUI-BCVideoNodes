"""SAM 3.1 Multiplex's entry: input parsing (points, boxes, pose_data) and what `track` rejects,
and ignores in each mode, and what `_multiplex_parts` accepts as a checkpoint, on synthetic
inputs. No model is loaded.

The module imports ComfyUI at its top, so this runs where ComfyUI is importable (the pod, with
the ComfyUI root on PYTHONPATH) and is skipped elsewhere."""
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import kps20, sam3  # noqa: E402


# --- inputs --------------------------------------------------------------------------------

def test_parse_coords_pixel_and_normalised_formats():
    assert sam3.parse_coords(None, 100, 50) == []
    assert sam3.parse_coords("  ", 100, 50) == []
    assert sam3.parse_coords(json.dumps([{"x": 10, "y": 20}]), 100, 50) == [(10.0, 20.0)]
    assert sam3.parse_coords(json.dumps({"points": [[0.5, 0.5]], "labels": [1]}), 100, 50) == [(50.0, 25.0)]


def test_parse_coords_raises_on_bad_input():
    with pytest.raises(ValueError):
        sam3.parse_coords("not json", 100, 50)
    with pytest.raises(ValueError):
        sam3.parse_coords(json.dumps([{"x": 10}]), 100, 50)
    with pytest.raises(ValueError):
        sam3.parse_coords(json.dumps([{"x": 100, "y": 10}]), 100, 50)


def test_key_frame_body_points_parse_as_positive_coords():
    """Pose Detection's key_frame_body_points wired into positive_coords, as its tooltip
    suggests: a confident keypoint outside the frame must not make the string unreadable."""
    from pose_fakes import pose
    body = kps20(0.9)
    body[0, :2] = (0.5, -0.02)      # the nose above the frame
    body[1, :2] = (1.02, 0.5)       # the neck right of it
    pose_data = {"pose_metas_original": [{"width": 100, "height": 50, "keypoints_body": body}]}
    points = sam3.parse_coords(pose.key_frame_body_points(pose_data, 0.5), 100, 50, "positive_coords")
    assert points == [(50.0, 25.0)] * (len(pose.KEY_FRAME_BODY_POINTS) - 2)


def test_parse_bboxes_formats():
    one = sam3.parse_bboxes([10, 20, 30, 40], 3)
    assert len(one) == 3 and one[0].tolist() == [10, 20, 30, 40, 1.0]
    kj = sam3.parse_bboxes(json.dumps([{"startX": 30, "startY": 40, "endX": 10, "endY": 20}]), 2)
    assert kj[1].tolist() == [10, 20, 30, 40, 1.0]
    per_frame = sam3.parse_bboxes([(0, 0, 5, 5), (1, 1, 6, 6)], 2)
    assert per_frame[1].tolist() == [1, 1, 6, 6, 1.0]


def test_parse_bboxes_raises_on_bad_input():
    with pytest.raises(ValueError):
        sam3.parse_bboxes([(0, 0, 5, 5), (1, 1, 6, 6)], 3)
    with pytest.raises(ValueError):
        sam3.parse_bboxes([(0, 0, 5)], 1)
    with pytest.raises(ValueError):
        sam3.parse_bboxes([5, 5, 5, 10], 1)


def test_pose_inputs_reads_detections_and_metas():
    metas = [{"keypoints_body": kps20()}] * 2
    pose_data = {"pose_metas_original": metas, "pose_metas": [], "pose_config": {"min_keypoint_conf": 0.25},
                 "detections": [{"bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.9, "persons": 1},
                                {"bbox": [0.0, 0.0, 10.0, 10.0], "score": -1.0, "persons": 0}]}
    boxes, got, conf = sam3.pose_inputs(pose_data, 2)
    assert got is metas and conf == 0.25
    assert boxes[0].tolist() == [1.0, 2.0, 3.0, 4.0, 0.9] and boxes[1][-1] == -1.0
    with pytest.raises(ValueError):
        sam3.pose_inputs(pose_data, 3)
    with pytest.raises(ValueError):
        sam3.pose_inputs({"pose_metas": []}, 2)
    without = {k: v for k, v in pose_data.items() if k != "pose_config"}
    with pytest.raises(ValueError, match="pose_config.min_keypoint_conf"):
        sam3.pose_inputs(without, 2)


def test_track_rejects_bad_arguments_before_touching_the_model():
    images = torch.zeros(2, 8, 8, 3)
    for kwargs in ({"mode": "boxes"}, {"max_objects": 0}, {"object_index": 1}, {"object_index": -2},
                   {"max_objects": 2, "object_index": 2}, {"config": {}}):
        with pytest.raises((ValueError, TypeError)):
            sam3.track((None, None), images, **kwargs)


# --- precedence: what a mode ignores -------------------------------------------------------

def pose_data_2():
    kps = kps20()
    kps[0] = [0.5, 0.2, 0.9]
    return {"pose_metas_original": [{"keypoints_body": kps}] * 2, "pose_metas": [],
            "pose_config": {"min_keypoint_conf": 0.3},
            "detections": [{"bbox": [1.0, 1.0, 7.0, 7.0], "score": 0.9, "persons": 1}] * 2}


@pytest.fixture
def fake_segment(monkeypatch):
    """segment_by_pose / segment_by_prompt stand-ins that record what they were handed."""
    calls = []

    def by_pose(model, images, bboxes, pose_metas, config, min_keypoint_conf, **kwargs):
        calls.append(("pose", bboxes, kwargs))
        return torch.zeros(images.shape[:3])

    def by_prompt(model, clip, images, prompt, config, result=None):
        calls.append(("prompt", prompt))
        return torch.zeros(images.shape[:3])

    monkeypatch.setattr(sam3, "segment_by_pose", by_pose)
    monkeypatch.setattr(sam3, "segment_by_prompt", by_prompt)
    return calls


def not_used_lines(caplog):
    return [r.getMessage() for r in caplog.records if "not used" in r.getMessage()]


def test_box_keypoint_mode_ignores_the_prompt_mode_inputs_in_one_line(fake_segment, caplog):
    # switching the mode without touching anything else must not raise
    config = sam3.SAM3Config(birth_threshold=0.6, new_object_threshold=0.7, temporal=False, reseed_interval=5,
                             negative_points=4)
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), pose_data=pose_data_2(), bboxes=[2, 2, 6, 6],
                   mode="box_keypoint", prompt="a dog", max_objects=3, object_index=2, config=config)
    (line,) = not_used_lines(caplog)
    for name in ("prompt 'a dog'", "max_objects 3", "object_index 2", "pose_data's person boxes",
                 "sam3_config.birth_threshold", "sam3_config.new_object_threshold",
                 "sam3_config.reseed_interval (temporal off)"):
        assert name in line, name
    assert "negative_points" not in line          # read in box_keypoint mode
    kind, boxes, _ = fake_segment[0]
    assert kind == "pose" and boxes[0].tolist() == [2, 2, 6, 6, 1.0]   # bboxes replaced pose_data's


def test_box_keypoint_mode_reads_the_tracker_fields_with_temporal_on(fake_segment, caplog):
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), pose_data=pose_data_2(), mode="box_keypoint",
                   config=sam3.SAM3Config(reseed_interval=5))
    assert not_used_lines(caplog) == []


def test_prompt_mode_ignores_the_box_keypoint_inputs_in_one_line(fake_segment, caplog):
    config = sam3.SAM3Config(reseed_interval=5, assoc_iou=0.2, birth_threshold=0.6)
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), pose_data=pose_data_2(), bboxes=[2, 2, 6, 6],
                   positive_coords='[{"x": 1, "y": 1}]', negative_coords="[]", config=config)
    (line,) = not_used_lines(caplog)
    for name in ("pose_data", "bboxes", "positive_coords", "negative_coords", "sam3_config.reseed_interval",
                 "sam3_config.assoc_iou (max_objects 1)"):
        assert name in line, name
    assert "birth_threshold" not in line          # read in prompt mode
    assert fake_segment == [("prompt", sam3.PROMPT)]


@pytest.mark.parametrize("mode", list(sam3.MODES))
def test_the_default_wiring_ignores_nothing(fake_segment, caplog, mode):
    pose_data = pose_data_2() if mode == sam3.MODE_BOX_KEYPOINT else None
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), pose_data=pose_data, mode=mode)
    assert not_used_lines(caplog) == []


# --- the model's parts -----------------------------------------------------------------------

def parts_model(sam):
    """The loaded checkpoint as `_multiplex_parts` reads it: model.model.diffusion_model."""
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    model.model.diffusion_model = sam
    return model


def test_multiplex_parts_refuses_what_is_not_a_multiplex_checkpoint_and_returns_the_parts():
    def sam(detector=None, tracker=None, name="Checkpoint"):
        return type(name, (), {"detector": detector, "tracker": tracker})()

    def detector(backbone):
        return type("Detector", (), {"backbone": backbone})()

    multiplex = type("Vision", (), {"multiplex": True})()
    single = type("Vision", (), {"multiplex": False})()
    primitives = ("_compute_backbone_frame", "track_step", "_condition_with_masks", "_deferred_memory_encode",
                  "_forward_sam_heads")
    complete = type("Tracker", (), {n: None for n in primitives})()
    partial = type("Tracker", (), {n: None for n in primitives[:3]})()
    refused = {
        "no_detector": sam(name="PlainModel"),
        "backbone_list": sam(detector([multiplex])),
        "no_vision_backbone": sam(detector({"language_backbone": None})),
        "module_dict_not_multiplex": sam(detector(torch.nn.ModuleDict({"vision_backbone": torch.nn.Identity()}))),
        "not_multiplex": sam(detector({"vision_backbone": single})),
        "tracker_missing_two": sam(detector({"vision_backbone": multiplex}), partial),
        "no_tracker": sam(detector({"vision_backbone": multiplex})),
    }
    for model in refused.values():
        with pytest.raises(ValueError):
            sam3._multiplex_parts(parts_model(model))
    whole = sam(detector({"vision_backbone": multiplex, "language_backbone": None}), complete)
    parts = sam3._multiplex_parts(parts_model(whole))
    assert parts[0] is whole and parts[1] is whole.detector and parts[2] is complete and parts[3] is multiplex
