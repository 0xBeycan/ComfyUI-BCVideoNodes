"""The SAM3 module's pure helpers - point sampling, hole capping, island handling, the
annexed-region memory, input parsing - on synthetic masks and keypoints. No model is loaded.

The module imports ComfyUI at its top, so this runs where ComfyUI is importable (the pod, with
the ComfyUI root on PYTHONPATH) and is skipped elsewhere."""
import dataclasses
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

sam3 = pytest.importorskip("preprocess.sam3")
C = sam3.SAM3Config()


def kps20(conf=0.0):
    """A 20-point body layout, every keypoint at the centre with confidence `conf`."""
    return np.array([[0.5, 0.5, conf]] * 20, dtype=np.float64)


# --- hole capping --------------------------------------------------------------------------

def test_fill_holes_fills_a_small_enclosed_hole():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 1          # 6400 px, 1% is 64
    mask[40:45, 40:45] = 0          # 25 px hole
    assert sam3.fill_holes(mask, C.max_hole_fraction)[40:45, 40:45].all()


def test_fill_holes_leaves_a_hole_above_the_cap():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 1
    mask[30:40, 30:40] = 0          # 100 px against 6300 px of mask: 1.6%, a limb gap
    out = sam3.fill_holes(mask, C.max_hole_fraction)
    assert not out[30:40, 30:40].any()


def test_fill_holes_cap_is_a_fraction_of_the_mask_itself():
    mask = np.zeros((100, 100), np.uint8)
    mask[10:90, 10:90] = 1
    mask[40:48, 40:48] = 0          # 64 px against 6336 px: 1.01% - just over
    assert not sam3.fill_holes(mask, C.max_hole_fraction)[40:48, 40:48].any()
    mask[40:48, 40:47] = 1          # now 8 px against 6392 px
    assert sam3.fill_holes(mask, C.max_hole_fraction)[40:48, 47].all()


def test_fill_holes_ignores_background_touching_the_border():
    mask = np.zeros((100, 100), np.uint8)
    mask[0:50, 0:50] = 1
    mask[0:3, 20:23] = 0            # a notch open to the top edge is not enclosed
    out = sam3.fill_holes(mask, C.max_hole_fraction)
    assert not out[0:3, 20:23].any()


def test_fill_holes_without_holes_is_unchanged():
    mask = np.zeros((50, 50), np.uint8)
    mask[10:40, 10:40] = 1
    assert np.array_equal(sam3.fill_holes(mask, C.max_hole_fraction), mask)


# --- island handling -----------------------------------------------------------------------

def test_drop_islands_removes_specks_and_keeps_real_pieces():
    mask = np.zeros((200, 200), np.uint8)
    mask[20:120, 20:120] = 1        # 10000 px body
    mask[150:153, 150:153] = 1      # 9 px speck: under 1%
    mask[150:162, 20:32] = 1        # 144 px piece: over 1%
    out = sam3.drop_islands(mask, C.min_island_fraction)
    assert out[20:120, 20:120].all()
    assert not out[150:153, 150:153].any()
    assert out[150:162, 20:32].all()


def test_drop_islands_single_region_is_returned_as_is():
    mask = np.zeros((50, 50), np.uint8)
    mask[5:10, 5:10] = 1
    assert sam3.drop_islands(mask, C.min_island_fraction) is mask


def test_clean_mask_fills_then_drops():
    mask = np.zeros((200, 200), bool)
    mask[20:120, 20:120] = True
    mask[60:62, 60:62] = False      # pinhole
    mask[180:182, 180:182] = True   # speck
    out = sam3.clean_mask(mask, C)
    assert out.dtype == np.uint8
    assert out[60:62, 60:62].all() and not out[180:182, 180:182].any()


# --- point sampling ------------------------------------------------------------------------

def test_box_bounds_clips_and_rejects_slivers():
    assert sam3.box_bounds([-10, -5, 50, 60, 1.0], 40, 40) == (0, 0, 40, 40)
    assert sam3.box_bounds([0, 0, 7, 50, 1.0], 100, 100) is None


def test_spread_points_one_per_cell_inside_the_map():
    free = np.zeros((90, 90), bool)
    free[:, :30] = True             # only the left third is free
    points = sam3.spread_points(free, 100, 200, 9)
    assert len(points) == 3         # one per row of the 3x3 grid, the other cells are empty
    for x, y in points:
        assert free[y - 200, x - 100]


def test_spread_points_caps_the_count_and_handles_empty():
    assert sam3.spread_points(np.zeros((10, 10), bool), 0, 0, 8) == []
    assert len(sam3.spread_points(np.ones((90, 90), bool), 0, 0, 8)) == 8


def test_background_points_stay_clear_of_the_previous_mask():
    W = H = 200
    previous = np.zeros((H, W), bool)
    previous[50:150, 80:120] = True
    bbox = [40, 40, 160, 160, 1.0]
    points = sam3.background_points(previous, bbox, W, H, C.negative_points, C.negative_margin)
    assert 0 < len(points) <= C.negative_points
    margin = int(C.negative_margin * np.hypot(120, 120))
    ys, xs = np.nonzero(previous)
    for x, y in points:
        assert 40 <= x < 160 and 40 <= y < 160
        # the distance transform is cv2's 3x3 approximation of L2, so allow it a pixel
        assert np.min(np.hypot(xs - x, ys - y)) > margin - 1


def test_background_points_need_a_previous_mask_inside_the_box():
    previous = np.zeros((100, 100), bool)
    assert sam3.background_points(None, [0, 0, 100, 100, 1.0], 100, 100, 8, 0.04) == []
    assert sam3.background_points(previous, [0, 0, 100, 100, 1.0], 100, 100, 8, 0.04) == []
    previous[0:10, 0:10] = True
    assert sam3.background_points(previous, [50, 50, 100, 100, 1.0], 100, 100, 8, 0.04) == []


def test_annexed_points_land_on_the_annexed_map_inside_the_box():
    annexed = np.zeros((100, 100), bool)
    annexed[10:30, 60:90] = True
    points = sam3.annexed_points(annexed, [0, 0, 100, 100, 1.0], 100, 100, C.annexed_points)
    assert 0 < len(points) <= C.annexed_points
    assert all(annexed[y, x] for x, y in points)
    assert sam3.annexed_points(None, [0, 0, 100, 100, 1.0], 100, 100, 8) == []


def test_body_points_torso_and_limbs():
    kps = kps20()
    kps[sam3.R_SHOULDER] = [0.4, 0.2, 0.9]
    kps[sam3.L_SHOULDER] = [0.6, 0.2, 0.9]
    kps[sam3.R_HIP] = [0.4, 0.6, 0.9]
    kps[sam3.L_HIP] = [0.6, 0.6, 0.9]
    kps[9] = [0.4, 0.8, 0.9]        # right knee: thigh R_HIP-9 is confident at both ends
    points = sam3.body_points(kps, 0.3)
    assert np.allclose(points, [(0.5, 0.2 + 0.35 * 0.4), (0.5, 0.2 + 0.65 * 0.4), (0.4, 0.7)])


def test_body_points_steps_down_by_shoulder_width_without_hips():
    kps = kps20()
    kps[sam3.R_SHOULDER] = [0.4, 0.2, 0.9]
    kps[sam3.L_SHOULDER] = [0.6, 0.2, 0.9]
    points = sam3.body_points(kps, 0.3, aspect=0.5)
    # the x span 0.2 is scaled by W / H before it is stepped along y
    assert np.allclose(points, [(0.5, 0.3), (0.5, 0.4)])


def test_body_points_foot_limbs():
    kps = kps20()
    kps[sam3.R_ANKLE] = [0.4, 0.9, 0.9]
    kps[sam3.R_FOOT] = [0.44, 0.96, 0.9]
    assert np.allclose(sam3.body_points(kps, 0.3), [(0.42, 0.93)])


def test_prompt_for_labels_box_and_hand_placed_points():
    W, H = 200, 100
    kps = kps20()
    kps[0] = [0.5, 0.2, 0.9]        # nose
    kps[1] = [0.5, 0.3, 0.2]        # neck, not confident
    bboxes = [np.array([20.0, 10.0, 180.0, 90.0, 0.8])]
    box, points = sam3.prompt_for(0, bboxes, [{"keypoints_body": kps}], W, H, "cpu", torch.float32, C, 0.3,
                                  extra_positive=[(10.0, 10.0)], extra_negative=[(190.0, 5.0)])
    s = sam3.SAM3_SIZE
    assert np.allclose(box.numpy(), [[[20 * s / W, 10 * s / H], [180 * s / W, 90 * s / H]]])
    assert points["point_labels"].tolist() == [[1, 1, 0]]
    assert np.allclose(points["point_coords"][0].numpy(),
                       [[100 * s / W, 20 * s / H], [10 * s / W, 10 * s / H], [190 * s / W, 5 * s / H]])


def test_prompt_for_undetected_frame_has_no_box_and_no_negatives():
    kps = kps20()
    kps[0] = [0.5, 0.2, 0.9]
    previous = np.ones((100, 100), bool)
    box, points = sam3.prompt_for(0, [np.array([0.0, 0.0, 100.0, 100.0, -1.0])], [{"keypoints_body": kps}],
                                  100, 100, "cpu", torch.float32, C, 0.3, previous_mask=previous)
    assert box is None
    assert points["point_labels"].tolist() == [[1]]


def test_prompt_for_without_confident_keypoints_has_no_points():
    box, points = sam3.prompt_for(0, [np.array([0.0, 0.0, 100.0, 100.0, 1.0])], [{"keypoints_body": kps20()}],
                                  100, 100, "cpu", torch.float32, C, 0.3)
    assert box is not None and points is None


# --- the annexed-region memory -------------------------------------------------------------

ANNEX = (C.min_annexed_fraction, 0.3)


def body_and_object():
    """A 100x100 frame: previous mask is the body, this frame's mask also took in an object."""
    previous = np.zeros((100, 100), bool)
    previous[20:80, 20:50] = True               # 1800 px body
    mask = previous.copy()
    mask[20:40, 60:80] = True                   # 400 px gained, detached, >3% of 2200
    return previous, mask


def test_remember_annexed_adds_a_large_gain_without_keypoints():
    previous, mask = body_and_object()
    kps = kps20()
    kps[0] = [0.35, 0.5, 0.9]                   # on the body
    annexed = sam3.remember_annexed(None, mask, previous, kps, 100, 100, *ANNEX)
    assert annexed is not None and annexed[20:40, 60:80].all() and annexed.sum() == 400


def test_remember_annexed_skips_a_gain_holding_a_keypoint():
    previous, mask = body_and_object()
    kps = kps20()
    kps[0] = [0.70, 0.30, 0.9]                  # a confident keypoint in the gained region
    assert sam3.remember_annexed(None, mask, previous, kps, 100, 100, *ANNEX) is None


def test_remember_annexed_skips_a_small_gain():
    previous = np.zeros((100, 100), bool)
    previous[20:80, 20:50] = True
    mask = previous.copy()
    mask[20:25, 60:65] = True                   # 25 px: under 3% of the mask
    assert sam3.remember_annexed(None, mask, previous, kps20(), 100, 100, *ANNEX) is None


def test_remember_annexed_forgets_a_region_the_person_moved_onto():
    previous, mask = body_and_object()
    annexed = sam3.remember_annexed(None, mask, previous, kps20(), 100, 100, *ANNEX)
    assert annexed is not None
    kps = kps20()
    kps[0] = [0.62, 0.22, 0.9]                  # now a keypoint inside the remembered region
    assert sam3.remember_annexed(annexed, previous, previous, kps, 100, 100, *ANNEX) is None


def test_remember_annexed_keeps_the_map_across_frames():
    previous, mask = body_and_object()
    annexed = sam3.remember_annexed(None, mask, previous, kps20(), 100, 100, *ANNEX)
    # the next frame gains nothing; the map is carried, not recomputed from the last frame
    kept = sam3.remember_annexed(annexed, previous, previous, kps20(), 100, 100, *ANNEX)
    assert kept is not None and np.array_equal(kept, annexed)


# --- anchors and recall --------------------------------------------------------------------

def test_keypoint_recall_counts_confident_keypoints_only():
    mask = np.zeros((100, 100), bool)
    mask[:, :50] = True
    kps = kps20()
    assert sam3.keypoint_recall(mask, kps, 100, 100, 0.3) == 1.0
    kps[0] = [0.2, 0.5, 0.9]
    kps[1] = [0.8, 0.5, 0.9]
    kps[2] = [0.8, 0.5, 0.1]
    assert sam3.keypoint_recall(mask, kps, 100, 100, 0.3) == 0.5


def test_is_anchor():
    kps = kps20(0.6)
    metas = [{"keypoints_body": kps}]
    assert sam3.is_anchor(0, [np.array([0, 0, 1, 1, 0.9])], metas, C, 0.3)
    assert not sam3.is_anchor(0, [np.array([0, 0, 1, 1, -1.0])], metas, C, 0.3)
    # 20 confident keypoints, but the segment was seeded from more than 20 / 0.9 of them
    assert not sam3.is_anchor(0, [np.array([0, 0, 1, 1, 0.9])], metas, C, 0.3, reference_count=23)
    low = kps20(0.4)
    assert not sam3.is_anchor(0, [np.array([0, 0, 1, 1, 0.9])], [{"keypoints_body": low}], C, 0.3)


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


# --- precedence: what a mode ignores, and hand-placed points against derived ones ----------

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


def test_a_hand_placed_negative_drops_the_derived_positive_under_it(caplog):
    W, H = 200, 100
    kps = kps20()
    kps[0] = [0.5, 0.2, 0.9]        # nose at (100, 20)
    kps[2] = [0.3, 0.4, 0.9]        # right shoulder at (60, 40)
    bboxes = [np.array([20.0, 10.0, 180.0, 90.0, 0.8])]    # diagonal 179: reach 7.2 px
    with caplog.at_level("INFO"):
        _, points = sam3.prompt_for(0, bboxes, [{"keypoints_body": kps}], W, H, "cpu", torch.float32, C, 0.3,
                                    extra_negative=[(103.0, 22.0)])
    s = sam3.SAM3_SIZE
    assert points["point_labels"].tolist() == [[1, 0]]
    assert np.allclose(points["point_coords"][0].numpy(), [[60 * s / W, 40 * s / H], [103 * s / W, 22 * s / H]])
    assert "1 derived positive and 0 derived negative point(s)" in caplog.text


def test_a_hand_placed_positive_drops_the_derived_negative_under_it():
    positive, negative = sam3._clear_of_hand_points([(10.0, 10.0)], [(50.0, 50.0), (90.0, 90.0)],
                                                    [(52.0, 51.0)], [], (0, 0, 100, 100))
    assert positive == [(10.0, 10.0)] and negative == [(90.0, 90.0)]


def test_hand_placed_points_away_from_the_derived_ones_change_nothing_else():
    W, H = 200, 100
    kps = kps20()
    kps[0] = [0.5, 0.2, 0.9]
    kps[2] = [0.3, 0.4, 0.9]
    args = (0, [np.array([20.0, 10.0, 180.0, 90.0, 0.8])], [{"keypoints_body": kps}], W, H, "cpu", torch.float32, C, 0.3)
    _, plain = sam3.prompt_for(*args)
    _, extra = sam3.prompt_for(*args, extra_positive=[(150.0, 80.0)], extra_negative=[(190.0, 5.0)])
    n = plain["point_labels"].shape[1]
    assert torch.equal(extra["point_coords"][:, :n], plain["point_coords"])
    assert extra["point_labels"].tolist() == [plain["point_labels"][0].tolist() + [1, 0]]


# --- config --------------------------------------------------------------------------------

def test_config_prompt_defaults_are_the_measured_values():
    c = sam3.SAM3Config()
    assert (c.birth_threshold, c.detection_threshold, c.nms_iou, c.match_iou) == (0.50, 0.30, 0.10, 0.50)
    assert (c.hotstart_frames, c.hotstart_unmatched) == (15, 8)
    assert (c.recondition_every, c.recondition_score, c.recondition_iou) == (16, 0.80, 0.80)
    assert (c.fill_hole_area, c.memory_gap) == (16, 7)


def test_config_box_keypoint_defaults_are_the_source_values():
    c = sam3.SAM3Config()
    assert not hasattr(c, "min_keypoint_conf")   # it is the pose's, read from pose_data
    assert (c.reseed_interval, c.max_propagate) == (24, 72)
    assert (c.min_anchor_keypoints, c.min_anchor_conf, c.min_anchor_completeness) == (8, 0.5, 0.9)
    assert (c.min_tracked_recall, c.negative_points, c.negative_margin) == (0.9, 8, 0.04)
    assert (c.min_annexed_fraction, c.annexed_points, c.mask_threshold) == (0.03, 8, -1.0)
    assert (c.min_island_fraction, c.max_hole_fraction, c.refine, c.temporal) == (0.01, 0.01, True, True)


def test_config_multi_object_defaults_are_their_own_fields():
    c = sam3.SAM3Config()
    assert (c.new_object_threshold, c.assoc_iou, c.duplicate_frames, c.occlusion_iou, c.shrink_keep) == \
        (0.50, 0.10, 8, 0.70, 0.30)


def test_config_fields_carry_a_mode_tagged_tooltip_and_a_range():
    tags = {}
    for f in dataclasses.fields(sam3.SAM3Config):
        tip = f.metadata["tooltip"]
        tag = tip[:tip.index("]") + 1]
        assert tag in ("[prompt]", "[prompt, max_objects > 1]", "[box_keypoint]"), f.name
        tags.setdefault(tag, []).append(f.name)
        if not isinstance(f.default, bool):
            assert f.metadata["min"] <= f.default <= f.metadata["max"], f.name
    assert tags["[prompt]"] == ["birth_threshold", "detection_threshold", "nms_iou", "match_iou", "hotstart_frames",
                                "hotstart_unmatched", "recondition_every", "recondition_score", "recondition_iou",
                                "fill_hole_area", "memory_gap"]
    assert tags["[prompt, max_objects > 1]"] == ["new_object_threshold", "assoc_iou", "duplicate_frames",
                                                 "occlusion_iou", "shrink_keep"]


# --- multi-object helpers ------------------------------------------------------------------

def logits(*boxes, size=20):
    """[K, size, size] mask logits, +5 inside each (y1, y2, x1, x2) box, -5 elsewhere."""
    out = torch.full((len(boxes), size, size), -5.0)
    for k, (y1, y2, x1, x2) in enumerate(boxes):
        out[k, y1:y2, x1:x2] = 5.0
    return out


def test_suppress_recently_occluded_blanks_the_later_occluded_of_an_overlapping_pair():
    masks = logits((0, 10, 0, 10), (0, 10, 0, 9), (12, 20, 12, 20))
    # tracks 0 and 1 overlap by IoU 0.9; 1 came out of occlusion more recently than 0
    got = sam3.suppress_recently_occluded(masks, torch.tensor([3, 7, -1]), 0.7)
    assert got.tolist() == [False, True, False]
    # never-occluded tracks are never suppressed by each other
    assert not sam3.suppress_recently_occluded(masks, torch.tensor([-1, -1, -1]), 0.7).any()
    # below the IoU threshold nothing happens
    assert not sam3.suppress_recently_occluded(masks, torch.tensor([3, 7, -1]), 0.95).any()
    assert sam3.suppress_recently_occluded(masks[:1], torch.tensor([5]), 0.7).tolist() == [False]


def test_suppress_shrunk_blanks_a_track_buried_under_another():
    masks = logits((0, 20, 0, 20), (5, 10, 5, 10), (0, 4, 0, 4))
    masks[0, 5:10, 5:10] = 9.0    # track 0 outscores track 1 everywhere track 1 is
    masks[2] = masks[2] * 2       # track 2 outscores track 0 on its own patch
    out = sam3.suppress_shrunk(masks, 0.3)
    assert (out[0] > 0).sum() == 400 and (out[2] > 0).sum() == 16
    assert not (out[1] > 0).any()
    single = logits((0, 5, 0, 5))
    assert torch.equal(sam3.suppress_shrunk(single, 0.3), single)


def test_non_overlapping_gives_shared_pixels_to_the_higher_score():
    binary = logits((0, 10, 0, 10), (5, 15, 5, 15)) > 0
    out = sam3.non_overlapping(binary, torch.tensor([0.6, 0.9]))
    assert not (out[0] & out[1]).any()
    assert out[1, 5:10, 5:10].all() and not out[0, 5:10, 5:10].any()
    assert (out[0] | out[1]).sum() == (binary[0] | binary[1]).sum()
    tie = sam3.non_overlapping(binary, torch.tensor([0.5, 0.5]))
    assert tie[0, 5:10, 5:10].all()   # a tie goes to the earlier-born object
