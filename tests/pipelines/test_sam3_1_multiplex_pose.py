"""SAM 3.1 Multiplex box_keypoint mode's pure helpers - point sampling, the annexed-region memory,
anchors and recall, hand-placed points against derived ones - on synthetic masks and keypoints.
No model is loaded.

The pose module imports without ComfyUI, but this file imports comfy.cli_args first, so it runs
where ComfyUI is importable (the pod, with the ComfyUI root on PYTHONPATH) and is skipped
elsewhere."""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import C, kps20, sam3  # noqa: E402


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


# --- precedence: hand-placed points against derived ones -----------------------------------

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
