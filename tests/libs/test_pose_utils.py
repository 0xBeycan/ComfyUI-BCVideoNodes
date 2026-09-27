"""The vendored pose primitives: `crop` rounds its corners half to even, as Python's round does,
`load_pose_metas_from_kp2ds_seq` leaves its input alone, and the drawing's draw_head off and
draw_body off hide their parts at every draw threshold, 0 included, and at any threshold above 0
draw what Wan's rule (the hidden confidences set to 0) draws.

    python -m pytest tests/libs/test_pose_utils.py
"""
import copy

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from bcvideonodes.libs.pose_utils import human_visualization, pose2d_utils  # noqa: E402

RESOLUTIONS = {"vitpose": (256, 192)}
H, W = 160, 120

# Crop centres whose four crop corners land exactly on .5 at the scale that maps the crop 1:1
# onto the frame (found by search: most .5 centres miss by a float ulp), with the corners
# Python's round half to even makes of them: 4.5 -> 4 and 2.5 -> 2 down, 1.5 -> 2 and 3.5 -> 4 up.
HALF_CENTERS = {
    "vitpose": [((100.5, 130.5), ((4, 120), (2, 160))), ((97.5, 131.5), ((2, 120), (4, 160)))],
}


@pytest.mark.parametrize("model", list(RESOLUTIONS))
def test_crop_rounds_its_corners_half_to_even(model):
    resolution = RESOLUTIONS[model]
    image = np.random.default_rng(15).random((H, W, 3), dtype=np.float32)
    one_to_one = np.array([resolution[0] / 200.0 * 0.75, resolution[0] / 200.0])
    corners = []
    for center, _ in HALF_CENTERS[model]:
        _, _, old, _ = pose2d_utils.crop(image, np.array(center), one_to_one, resolution)
        corners.append(tuple(tuple(int(v) for v in axis) for axis in old))
    assert corners == [expected for _, expected in HALF_CENTERS[model]]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_load_pose_metas_from_kp2ds_seq_leaves_its_input_alone(dtype):
    rng = np.random.default_rng(17)
    kp2ds = (rng.random((3, 133, 3)) * np.array([W, H, 1.0])).astype(dtype)
    before = kp2ds.copy()
    pose2d_utils.load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
    assert np.array_equal(kp2ds, before) and kp2ds.dtype == before.dtype


# -- draw_head and draw_body -------------------------------------------------------------------------

FRAME_H, FRAME_W = 1280, 720          # 2 px body sticks, 1 px hand sticks at -1
HEAD = [0, 14, 15, 16, 17]            # the AAPose body keypoints draw_head leaves out: nose, eyes, ears
BODY = list(range(20))


def pose_meta(seed):
    """One frame's AAPoseMeta as detect makes it: 133 wholebody keypoints inside the frame, with
    confidences uniform in 0..1."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(10, [FRAME_W - 10, FRAME_H - 10], (133, 2))
    kp2ds = np.concatenate([xy, rng.random((133, 1))], axis=1).astype(np.float32)[None]
    meta = pose2d_utils.load_pose_metas_from_kp2ds_seq(kp2ds, width=FRAME_W, height=FRAME_H)[0]
    return pose2d_utils.AAPoseMeta.from_humanapi_meta(meta)


def with_confidence(meta, body=(), value=0.0, hands=False):
    """A copy of `meta` whose body keypoints `body`, and both hands when `hands`, have confidence
    `value`."""
    meta = copy.copy(meta)
    meta.kps_body_p = meta.kps_body_p.copy()
    meta.kps_body_p[list(body)] = value
    if hands:
        meta.kps_lhand_p = np.full_like(meta.kps_lhand_p, value)
        meta.kps_rhand_p = np.full_like(meta.kps_rhand_p, value)
    return meta


def draw(meta, threshold, **switches):
    """The pose image of `meta`, drawn as pipelines/pose.draw calls the vendored drawing."""
    options = dict(draw_body=True, draw_hand=True, draw_head=True, body_stick_width=-1, hand_stick_width=-1)
    canvas = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)
    return human_visualization.draw_aapose_by_meta_new(canvas, meta, threshold=threshold, **{**options, **switches})


def test_at_draw_threshold_0_draw_head_off_draws_no_head():
    head = with_confidence(pose_meta(3), [i for i in BODY if i not in HEAD], -np.inf, hands=True)
    assert draw(head, 0.0).any()
    assert not draw(head, 0.0, draw_head=False).any()
    # the whole pose drawn without its head is the pose whose head keypoints are not there at all
    meta = pose_meta(3)
    assert np.array_equal(draw(meta, 0.0, draw_head=False), draw(with_confidence(meta, HEAD, -np.inf), 0.0))


def test_at_draw_threshold_0_a_0_body_width_draws_no_body():
    body = with_confidence(pose_meta(4), hands=True, value=-np.inf)
    assert draw(body, 0.0, body_stick_width=0).any()
    assert not draw(body, 0.0, draw_body=False, body_stick_width=0).any()


@pytest.mark.parametrize("threshold", [0.05, 0.3, 0.5, 0.9])
@pytest.mark.parametrize("seed", [5, 6])
def test_above_threshold_0_the_images_are_wans(threshold, seed):
    meta = pose_meta(seed)
    assert np.array_equal(draw(meta, threshold, draw_head=False), draw(with_confidence(meta, HEAD), threshold))
    assert np.array_equal(draw(meta, threshold, draw_body=False, body_stick_width=0),
                          draw(with_confidence(meta, BODY), threshold, body_stick_width=0))
    assert np.array_equal(draw(meta, threshold, draw_body=False, draw_head=False, body_stick_width=0),
                          draw(with_confidence(meta, BODY), threshold, body_stick_width=0))
    if threshold < 0.5:
        # the switches hide something here
        assert not np.array_equal(draw(meta, threshold, draw_head=False), draw(meta, threshold))
        assert not np.array_equal(draw(meta, threshold, draw_body=False, body_stick_width=0),
                                  draw(meta, threshold, body_stick_width=0))
