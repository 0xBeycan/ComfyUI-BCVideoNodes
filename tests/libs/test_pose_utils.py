"""The vendored pose primitives: `crop` rounds its corners half to even, as Python's round does,
and `load_pose_metas_from_kp2ds_seq` leaves its input alone.

    python -m pytest tests/libs/test_pose_utils.py
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from bcvideonodes.libs.pose_utils import pose2d_utils  # noqa: E402

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
