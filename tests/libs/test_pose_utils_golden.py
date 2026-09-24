"""G15: the vendored pose primitives, recorded from the production code (specs refactor plan
8.2): `bbox_from_detector`, `crop` (padding beyond the frame, and Python's round half to even
on the crop corners), `decode_heatmaps` (ViTPose's DARK decode, which warns about its own
arguments) and the dtype `load_pose_metas_from_kp2ds_seq` keeps.

`crop` resizes with cv2, so its digests hold for the cv2 build of the ComfyUI venv only; they
are in tests/goldens/test_pose_utils_golden.json:

    python -m pytest tests/libs/test_pose_utils_golden.py
"""
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("cv2")

from golden import check, digest  # noqa: E402
from pose_fakes import peaks  # noqa: E402

from bcvideonodes.models.vitpose.decode import decode_heatmaps  # noqa: E402
from bcvideonodes.libs.pose_utils import pose2d_utils  # noqa: E402

RESOLUTIONS = {"vitpose": (256, 192)}
H, W = 160, 120
BOXES = (
    (30.0, 20.0, 90.0, 140.0),                     # taller than the crop's 4:3
    (10.0, 60.0, 110.0, 100.0),                    # wider than it
    (12.3, 45.6, 78.9, 150.1),
    np.array([30.0, 20.0, 90.0, 140.0, 0.9]),      # with its score, as the pose code passes it
    (-20.0, -30.0, 60.0, 100.0),                   # past the left and the top edge
    (70.0, 90.0, 150.0, 200.0),                    # past the right and the bottom edge
)


def frame():
    return np.random.default_rng(15).random((H, W, 3), dtype=np.float32)


@pytest.mark.parametrize("model", list(RESOLUTIONS))
def test_bbox_from_detector_golden(model):
    found = [pose2d_utils.bbox_from_detector(box, RESOLUTIONS[model], rescale=1.25) for box in BOXES]
    check(__file__, f"bbox_from_detector.{model}", digest([(digest(c), digest(s)) for c, s in found]))


# Crop centres whose four crop corners land exactly on .5 at the scale that maps the crop 1:1
# onto the frame (found by search: most .5 centres miss by a float ulp), with the corners
# Python's round half to even makes of them: 4.5 -> 4 and 2.5 -> 2 down, 1.5 -> 2 and 3.5 -> 4 up.
HALF_CENTERS = {
    "vitpose": [((100.5, 130.5), ((4, 120), (2, 160))), ((97.5, 131.5), ((2, 120), (4, 160)))],
}


@pytest.mark.parametrize("model", list(RESOLUTIONS))
def test_crop_golden(model):
    resolution = RESOLUTIONS[model]
    image = frame()
    one_to_one = np.array([resolution[0] / 200.0 * 0.75, resolution[0] / 200.0])
    cases = [pose2d_utils.bbox_from_detector(box, resolution, rescale=1.25) for box in BOXES]
    cases += [(np.array(center), one_to_one) for center, _ in HALF_CENTERS[model]]
    found = []
    for center, scale in cases:
        new_img, new_shape, old, new = pose2d_utils.crop(image, center, scale, resolution)
        found.append((new_img, new_shape, old, new))
    assert [tuple(tuple(int(v) for v in axis) for axis in f[2]) for f in found[-2:]] == \
        [corners for _, corners in HALF_CENTERS[model]]
    check(__file__, f"crop.{model}", digest([digest(f[0]) for f in found]))
    check(__file__, f"crop.{model}.corners", repr([f[1:] for f in found]))


def test_decode_heatmaps_golden():
    heatmaps = peaks()
    center, scale = np.array([[60.0, 80.0]]), np.array([[0.6, 0.8]])
    with pytest.warns(DeprecationWarning):
        plain = decode_heatmaps(heatmaps, center, scale)
    check(__file__, "decode_heatmaps", digest(plain))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_load_pose_metas_from_kp2ds_seq_golden(dtype):
    rng = np.random.default_rng(17)
    kp2ds = (rng.random((3, 133, 3)) * np.array([W, H, 1.0])).astype(dtype)
    before = digest(kp2ds)
    metas = pose2d_utils.load_pose_metas_from_kp2ds_seq(kp2ds, width=W, height=H)
    assert digest(kp2ds) == before
    found = [[(key, digest(value) if isinstance(value, np.ndarray) else repr(value)) for key, value in meta.items()]
             for meta in metas]
    check(__file__, f"load_pose_metas_from_kp2ds_seq.{np.dtype(dtype).str}", digest(found))
