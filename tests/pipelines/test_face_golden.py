"""Golden digests of the face crop (G12, with its log lines, G18): the crops, boxes and log of
crop_faces over face keypoint sets that take every branch of the computed box (row 0 left out,
NaN and inf points dropped, the image centre when none remain, the minimum side on a collapsed
axis, the growth that reaches further up than down, int boxes, face_padding kept inside the
frame, keypoints off the frame), both fallbacks of an empty crop, supplied boxes with a bad one,
and the frame-count error. Recorded once from the production code before the refactor moved
anything; see tests/golden.py for the storage and the one way to re-record.

The crops are resized by cv2, so their digests hold for the cv2 build of the ComfyUI venv only.
Run with that venv:

    python -m pytest tests/pipelines/test_face_golden.py
"""
import logging

import cv2
import numpy as np
import pytest
import torch

from bcvideonodes.pipelines import face
from golden import check, digest, log_text

H, W = 200, 160


def frames(count, height=H, width=W, seed=0):
    return torch.from_numpy(np.random.default_rng(seed).random((count, height, width, 3), dtype=np.float32))


def cloud(cx, cy, spread, seed):
    """69 face keypoints (normalised x, y) scattered around (cx, cy)."""
    return np.array([cx, cy]) + np.random.default_rng(seed).normal(0.0, spread, (69, 2))


def with_row_0(points, row_0):
    points = points.copy()
    points[0] = row_0
    return points


def with_rows(points, rows, value):
    points = points.copy()
    points[rows] = value
    return points


# one frame each, in this order
KEYPOINTS = {
    "cloud": cloud(0.5, 0.3, 0.04, 1),
    "row_0_far": with_row_0(cloud(0.3, 0.4, 0.03, 2), (0.95, 0.95)),      # row 0 is not read
    "nan_and_inf": with_rows(with_rows(cloud(0.6, 0.5, 0.03, 3), slice(5, 21), np.nan), slice(30, 36), np.inf),
    "all_nan": with_rows(with_row_0(cloud(0.2, 0.2, 0.03, 4), (0.2, 0.2)), slice(1, None), np.nan),   # the centre
    "collapsed": np.tile([0.4, 0.6], (69, 1)),                             # both sides below the minimum
    "line": np.stack([np.linspace(0.3, 0.7, 69), np.full(69, 0.5)], axis=1),   # the height only
    "top_left": cloud(0.03, 0.03, 0.02, 5),
    "bottom_right": cloud(0.97, 0.97, 0.02, 6),
    "off_right": cloud(1.4, 0.3, 0.02, 7),                                 # an empty crop: the centre crop
    "off_left": cloud(-0.4, 0.3, 0.02, 8),
}
B = len(KEYPOINTS)


def pose_data(points_per_frame, height=H, width=W):
    """pose_data holding `points_per_frame` ([69, 2] normalised each) as the face keypoints."""
    return {"pose_metas_original": [
        {"width": width, "height": height,
         "keypoints_face": np.concatenate([np.asarray(points, dtype=np.float64), np.full((69, 1), 0.9)], axis=1)}
        for points in points_per_frame]}


POSE = pose_data(KEYPOINTS.values())
# floats, cut to ints: the first starts at -0.5 (0 once cut), the last runs past the frame
SUPPLIED = ([(-0.5, 20.7, 60.2, 90.9)] + [(10.0 + 7 * i, 15.5 + 9 * i, 70.9 + 7 * i, 99.2 + 9 * i) for i in range(1, B - 1)]
            + [(120.3, 150.8, 175.0, 230.0)])

CROPS = {
    "computed": lambda: face.crop_faces(frames(B), POSE),
    "padded": lambda: face.crop_faces(frames(B), POSE, face_padding=10),
    "tiny_2x2": lambda: face.crop_faces(frames(1, 2, 2), pose_data([cloud(0.5, 0.5, 0.1, 9)], 2, 2)),
    "supplied": lambda: face.crop_faces(frames(B), POSE, face_padding=10, face_bboxes=SUPPLIED),
    "supplied_single": lambda: face.crop_faces(frames(B), POSE, face_bboxes=[(10.6, 20.4, 60.5, 90.5)]),
}
ERRORS = {
    "supplied_bad": lambda: face.crop_faces(frames(B), POSE, face_bboxes=[(-5.0, 20.0, 60.0, 90.0), (10.2, 20.0, 10.9, 90.0)]
                                            + [(10.0, 20.0, 60.0, 90.0)] * (B - 2)),
    "frame_count": lambda: face.crop_faces(frames(2), POSE),
}


def capture(caplog):
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="BCVideoNodes")


def logged(caplog):
    return log_text(r for r in caplog.records if r.name in ("BCVideoNodes", "root"))


@pytest.mark.parametrize("name", CROPS)
def test_face_crops(name, caplog):
    capture(caplog)
    crops, boxes = CROPS[name]()
    check(__file__, f"{name}.crops", digest(crops))
    check(__file__, f"{name}.boxes", repr(boxes))
    check(__file__, f"{name}.log", digest(logged(caplog)))


@pytest.mark.parametrize("name", ERRORS)
def test_face_errors(name, caplog):
    capture(caplog)
    with pytest.raises(ValueError) as failure:
        ERRORS[name]()
    check(__file__, f"{name}.error", str(failure.value))
    check(__file__, f"{name}.log", digest(logged(caplog)))


def test_the_second_fallback_on_a_tiny_frame(caplog):
    # the face box lies off a 2x2 frame, so the crop is empty; the centre crop is empty too
    # (its side is int(0.3 * 2) = 0), and the zeros that stand in for it are 0x0, which cv2.resize refuses
    capture(caplog)
    with pytest.raises(cv2.error) as failure:
        face.crop_faces(frames(1, 2, 2), pose_data([cloud(10.0, 0.5, 0.1, 10)], 2, 2))
    # what precedes " error: " is the OpenCV version and its build path
    check(__file__, "tiny_2x2_fallback.error", str(failure.value).split(" error: ", 1)[1].strip())
    check(__file__, "tiny_2x2_fallback.log", digest(logged(caplog)))
