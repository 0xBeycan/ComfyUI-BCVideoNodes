"""The shared BBOX reader Pose Detection and SAM3 both call. No ComfyUI and no model:

    python -m pytest tests/libs/test_bbox.py
"""
import json

import numpy as np
import pytest

from bcvideonodes.libs.bbox import parse_bboxes


def test_one_flat_box_is_used_on_every_frame():
    assert parse_bboxes([10, 20, 30, 40], 3) == [(10.0, 20.0, 30.0, 40.0)] * 3


def test_one_box_in_a_list_is_used_on_every_frame():
    assert parse_bboxes([(10, 20, 30, 40)], 2) == [(10.0, 20.0, 30.0, 40.0)] * 2


def test_one_box_per_frame():
    assert parse_bboxes([(0, 0, 5, 5), (1, 1, 6, 6)], 2) == [(0.0, 0.0, 5.0, 5.0), (1.0, 1.0, 6.0, 6.0)]


def test_a_score_after_the_corners_is_not_read():
    assert parse_bboxes([(10, 20, 30, 40, 0.5)], 1) == [(10.0, 20.0, 30.0, 40.0)]
    assert parse_bboxes([10, 20, 30, 40, 0.5], 1) == [(10.0, 20.0, 30.0, 40.0)]


def test_a_json_string():
    assert parse_bboxes(json.dumps([[0, 0, 5, 5], [1, 1, 6, 6]]), 2)[1] == (1.0, 1.0, 6.0, 6.0)


def test_kjnodes_dicts_are_put_in_corner_order():
    kj = json.dumps([{"startX": 30, "startY": 40, "endX": 10, "endY": 20}])
    assert parse_bboxes(kj, 2) == [(10.0, 20.0, 30.0, 40.0)] * 2
    assert parse_bboxes({"startX": 1, "startY": 2, "endX": 3, "endY": 4}, 1) == [(1.0, 2.0, 3.0, 4.0)]


def test_numpy_boxes():
    boxes = np.array([[0, 0, 5, 5], [1, 1, 6, 6]], dtype=np.float32)
    assert parse_bboxes(boxes, 2) == [(0.0, 0.0, 5.0, 5.0), (1.0, 1.0, 6.0, 6.0)]
    assert parse_bboxes([np.array([0.0, 0.0, 5.0, 5.0])], 1) == [(0.0, 0.0, 5.0, 5.0)]


def test_the_wrong_number_of_boxes_raises():
    with pytest.raises(ValueError, match="3 boxes for 12 frames"):
        parse_bboxes([(0, 0, 10, 10)] * 3, 12)


def test_an_inverted_or_empty_box_raises():
    with pytest.raises(ValueError, match="x1 < x2"):
        parse_bboxes([(90, 20, 30, 140)], 1)
    with pytest.raises(ValueError, match="x1 < x2"):
        parse_bboxes([5, 5, 5, 10], 1)


def test_too_few_numbers_raise():
    with pytest.raises(ValueError, match=r"must hold \(x1, y1, x2, y2\) boxes"):
        parse_bboxes([(0, 0, 5)], 1)


def test_not_json_raises():
    with pytest.raises(ValueError, match="JSON string"):
        parse_bboxes("not json", 1)


def test_an_empty_list_or_a_non_box_raises():
    with pytest.raises(ValueError):
        parse_bboxes([], 1)
    with pytest.raises(ValueError):
        parse_bboxes(7, 1)
    with pytest.raises(ValueError):
        parse_bboxes({"x": 1}, 1)
