"""The shared BBOX reader Pose Detection and SAM3 both call, and the box widening over the
neighbouring frames. No ComfyUI and no model:

    python -m pytest tests/libs/test_bbox.py
"""
import json

import numpy as np
import pytest

from bcvideonodes.libs.bbox import parse_bboxes, widen_over_time


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


# six boxes on a 120x160 frame: inside, 5 px from the left edge, undetected, close to the top
# and the bottom, float32 and 4 px from the right edge, undetected again
TABLE = (
    np.array([30.0, 20.0, 90.0, 130.0, 0.9]),
    np.array([5.0, 20.0, 65.0, 130.0, 0.8]),
    np.array([0.0, 0.0, 120.0, 160.0, -1.0]),
    np.array([40.0, 3.0, 100.0, 150.0, 0.7]),
    np.array([50.0, 30.0, 116.0, 140.0, 0.95], dtype=np.float32),
    np.array([0.0, 0.0, 120.0, 160.0, -1.0]),
)


@pytest.mark.parametrize("box_window", [0, 1, 4])
def test_widen_over_time_hands_an_undetected_frame_back_as_it_came(box_window):
    table = list(TABLE)
    widened = widen_over_time(table, box_window)
    assert widened[2] is table[2] and widened[5] is table[5]


def test_widen_over_time_takes_the_union_of_the_detected_neighbours_and_keeps_the_score():
    widened = widen_over_time(list(TABLE), 1)
    # frame 1: frames 0-2, the undetected frame 2 not counted
    assert widened[1].tolist() == [5.0, 20.0, 90.0, 130.0, 0.8]
    # frame 3: frames 2-4, the undetected frame 2 not counted
    assert widened[3].tolist() == [40.0, 3.0, 116.0, 150.0, 0.7]
    # box_window 0 keeps every detected box as it was
    assert [b.tolist() for b in widen_over_time(list(TABLE), 0)] == [b.tolist() for b in TABLE]
