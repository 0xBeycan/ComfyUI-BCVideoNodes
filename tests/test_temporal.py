"""The temporal layer on synthetic sequences: a person walking across the frame, with one
fault injected at a time. Runs without ComfyUI or any model:

    python -m pytest tests/test_temporal.py
"""
import importlib.util
import os

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("temporal", os.path.join(ROOT, "preprocess", "temporal.py"))
temporal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(temporal)

N, K = 40, 133
BOX_W, BOX_H = 120.0, 300.0
DIAG = float(np.hypot(BOX_W, BOX_H))


def boxes(missing=()):
    """A person-sized box walking one pixel per frame to the right; `missing` frames come in
    the way the detector hands them over when it found nobody: the whole frame, score -1."""
    out = []
    for i in range(N):
        if i in missing:
            out.append(np.array([0.0, 0.0, 640.0, 960.0, -1.0]))
        else:
            out.append(np.array([100.0 + i, 50.0, 100.0 + i + BOX_W, 50.0 + BOX_H, 0.9]))
    return out


def keypoints(conf=0.9):
    """Every keypoint moving one pixel per frame with the box."""
    kp = np.zeros((N, K, 3))
    for i in range(N):
        kp[i, :, 0] = 110.0 + i + np.arange(K) * 0.1
        kp[i, :, 1] = 60.0 + np.arange(K) * 0.2
        kp[i, :, 2] = conf
    return kp


def test_an_undetected_frame_takes_the_box_between_its_neighbours():
    out = temporal.boxes_over_time(boxes(missing=(20,)))
    # the neighbours' boxes, widened over BOX_WINDOW frames, not the whole frame
    assert out[20][0] == pytest.approx(100.0 + 20 - temporal.BOX_WINDOW)
    assert out[20][2] == pytest.approx(100.0 + 20 + temporal.BOX_WINDOW + BOX_W)
    assert out[20][1] == pytest.approx(50.0) and out[20][3] == pytest.approx(50.0 + BOX_H)


def test_an_undetected_frame_stays_undetected():
    out = temporal.boxes_over_time(boxes(missing=(20, 21, 22)))
    assert [b[4] for b in out[19:24]] == [0.9, -1.0, -1.0, -1.0, 0.9]


def test_a_gap_at_the_start_takes_the_first_detection():
    out = temporal.boxes_over_time(boxes(missing=(0, 1)))
    # frames 0 and 1 take frame 2's box, and the window then widens to frame 4's
    assert out[0][0] == pytest.approx(102.0) and out[0][2] == pytest.approx(104.0 + BOX_W)


def test_a_clip_with_no_detection_at_all_is_left_alone():
    out = temporal.boxes_over_time(boxes(missing=range(N)))
    assert all(b[0] == 0.0 and b[2] == 640.0 and b[4] == -1.0 for b in out)


def test_a_clean_sequence_is_untouched():
    kp = keypoints()
    out, source = temporal.keypoints_over_time(kp, boxes())
    assert np.array_equal(out, kp) and (source == temporal.MEASURED).all()


def test_a_one_frame_drop_is_recovered_from_its_neighbours():
    kp = keypoints()
    kp[20, 7, 2] = 0.1
    out, source = temporal.keypoints_over_time(kp, boxes())
    assert source[20, 7] == temporal.RECOVERED
    assert out[20, 7, :2] == pytest.approx(kp[20, 7, :2])
    # no more certain than the weaker of the two measurements it sits between
    assert out[20, 7, 2] == pytest.approx(0.9)
    assert (np.delete(source, 20, axis=0) == temporal.MEASURED).all()


def test_a_gap_longer_than_max_gap_is_left_as_the_model_gave_it():
    kp = keypoints()
    dropped = range(10, 10 + temporal.MAX_GAP + 1)
    kp[dropped, 7, 2] = 0.1
    out, source = temporal.keypoints_over_time(kp, boxes())
    assert (source[dropped, 7] == temporal.MEASURED).all()
    assert out[dropped, 7, 2] == pytest.approx(0.1)


def test_a_gap_over_fast_motion_is_left_alone():
    """The anchors are far apart in space, so the line between them is a guess."""
    kp = keypoints()
    kp[21:, 7, 0] += temporal.MAX_STEP * DIAG * 10
    kp[20, 7, 2] = 0.1
    _, source = temporal.keypoints_over_time(kp, boxes())
    assert source[20, 7] == temporal.MEASURED


def test_a_confident_keypoint_far_from_its_neighbours_is_replaced():
    kp = keypoints()
    kp[20, 7, 0] += 3 * temporal.MAX_RESIDUAL * DIAG
    out, source = temporal.keypoints_over_time(kp, boxes())
    assert source[20, 7] == temporal.REPLACED
    assert out[20, 7, :2] == pytest.approx(keypoints()[20, 7, :2])
    assert (np.delete(source, 20, axis=0) == temporal.MEASURED).all()


def test_a_glitch_with_nothing_to_replace_it_is_dropped():
    """No anchor after the glitch, so there is no line to put in its place."""
    kp = keypoints()
    kp[21:, 7, 2] = 0.1
    kp[20, 7, 0] += 3 * temporal.MAX_RESIDUAL * DIAG
    out, source = temporal.keypoints_over_time(kp, boxes())
    assert source[20, 7] == temporal.DROPPED and out[20, 7, 2] == 0.0
    # the position is left where the model put it; the confidence is what says not to use it
    assert out[20, 7, 0] == pytest.approx(kp[20, 7, 0])


def test_a_run_of_wrong_frames_is_taken_out_together():
    """Three wrong frames still leave the median of seven on the right ones."""
    kp = keypoints()
    kp[20:23, 7, 0] += 3 * temporal.MAX_RESIDUAL * DIAG
    out, source = temporal.keypoints_over_time(kp, boxes())
    assert (source[20:23, 7] == temporal.REPLACED).all()
    assert out[20:23, 7, 0] == pytest.approx(keypoints()[20:23, 7, 0])


def test_motion_within_the_step_limit_is_not_a_glitch():
    kp = keypoints()
    kp[20:, 7, 0] += temporal.MAX_STEP * DIAG * 0.5
    _, source = temporal.keypoints_over_time(kp, boxes())
    assert (source[:, 7] == temporal.MEASURED).all()


def test_the_hands_are_treated_like_the_body():
    kp = keypoints()
    kp[20, 100, 2] = 0.1
    kp[25, 120, 0] += 3 * temporal.MAX_RESIDUAL * DIAG
    _, source = temporal.keypoints_over_time(kp, boxes())
    assert source[20, 100] == temporal.RECOVERED and source[25, 120] == temporal.REPLACED


def test_keypoints_are_measured_in_box_diagonals_not_pixels():
    """The same displacement is a glitch on a small person and motion on a large one."""
    kp = keypoints()
    kp[20, 7, 0] += 2 * temporal.MAX_RESIDUAL * DIAG
    small = boxes()
    large = [np.array([0.0, 0.0, 10 * BOX_W, 10 * BOX_H, 0.9]) for _ in range(N)]
    assert temporal.keypoints_over_time(kp, small)[1][20, 7] == temporal.REPLACED
    assert temporal.keypoints_over_time(kp, large)[1][20, 7] == temporal.MEASURED


def test_a_sequence_too_short_to_bracket_anything_is_returned_as_it_came():
    kp = keypoints()[:2]
    out, source = temporal.keypoints_over_time(kp, boxes()[:2])
    assert np.array_equal(out, kp) and (source == temporal.MEASURED).all()


def test_the_input_array_is_not_modified():
    kp = keypoints()
    kp[20, 7, 2] = 0.1
    before = kp.copy()
    temporal.keypoints_over_time(kp, boxes())
    assert np.array_equal(kp, before)


def test_the_limits_can_be_overridden_per_call():
    """The config passes its own gap, step, residual and window; the constants are defaults."""
    kp = keypoints()
    dropped = range(10, 10 + temporal.MAX_GAP + 1)
    kp[dropped, 7, 2] = 0.1
    _, source = temporal.keypoints_over_time(kp, boxes(), max_gap=temporal.MAX_GAP + 1)
    assert (source[dropped, 7] == temporal.RECOVERED).all()

    kp = keypoints()
    kp[20, 7, 0] += 3 * temporal.MAX_RESIDUAL * DIAG
    _, source = temporal.keypoints_over_time(kp, boxes(), max_residual=4 * temporal.MAX_RESIDUAL)
    assert source[20, 7] == temporal.MEASURED

    kp = keypoints()
    kp[21:, 7, 0] += temporal.MAX_STEP * DIAG * 10
    kp[20, 7, 2] = 0.1
    _, source = temporal.keypoints_over_time(kp, boxes(), max_step=temporal.MAX_STEP * 20)
    assert source[20, 7] == temporal.RECOVERED

    out = temporal.boxes_over_time(boxes(missing=(20,)), box_window=1)
    assert out[20][0] == pytest.approx(100.0 + 20 - 1)
