"""The draw rule (libs/draw_rules.py) on hand-built keypoints: which frames forearm_rule fires on at
which forearm_limit, which side it names and what it leaves out, with the geometry written out in
pixels:

    python -m pytest tests/libs/test_draw_rules.py
"""
import numpy as np
import pytest

from pose_fakes import rules  # noqa: E402

W, H = 200, 400
# (elbow, wrist) of each arm in the AAPose layout
RIGHT, LEFT = (3, 4), (6, 7)


def meta(body=None):
    """One pose meta as pose_data's pose_metas_original holds it, from pixel rows: `body` a dict of
    AAPose index -> (x, y, conf); anything not given sits at (0, 0) with confidence 0."""
    rows = np.zeros((20, 3), np.float32)
    for j, (x, y, conf) in (body or {}).items():
        rows[j] = (x / W, y / H, conf)
    return {"width": W, "height": H, "keypoints_body": rows, "keypoints_left_hand": np.zeros((21, 3), np.float32),
            "keypoints_right_hand": np.zeros((21, 3), np.float32), "keypoints_face": np.zeros((69, 3), np.float32)}


def forearm_clip(lengths, wrist_conf=0.9, arms=(RIGHT,)):
    """One frame per forearm length: each arm of `arms` with its elbow at (100, 100) and its wrist
    straight below it."""
    def frame(length):
        body = {}
        for elbow, wrist in arms:
            body[elbow] = (100.0, 100.0, 0.9)
            body[wrist] = (100.0, 100.0 + length, wrist_conf)
        return meta(body)

    return [frame(length) for length in lengths]


def hidden(clip, limit):
    return rules.hidden_parts(len(clip), rules.overlong_forearms(clip, 0.5, limit))


def test_a_forearm_over_the_limit_times_its_median_drops_its_wrist():
    # median 50 px: at limit 2.0, 100 px is not over twice it, 101 px is (2.02 x)
    clip = forearm_clip([50.0] * 7 + [100.0, 101.0])
    overlong = rules.overlong_forearms(clip, 0.5, 2.0)
    assert list(overlong) == ["right"] and overlong["right"]["median"] == pytest.approx(50.0, abs=1e-4)
    assert list(overlong["right"]["ratios"]) == [8] and overlong["right"]["ratios"][8] == pytest.approx(2.02, abs=1e-5)


def test_a_larger_limit_hides_fewer_wrists():
    # median 50 px: 101, 160 and 260 px are over 2.0 x 50, 160 and 260 over 3.0 x 50, 260 over 5.0 x 50
    clip = forearm_clip([50.0] * 7 + [101.0, 160.0, 260.0])
    fired = {limit: [i for i, parts in enumerate(hidden(clip, limit)) if parts["rules"]]
             for limit in (2.0, 3.0, 5.0, 6.0)}
    assert fired == {2.0: [7, 8, 9], 3.0: [8, 9], 5.0: [9], 6.0: []}
    assert rules.overlong_forearms(clip, 0.5, 6.0) == {}


def test_the_forearm_rule_reads_drawn_forearms_only():
    clip = forearm_clip([50.0] * 7 + [300.0])
    clip[-1]["keypoints_body"][4, 2] = 0.3
    assert rules.overlong_forearms(clip, 0.5, 2.0) == {}
    # an arm never drawn has no length to hold to
    assert rules.overlong_forearms(forearm_clip([50.0, 300.0], wrist_conf=0.1), 0.5, 2.0) == {}


def test_nothing_is_hidden_at_forearm_limit_0():
    clip = forearm_clip([50.0] * 4 + [300.0], arms=(RIGHT, LEFT))
    assert rules.overlong_forearms(clip, 0.5, 0.0) == {}
    assert hidden(clip, 0.0) == [{"body": [], "hands": [], "rules": {}}] * 5


def test_the_rule_names_the_wrist_and_the_hand_of_each_side():
    # the right wrist is 4, the left 7
    assert hidden(forearm_clip([50.0] * 4 + [120.0]), 2.0)[-1] == {
        "body": [4], "hands": ["right"], "rules": {"forearm_rule": ["right"]}}
    assert hidden(forearm_clip([50.0] * 4 + [120.0], arms=(LEFT,)), 2.0)[-1] == {
        "body": [7], "hands": ["left"], "rules": {"forearm_rule": ["left"]}}
    both = hidden(forearm_clip([50.0] * 4 + [120.0], arms=(RIGHT, LEFT)), 2.0)
    assert both[:4] == [{"body": [], "hands": [], "rules": {}}] * 4
    assert both[-1] == {"body": [4, 7], "hands": ["left", "right"], "rules": {"forearm_rule": ["right", "left"]}}


def test_the_rule_leaves_the_keypoints_alone():
    clip = forearm_clip([50.0] * 4 + [300.0], arms=(RIGHT, LEFT))
    before = [{k: np.array(v, copy=True) for k, v in frame.items() if isinstance(v, np.ndarray)} for frame in clip]
    rules.overlong_forearms(clip, 0.5, 2.0)
    assert all(np.array_equal(frame[k], v) for frame, kept in zip(clip, before) for k, v in kept.items())


def test_the_rule_is_a_pose_config_value_off_by_default():
    from pose_fakes import pose

    assert rules.RULES == ("forearm_rule",) and pose.PoseConfig().forearm_limit == 0.0
    for outside in (-0.1, 10.5):
        with pytest.raises(ValueError, match="forearm_limit"):
            pose.PoseConfig(forearm_limit=outside)
