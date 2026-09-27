"""The draw rules (libs/draw_rules.py) on hand-built keypoints: which frames hand_dedup fires on and
which side it names, which frames forearm_rule fires on at which forearm_limit, and what each leaves
out, with the geometry written out in pixels:

    python -m pytest tests/libs/test_draw_rules.py
"""
import numpy as np
import pytest

from pose_fakes import rules  # noqa: E402

W, H = 200, 400
# (elbow, wrist) of each arm in the AAPose layout
RIGHT, LEFT = (3, 4), (6, 7)
# A 21-point hand in pixels: a 5 x 5 px grid from (100, 200) to (120, 220), diagonal 28.3 px.
HAND = [(100.0 + (k % 5) * 5, 200.0 + (k // 5) * 5) for k in range(21)]


def meta(body=None, left=None, right=None):
    """One pose meta as pose_data's pose_metas_original holds it, from pixel rows: `body` a dict of
    AAPose index -> (x, y, conf), the hands lists of 21 (x, y, conf); anything not given sits at
    (0, 0) with confidence 0."""
    def rows(points, count):
        out = np.zeros((count, 3), np.float32)
        for j, (x, y, conf) in (points.items() if isinstance(points, dict) else enumerate(points or [])):
            out[j] = (x / W, y / H, conf)
        return out

    return {"width": W, "height": H, "keypoints_body": rows(body or {}, 20), "keypoints_left_hand": rows(left, 21),
            "keypoints_right_hand": rows(right, 21), "keypoints_face": np.zeros((69, 3), np.float32)}


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
    return rules.hidden_parts(len(clip), {}, rules.overlong_forearms(clip, 0.5, limit))


# -- hand_dedup -----------------------------------------------------------------------------------

def hand(dx=0.0, conf=0.9):
    return [(x + dx, y, conf) for x, y in HAND]


RIGHT_INTACT = {3: (90.0, 150.0, 0.9), 4: (100.0, 195.0, 0.9)}
RIGHT_BROKEN = {3: (90.0, 150.0, 0.2), 4: (100.0, 195.0, 0.9)}      # its elbow under 0.5
LEFT_INTACT = {6: (110.0, 150.0, 0.9), 7: (105.0, 195.0, 0.8)}
LEFT_BROKEN = {6: (110.0, 150.0, 0.3), 7: (105.0, 195.0, 0.8)}      # its elbow under 0.5
LEFT_NO_WRIST = {6: (110.0, 150.0, 0.9), 7: (105.0, 195.0, 0.4)}    # its wrist under 0.5


def frame(arms, dx=5.0):
    """The right hand at HAND, the left hand `dx` px to its right: 5 px apart is 5 / 28.3 = 0.18 of
    the hand, under 0.6 (one hand drawn twice); 20 px is 0.71 (two hands)."""
    return meta(arms, left=hand(dx=dx), right=hand())


def test_a_hand_on_the_other_hand_is_the_broken_arms():
    clip = [frame({**RIGHT_INTACT, **LEFT_BROKEN})] * 3
    assert rules.duplicate_hands(clip, 0.5) == {"left": [0, 1, 2]}
    clip = [frame({**RIGHT_BROKEN, **LEFT_INTACT})] * 3
    assert rules.duplicate_hands(clip, 0.5) == {"right": [0, 1, 2]}
    # an arm is broken when its elbow or its wrist is not drawn
    assert rules.duplicate_hands([frame({**RIGHT_INTACT, **LEFT_NO_WRIST})] * 2, 0.5) == {"left": [0, 1]}


def test_overlapping_hands_with_both_arms_intact_or_both_broken_are_left_alone():
    assert rules.duplicate_hands([frame({**RIGHT_INTACT, **LEFT_INTACT})] * 3, 0.5) == {}
    assert rules.duplicate_hands([frame({**RIGHT_BROKEN, **LEFT_BROKEN})] * 3, 0.5) == {}


def test_two_hands_apart_are_two_hands():
    assert rules.duplicate_hands([frame({**RIGHT_INTACT, **LEFT_BROKEN}, dx=20.0)] * 3, 0.5) == {}


def test_hands_sharing_fewer_than_three_drawn_points_are_not_compared():
    left = hand(dx=5.0, conf=0.2)
    left[0], left[1] = (*HAND[0], 0.9), (*HAND[1], 0.9)
    clip = [meta({**RIGHT_INTACT, **LEFT_BROKEN}, left=left, right=hand())] * 3
    assert rules.duplicate_hands(clip, 0.5) == {}
    # a third shared point makes them comparable: the two hands coincide
    left[2] = (*HAND[2], 0.9)
    clip = [meta({**RIGHT_INTACT, **LEFT_BROKEN}, left=left, right=hand())] * 3
    assert rules.duplicate_hands(clip, 0.5) == {"left": [0, 1, 2]}
    # the rule reads what is drawn: at a threshold above the hands' 0.9 nothing is shared
    assert rules.duplicate_hands([frame({**RIGHT_INTACT, **LEFT_BROKEN})] * 3, 0.95) == {}


def test_a_one_frame_dip_of_an_intact_arm_is_no_hidden_arm():
    # the left arm is broken on frame 1 only, intact on frames 0 and 2
    clip = [frame({**RIGHT_INTACT, **LEFT_INTACT}), frame({**RIGHT_INTACT, **LEFT_BROKEN}),
            frame({**RIGHT_INTACT, **LEFT_INTACT})]
    assert rules.duplicate_hands(clip, 0.5) == {}


def test_the_copys_arm_broken_on_one_neighbour_is_a_hidden_arm():
    # the left arm is broken on frame 1 and on one neighbour; that neighbour's hands are apart, so
    # only frame 1 fires
    before = [frame({**RIGHT_INTACT, **LEFT_BROKEN}, dx=20.0), frame({**RIGHT_INTACT, **LEFT_BROKEN}),
              frame({**RIGHT_INTACT, **LEFT_INTACT})]
    after = [frame({**RIGHT_INTACT, **LEFT_INTACT}), frame({**RIGHT_INTACT, **LEFT_BROKEN}),
             frame({**RIGHT_INTACT, **LEFT_NO_WRIST}, dx=20.0)]
    assert rules.duplicate_hands(before, 0.5) == {"left": [1]}
    assert rules.duplicate_hands(after, 0.5) == {"left": [1]}
    # the neighbour counts only when it is the copy's own arm that is broken there
    other = [frame({**RIGHT_BROKEN, **LEFT_INTACT}, dx=20.0), frame({**RIGHT_INTACT, **LEFT_BROKEN}),
             frame({**RIGHT_INTACT, **LEFT_INTACT})]
    assert rules.duplicate_hands(other, 0.5) == {}


def test_the_first_and_the_last_frame_have_one_neighbour():
    broken_apart = frame({**RIGHT_INTACT, **LEFT_BROKEN}, dx=20.0)
    copy, intact = frame({**RIGHT_INTACT, **LEFT_BROKEN}), frame({**RIGHT_INTACT, **LEFT_INTACT})
    assert rules.duplicate_hands([copy, broken_apart, intact], 0.5) == {"left": [0]}
    assert rules.duplicate_hands([copy, intact, intact], 0.5) == {}
    assert rules.duplicate_hands([intact, broken_apart, copy], 0.5) == {"left": [2]}
    assert rules.duplicate_hands([intact, intact, copy], 0.5) == {}
    # a one-frame clip has no neighbour to be broken on
    assert rules.duplicate_hands([copy], 0.5) == {}


def test_hand_dedup_names_the_copys_hand_only():
    clip = [frame({**RIGHT_INTACT, **LEFT_INTACT}), frame({**RIGHT_INTACT, **LEFT_BROKEN}),
            frame({**RIGHT_INTACT, **LEFT_BROKEN})]
    assert rules.hidden_parts(3, rules.duplicate_hands(clip, 0.5), {}) == [
        {"body": [], "hands": [], "rules": {}},
        {"body": [], "hands": ["left"], "rules": {"hand_dedup": ["left"]}},
        {"body": [], "hands": ["left"], "rules": {"hand_dedup": ["left"]}}]


def test_hand_dedup_leaves_the_keypoints_alone():
    clip = [frame({**RIGHT_INTACT, **LEFT_BROKEN}) for _ in range(3)]
    before = [{k: np.array(v, copy=True) for k, v in f.items() if isinstance(v, np.ndarray)} for f in clip]
    assert rules.duplicate_hands(clip, 0.5) == {"left": [0, 1, 2]}
    assert all(np.array_equal(f[k], v) for f, kept in zip(clip, before) for k, v in kept.items())


# -- forearm_rule ---------------------------------------------------------------------------------

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


# -- both rules -----------------------------------------------------------------------------------

def test_a_part_either_rule_names_is_left_out():
    # hand_dedup: the left hand on frames 1 and 2; forearm_rule: the right wrist and hand on frames
    # 2 and 3
    overlong = {"right": {"median": 50.0, "ratios": {2: 2.4, 3: 2.2}}}
    assert rules.hidden_parts(4, {"left": [1, 2]}, overlong) == [
        {"body": [], "hands": [], "rules": {}},
        {"body": [], "hands": ["left"], "rules": {"hand_dedup": ["left"]}},
        {"body": [4], "hands": ["left", "right"], "rules": {"hand_dedup": ["left"], "forearm_rule": ["right"]}},
        {"body": [4], "hands": ["right"], "rules": {"forearm_rule": ["right"]}}]
    # both naming one hand leave it out once
    assert rules.hidden_parts(1, {"right": [0]}, {"right": {"median": 50.0, "ratios": {0: 2.4}}}) == [
        {"body": [4], "hands": ["right"], "rules": {"hand_dedup": ["right"], "forearm_rule": ["right"]}}]


def test_the_rules_are_pose_config_values_off_by_default():
    from pose_fakes import pose

    config = pose.PoseConfig()
    assert rules.RULES == ("hand_dedup", "forearm_rule")
    assert config.hand_dedup is False and config.forearm_limit == 0.0
    for outside in (-0.1, 10.5):
        with pytest.raises(ValueError, match="forearm_limit"):
            pose.PoseConfig(forearm_limit=outside)
