"""The draw rules (libs/draw_rules.py) on hand-built keypoints: which frames limb_dedup's hand
and arm tests fire on and which side they name, which frames forearm_rule fires on at which
forearm_limit, and what each leaves out, with the geometry written out in pixels:

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
    return rules.hidden_parts(len(clip), {}, {}, rules.overlong_forearms(clip, 0.5, limit))


# -- limb_dedup: a hand on the other hand ----------------------------------------------------------

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


def test_the_hand_test_names_the_copys_hand_only():
    clip = [frame({**RIGHT_INTACT, **LEFT_INTACT}), frame({**RIGHT_INTACT, **LEFT_BROKEN}),
            frame({**RIGHT_INTACT, **LEFT_BROKEN})]
    assert rules.hidden_parts(3, rules.duplicate_hands(clip, 0.5), {}, {}) == [
        {"body": [], "hands": [], "rules": {}},
        {"body": [], "hands": ["left"], "rules": {"limb_dedup": ["left"]}},
        {"body": [], "hands": ["left"], "rules": {"limb_dedup": ["left"]}}]


def test_the_hand_test_leaves_the_keypoints_alone():
    clip = [frame({**RIGHT_INTACT, **LEFT_BROKEN}) for _ in range(3)]
    before = [{k: np.array(v, copy=True) for k, v in f.items() if isinstance(v, np.ndarray)} for f in clip]
    assert rules.duplicate_hands(clip, 0.5) == {"left": [0, 1, 2]}
    assert all(np.array_equal(f[k], v) for f, kept in zip(clip, before) for k, v in kept.items())


# -- limb_dedup: an arm along the other arm --------------------------------------------------------

# The torso: the shoulders 40 px apart, the hips 30 px, the nose 20 px above the neck (1.5 x 20 =
# 30 px): the body scale is the shoulders' 40 px, the reach 0.2 x 40 = 8 px.
TORSO = {0: (100.0, 80.0, 0.9), 1: (100.0, 100.0, 0.9), 2: (80.0, 100.0, 0.9), 5: (120.0, 100.0, 0.9),
         8: (85.0, 200.0, 0.9), 11: (115.0, 200.0, 0.9)}


def arms(right=(0.9, 0.9), left=(0.6, 0.6), elbow_gap=5.0, wrist_gap=5.0, torso=TORSO):
    """One frame on `torso`: the right elbow at (100, 150) and the right wrist at (100, 190), the left
    elbow `elbow_gap` px and the left wrist `wrist_gap` px right of them; `right` and `left` are the
    (elbow, wrist) confidences."""
    return meta({**torso, 3: (100.0, 150.0, right[0]), 4: (100.0, 190.0, right[1]),
                 6: (100.0 + elbow_gap, 150.0, left[0]), 7: (100.0 + wrist_gap, 190.0, left[1])})


def test_an_arm_drawn_on_the_other_arm_is_the_less_confident_one():
    # 5 px apart, under the 8 px reach: the left arm (0.6 + 0.6) is less confident than the right
    # (0.9 + 0.9)
    assert rules.mirrored_arms([arms()], 0.5) == {"left": [0]}
    assert rules.mirrored_arms([arms(right=(0.6, 0.7), left=(0.9, 0.8))], 0.5) == {"right": [0]}
    # the confidence names the copy, not what is drawn: the right forearm is drawn (0.55 + 0.55 =
    # 1.10), of the left arm only the elbow (0.95 + 0.3 = 1.25)
    assert rules.mirrored_arms([arms(right=(0.55, 0.55), left=(0.95, 0.3))], 0.5) == {"right": [0]}


def test_arms_apart_at_the_elbows_or_at_the_wrists_are_two_arms():
    # 10 px is over the 8 px reach
    assert rules.mirrored_arms([arms(elbow_gap=5.0, wrist_gap=10.0)], 0.5) == {}
    assert rules.mirrored_arms([arms(elbow_gap=10.0, wrist_gap=5.0)], 0.5) == {}
    assert rules.mirrored_arms([arms(elbow_gap=7.5, wrist_gap=7.5)], 0.5) == {"left": [0]}
    assert rules.mirrored_arms([arms(elbow_gap=8.5, wrist_gap=8.5)], 0.5) == {}


def test_equally_confident_arms_have_no_copy():
    assert rules.mirrored_arms([arms(right=(0.8, 0.7), left=(0.8, 0.7))], 0.5) == {}
    assert rules.mirrored_arms([arms(right=(0.8, 0.7), left=(0.7, 0.8))], 0.5) == {}


def test_one_forearm_drawn_and_something_of_the_other_arm():
    # the right forearm drawn, of the left arm the elbow only, or the wrist only: it fires
    assert rules.mirrored_arms([arms(left=(0.6, 0.3))], 0.5) == {"left": [0]}
    assert rules.mirrored_arms([arms(left=(0.3, 0.6))], 0.5) == {"left": [0]}
    # nothing of the left arm drawn: nothing to leave out
    assert rules.mirrored_arms([arms(left=(0.3, 0.3))], 0.5) == {}
    # no forearm drawn on either side: the right elbow and the left wrist only
    assert rules.mirrored_arms([arms(right=(0.9, 0.3), left=(0.3, 0.8))], 0.5) == {}
    # the rule reads what is drawn: at a threshold above every arm keypoint nothing is
    assert rules.mirrored_arms([arms()], 0.95) == {}


@pytest.mark.parametrize("torso, reach", [
    # the hips 60 px apart, over the 20 px shoulders and 1.5 x the 20 px neck-to-nose
    ({1: (100.0, 100.0, 0.9), 0: (100.0, 80.0, 0.9), 2: (90.0, 100.0, 0.9), 5: (110.0, 100.0, 0.9),
      8: (70.0, 200.0, 0.9), 11: (130.0, 200.0, 0.9)}, 12.0),
    # 1.5 x the 50 px neck-to-nose, over the 20 px shoulders and hips
    ({1: (100.0, 100.0, 0.9), 0: (100.0, 50.0, 0.9), 2: (90.0, 100.0, 0.9), 5: (110.0, 100.0, 0.9),
      8: (90.0, 200.0, 0.9), 11: (110.0, 200.0, 0.9)}, 15.0),
    # the positions count whether drawn or not: TORSO at confidence 0.1 still has the 40 px shoulders
    ({j: (x, y, 0.1) for j, (x, y, _) in TORSO.items()}, 8.0),
    # no torso at all (every point at (0, 0)): the 1 px floor
    ({}, 0.2),
])
def test_the_reach_is_0_2_of_the_widest_body_scale_term(torso, reach):
    assert rules.mirrored_arms([arms(elbow_gap=0.9 * reach, wrist_gap=0.9 * reach, torso=torso)], 0.5) == {"left": [0]}
    assert rules.mirrored_arms([arms(elbow_gap=1.1 * reach, wrist_gap=1.1 * reach, torso=torso)], 0.5) == {}


def test_the_arm_test_reads_each_frame_alone():
    # no neighbour condition: a copy on one frame between two arms apart fires
    apart = arms(elbow_gap=20.0, wrist_gap=20.0)
    clip = [arms(), apart, arms(right=(0.6, 0.6), left=(0.9, 0.9)), apart, arms()]
    mirrored = rules.mirrored_arms(clip, 0.5)
    assert mirrored == {"right": [2], "left": [0, 4]} and list(mirrored) == ["right", "left"]
    assert rules.mirrored_arms([arms()], 0.5) == {"left": [0]}


def test_the_arm_test_names_the_copys_elbow_wrist_and_hand():
    # the right elbow and wrist are 3 and 4, the left 6 and 7
    clip = [arms(), arms(elbow_gap=20.0, wrist_gap=20.0), arms(right=(0.6, 0.6), left=(0.9, 0.9))]
    assert rules.hidden_parts(3, {}, rules.mirrored_arms(clip, 0.5), {}) == [
        {"body": [6, 7], "hands": ["left"], "rules": {"limb_dedup": ["left"]}},
        {"body": [], "hands": [], "rules": {}},
        {"body": [3, 4], "hands": ["right"], "rules": {"limb_dedup": ["right"]}}]


def test_the_arm_test_leaves_the_keypoints_alone():
    clip = [arms(), arms(right=(0.6, 0.6), left=(0.9, 0.9))]
    before = [{k: np.array(v, copy=True) for k, v in f.items() if isinstance(v, np.ndarray)} for f in clip]
    assert rules.mirrored_arms(clip, 0.5) == {"right": [1], "left": [0]}
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


# -- all rules ------------------------------------------------------------------------------------

def test_limb_dedup_leaves_out_what_either_test_names():
    # the hand test: the left hand on frames 1 and 2; the arm test: the left elbow, wrist and hand on
    # frames 2 and 4, the right ones on frame 3
    assert rules.hidden_parts(5, {"left": [1, 2]}, {"right": [3], "left": [2, 4]}, {}) == [
        {"body": [], "hands": [], "rules": {}},
        {"body": [], "hands": ["left"], "rules": {"limb_dedup": ["left"]}},
        {"body": [6, 7], "hands": ["left"], "rules": {"limb_dedup": ["left"]}},
        {"body": [3, 4], "hands": ["right"], "rules": {"limb_dedup": ["right"]}},
        {"body": [6, 7], "hands": ["left"], "rules": {"limb_dedup": ["left"]}}]
    # a hand on one side and an arm on the other on one frame: both, the sides in SIDES order
    assert rules.hidden_parts(1, {"left": [0]}, {"right": [0]}, {}) == [
        {"body": [3, 4], "hands": ["left", "right"], "rules": {"limb_dedup": ["right", "left"]}}]


def test_a_part_any_rule_names_is_left_out():
    # limb_dedup: the left hand on frames 1 and 2, the left elbow, wrist and hand on frames 2 and 4;
    # forearm_rule: the right wrist and hand on frames 2 and 3
    overlong = {"right": {"median": 50.0, "ratios": {2: 2.4, 3: 2.2}}}
    assert rules.hidden_parts(5, {"left": [1, 2]}, {"left": [2, 4]}, overlong) == [
        {"body": [], "hands": [], "rules": {}},
        {"body": [], "hands": ["left"], "rules": {"limb_dedup": ["left"]}},
        {"body": [4, 6, 7], "hands": ["left", "right"], "rules": {"limb_dedup": ["left"], "forearm_rule": ["right"]}},
        {"body": [4], "hands": ["right"], "rules": {"forearm_rule": ["right"]}},
        {"body": [6, 7], "hands": ["left"], "rules": {"limb_dedup": ["left"]}}]
    # rules and tests naming one part leave it out once
    assert rules.hidden_parts(1, {"right": [0]}, {}, {"right": {"median": 50.0, "ratios": {0: 2.4}}}) == [
        {"body": [4], "hands": ["right"], "rules": {"limb_dedup": ["right"], "forearm_rule": ["right"]}}]
    assert rules.hidden_parts(1, {"right": [0]}, {"right": [0]}, {"right": {"median": 50.0, "ratios": {0: 2.4}}}) == [
        {"body": [3, 4], "hands": ["right"], "rules": {"limb_dedup": ["right"], "forearm_rule": ["right"]}}]


def test_the_rules_are_pose_config_values_off_by_default():
    from pose_fakes import pose

    config = pose.PoseConfig()
    assert rules.RULES == ("limb_dedup", "forearm_rule")
    assert config.limb_dedup is False and config.forearm_limit == 0.0
    for outside in (-0.1, 10.5):
        with pytest.raises(ValueError, match="forearm_limit"):
            pose.PoseConfig(forearm_limit=outside)
