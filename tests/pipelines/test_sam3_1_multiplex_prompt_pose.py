"""SAM 3.1 Multiplex prompt_pose mode: the keypoints it reads (those the pose images draw), the rule
that picks the frames it refines and their points (C1, C4 and C5, each frame on its own, on
synthetic masks and keypoints), the
demotion and the two-sided memory of its second pass, the first frame the refine can reach, the
capture segment_by_prompt hands over, and the whole mode on the scripted tracker of
test_sam3_1_multiplex_ab (the refine stood in for; its own tests are in
tests/models/test_sam3_1_multiplex_refine.py). No model is loaded.

Needs ComfyUI importable (the ComfyUI root on PYTHONPATH), like the other SAM 3.1 Multiplex tests:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/pipelines/test_sam3_1_multiplex_prompt_pose.py
"""
import dataclasses

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import sam3  # noqa: E402
from test_sam3_1_multiplex_ab import (PROBATION, SPECK, DefaultsTracker, FakeModel, FakeTracker,  # noqa: E402,F401
                                      box, config, pointers, position, range_prep, reproduce, rig)
from test_sam3_1_multiplex_ab import H as RIG_H  # noqa: E402
from test_sam3_1_multiplex_ab import N as RIG_N  # noqa: E402
from test_sam3_1_multiplex_ab import W as RIG_W  # noqa: E402

NOTHING = {"body": [], "hands": [], "rules": {}}    # a frame no draw rule leaves anything out of


def right_hand(j):
    return sam3.HANDS["right"] + j


def left_hand(j):
    return sam3.HANDS["left"] + j


def meta(points, W, H, conf=0.9):
    """One frame of pose_metas_original: the pose-order keypoints `points` ({keypoint: (x, y)} in
    pixels) at confidence `conf` (or {keypoint: (x, y, conf)}), every other keypoint at confidence 0;
    the face keypoints all confident and all where the first point is."""
    rows = np.zeros((sam3.KEYPOINT_COUNT, 3))
    for k, p in points.items():
        x, y, c = p if len(p) == 3 else (*p, conf)
        rows[k] = (x / W, y / H, c)
    face = np.tile([*rows[next(iter(points))][:2], 0.9], (69, 1)) if points else np.zeros((69, 3))
    return {"width": W, "height": H, "keypoints_body": rows[:20].copy(), "keypoints_left_hand": rows[20:41].copy(),
            "keypoints_right_hand": rows[41:].copy(), "keypoints_face": face}


def pose_data(metas, threshold=0.5, **rules):
    return {"pose_metas_original": metas, "draw_threshold": threshold,
            "pose_config": {"min_keypoint_conf": 0.3, "forearm_limit": 0.0, "limb_dedup": False,
                            "back_view_face": False, **rules}}


# --- the scene the rule is tested on ------------------------------------------------------------
# A 200 x 300 portrait frame (the shorter side 200, so the default D is 14 px). The body is on
# every frame; the forearm and hand are on every frame but LOST, where the track dropped them. Five
# right-hand keypoints hold still in the middle of the forearm-and-hand block, 60 px right of the
# body's last column on the frame that lost it.
W, H, FRAMES, LOST = 200, 300, 5, 2
BODY = (slice(50, 250), slice(20, 60))
ARM = (slice(80, 120), slice(60, 140))
HAND = {right_hand(j): (119.5, 90.5 + 4 * j) for j in range(5)}


def masks_of(lost=(LOST,), arm=ARM, empty=(), W=W, H=H):
    masks = torch.zeros(FRAMES, H, W)
    for f in range(FRAMES):
        if f not in empty:
            masks[f][BODY] = 1.0
            if f not in lost:
                masks[f][arm] = 1.0
    return masks


def fired(masks, frames_points, birth=0, share=0.07, threshold=0.5, **rules):
    """The rule on `masks` and one {keypoint: position} per frame, through the pose reader."""
    N, H, W = masks.shape
    metas = [meta(points, W, H) for points in frames_points]
    metas, threshold, hidden = sam3.prompt_pose_inputs(pose_data(metas, threshold, **rules), N)
    xy, drawn = sam3.drawn_keypoints(metas, threshold, hidden, H, W)
    return sam3.refine_points(masks, xy, drawn, birth, share * min(H, W))


def still(points, frames=FRAMES):
    return [dict(points) for _ in range(frames)]


def test_a_hand_lost_for_one_frame_far_outside_the_mask_is_refined_with_its_drawn_keypoints():
    assert fired(masks_of(), still(HAND)) == {LOST: sorted(HAND)}


def test_c1_the_keypoints_must_lie_the_distance_or_more_from_the_mask():
    # the hand is 60 px out: D 59 px refines it, D 61 px (60 = D - 1) does not
    assert fired(masks_of(), still(HAND), share=59 / 200) == {LOST: sorted(HAND)}
    assert fired(masks_of(), still(HAND), share=61 / 200) == {}


def test_c1_the_distance_is_a_share_of_the_shorter_side():
    # the same 60 px at D = 0.25 of the shorter side: 50 px on a 200 px wide frame, 75 px on 400 x 300
    assert fired(masks_of(), still(HAND), share=0.25) == {LOST: sorted(HAND)}
    wide = masks_of(W=400, H=300)
    assert fired(wide, still(HAND), share=0.25) == {}
    assert fired(wide, still(HAND), share=0.195) == {LOST: sorted(HAND)}    # 58.5 px


def test_a_limb_lost_over_several_frames_refines_each_frame_it_qualifies_on():
    assert fired(masks_of(lost=(1, 2, 3)), still(HAND)) == {1: sorted(HAND), 2: sorted(HAND), 3: sorted(HAND)}
    # each frame is judged on its own: not drawn on 2, and 6 px from the mask on 3 (under D, 14 px)
    frames = still(HAND)
    frames[2] = {}
    frames[3] = {k: (65.5, y) for k, (x, y) in HAND.items()}
    assert fired(masks_of(lost=(1, 2, 3)), frames) == {1: sorted(HAND)}
    # nor do the frames around a loss decide it: not drawn on the frame before, the hand is refined
    frames = still(HAND)
    frames[LOST - 1] = {}
    assert fired(masks_of(), frames) == {LOST: sorted(HAND)}


def test_a_keypoint_is_judged_where_it_lies_on_the_frame_however_far_it_moved():
    frames = still(HAND)
    frames[LOST] = {k: (179.5, y) for k, (x, y) in HAND.items()}           # 60 px off where it was before and after
    assert fired(masks_of(), frames) == {LOST: sorted(HAND)}


def test_a_loss_from_the_birth_on_refines_the_birth_frame():
    assert fired(masks_of(lost=(0, 1)), still(HAND), birth=0) == {0: sorted(HAND), 1: sorted(HAND)}
    # born on 1: frame 0, before the birth (pass 1's backward fill), is not refined
    assert fired(masks_of(lost=(0, 1, 2)), still(HAND), birth=1) == {1: sorted(HAND), 2: sorted(HAND)}


def test_c4_a_limb_needs_three_qualifying_keypoints():
    two = {k: HAND[k] for k in sorted(HAND)[:2]}
    three = {k: HAND[k] for k in sorted(HAND)[:3]}
    assert fired(masks_of(), still(two)) == {}
    assert fired(masks_of(), still(three)) == {LOST: sorted(three)}


def test_c5_a_hand_still_20_percent_inside_is_no_whole_limb_loss():
    points = {**{k: HAND[k] for k in sorted(HAND)[:4]}, right_hand(4): (40.5, 100.5)}   # the fifth on the body
    assert fired(masks_of(), still(points)) == {}
    # an elbow is not part of its hand: an elbow on the body leaves the hand whole
    assert fired(masks_of(), still({**HAND, sam3.R_ELBOW: (40.5, 100.5)})) == {LOST: sorted(HAND)}


def test_only_keypoints_the_pose_images_draw_count():
    faint = {k: (x, y, 0.49) for k, (x, y) in HAND.items()}
    # 0.49 against pose_data's draw_threshold 0.5, whatever the pose's min_keypoint_conf (0.3)
    assert fired(masks_of(), still(faint)) == {}
    assert fired(masks_of(), still({k: (x, y, 0.5) for k, (x, y) in HAND.items()})) == {LOST: sorted(HAND)}
    assert fired(masks_of(), still(faint), threshold=0.45) == {LOST: sorted(HAND)}


def test_a_keypoint_off_the_canvas_is_dropped_not_clamped():
    arm = (slice(80, 120), slice(60, 200))                  # the forearm and hand reach the right edge
    edge = {right_hand(0): (150.5, 90.5), right_hand(1): (150.5, 94.5), right_hand(2): (199.6, 98.5)}
    frames = still(edge)
    frames[LOST] = {**edge, right_hand(2): (200.4, 98.5)}      # past the edge on the lost frame
    assert fired(masks_of(arm=arm), frames) == {}              # two keypoints are left
    frames[LOST] = {**edge, right_hand(2): (199.9, 98.5)}      # on the canvas: three
    assert fired(masks_of(arm=arm), frames) == {LOST: sorted(edge)}


def test_the_reader_keeps_the_draw_code_s_rules():
    points = {right_hand(0): (0.5, 50.0), right_hand(1): (1.0, 50.0), right_hand(2): (50.0, 0.9),
              sam3.NOSE: (0.5, 50.0), sam3.R_WRIST: (200.0, 50.0), sam3.R_ELBOW: (-0.1, 50.0),
              left_hand(0): (199.9, 299.9)}
    xy, drawn = sam3.drawn_keypoints([meta(points, W, H)], 0.5, [NOTHING], H, W)
    drawn = {k: bool(drawn[0, k]) for k in points}
    # a hand keypoint needs int(x) and int(y) of 1 or more (the eps rule), a body keypoint does not;
    # x = W or x < 0 is off the canvas
    assert drawn == {right_hand(0): False, right_hand(1): True, right_hand(2): False, sam3.NOSE: True,
                     sam3.R_WRIST: False, sam3.R_ELBOW: False, left_hand(0): True}
    assert np.allclose(xy[0, right_hand(1)], (1.0 / W, 50.0 / H))
    hidden = [{"body": [sam3.NOSE], "hands": ["right"], "rules": {}}]
    _, drawn = sam3.drawn_keypoints([meta(points, W, H)], 0.5, hidden, H, W)
    assert not drawn[0, sam3.NOSE] and not drawn[0, right_hand(1)] and drawn[0, left_hand(0)]


def test_head_shoulders_hips_and_face_never_trigger():
    trunk = {k: (119.5, 90.5 + 2 * k) for k in (0, 1, 2, 5, 8, 11, 14, 15, 16, 17)}
    assert fired(masks_of(), still(trunk)) == {}
    # the face keypoints (all confident, on the lost region) are never read
    assert fired(masks_of(), still({**trunk, **{k: HAND[k] for k in sorted(HAND)[:2]}})) == {}


def test_a_forearm_the_forearm_limit_draw_rule_hides_is_no_candidate():
    """The right forearm is 20 px long on every frame but LOST, where its wrist lies 60 px from the
    elbow: 3 times its clip median, so forearm_limit 2.0 leaves that wrist and its hand out of the
    pose image, and out of the points."""
    arm = {sam3.R_ELBOW: (50.5, 100.5), sam3.R_WRIST: (70.5, 100.5)}
    frames = still({**arm, **HAND})
    frames[LOST] = {**arm, sam3.R_WRIST: (110.5, 100.5), **HAND}
    assert fired(masks_of(), frames) == {LOST: sorted([sam3.R_WRIST, *HAND])}   # the wrist 51 px out, with the hand
    assert fired(masks_of(), frames, forearm_limit=2.0) == {}


@pytest.mark.parametrize("birth, lost, empty, refined", [
    (1, (2,), (), {2}),          # the frame after the birth can be refined
    (2, (2,), (), {2}),          # the birth frame too
    (3, (2,), (), set()),        # never a frame before the birth
    (0, (4,), (), {4}),          # the last frame can be
    (0, (3,), (), {3}),
    (0, (2,), (2,), set()),      # never a frame whose mask is empty
    (-1, (2,), (), set()),       # no track, nothing
])
def test_which_frames_the_rule_can_refine(birth, lost, empty, refined):
    assert set(fired(masks_of(lost=lost, empty=empty), still(HAND), birth=birth)) == refined


def test_the_points_go_in_pose_order_body_left_hand_right_hand():
    """Both hands dropped together, and the right wrist: 23 points, the body keypoint first, then
    the left hand's, then the right hand's (the refine then sends the first 8 and the last 8)."""
    points = {**{left_hand(j): (100.5 + 3 * j, 84.5) for j in range(11)},
              **{right_hand(j): (100.5 + 3 * j, 110.5) for j in range(11)},
              sam3.R_WRIST: (75.5, 100.5)}
    got = fired(masks_of(), still(points))
    assert got == {LOST: [sam3.R_WRIST] + [left_hand(j) for j in range(11)] + [right_hand(j) for j in range(11)]}


def test_the_pose_reader_raises_on_pose_data_it_cannot_read():
    good = pose_data([meta(HAND, W, H)] * 2)
    metas, threshold, hidden = sam3.prompt_pose_inputs(good, 2)
    assert metas is good["pose_metas_original"] and threshold == 0.5 and hidden == [NOTHING, NOTHING]
    with pytest.raises(ValueError, match="covers 2 pose frames, the images are 3"):
        sam3.prompt_pose_inputs(good, 3)
    for missing in ("draw_threshold", "pose_metas_original"):
        with pytest.raises(ValueError, match="draw_threshold"):
            sam3.prompt_pose_inputs({k: v for k, v in good.items() if k != missing}, 2)
    no_rules = {**good, "pose_config": {"min_keypoint_conf": 0.3}}
    with pytest.raises(ValueError, match="forearm_limit, limb_dedup, back_view_face"):
        sam3.prompt_pose_inputs(no_rules, 2)


def test_the_draw_rules_are_read_from_pose_config():
    frames = [meta({sam3.R_ELBOW: (50.0, 100.0), sam3.R_WRIST: (50.0, 120.0 if f != 2 else 170.0)}, W, H)
              for f in range(5)]
    _, _, hidden = sam3.prompt_pose_inputs(pose_data(frames, forearm_limit=2.0), 5)
    assert [h["body"] for h in hidden] == [[], [], [sam3.R_WRIST], [], []] and hidden[2]["hands"] == ["right"]


# --- the mask-driven trigger: a region the mask drops for a few frames ---------------------------
# Her rectangle (400 x 200 by default, rows and columns from 40) and an 80 x 80 square held out at
# its right (rows 100-180) on every frame but the lost ones. On a frame without the square the Wan
# Animate workflow's final covers her rectangle grown by 10 px, so the region it loses is the
# square's part more than 10 px from her: on the default scene rows 100-180, columns 250-320, 5600
# px of her 86400 (6.5%), and the final's grid on the frames around (the grown box, columns 30-330
# in 9 blocks of 33, rows 30-450 in 13 of 32) has a whole block in it: columns 261-294, rows 126-158.
# Her right elbow is drawn on her rectangle and her right wrist on the square, so her forearm
# crosses the region on the frames that hold it; on a lost frame the pose loses the wrist too (under
# the draw threshold where it was), as on the clip the trigger was made for.

DROP_FRAMES, DROP_LOST = 7, (3,)


def dropout_scene(lost=DROP_LOST, body=(400, 200), moved=(), empty=(), wrist=None, frames=DROP_FRAMES):
    """(masks [frames, H, W], pose_metas) of the scene: the square is gone on `lost` and 160 px lower
    on `moved`; the whole mask is empty on `empty`; the wrist at `wrist` (x, y) when given, else on
    the square."""
    rows, cols = body
    H, W = 40 + rows + 40, 40 + cols + 80 + 40
    right = 40 + cols
    masks = torch.zeros(frames, H, W)
    metas = []
    for f in range(frames):
        top = 260 if f in moved else 100
        if f not in empty:
            masks[f, 40:40 + rows, 40:right] = 1.0
            if f not in lost:
                masks[f, top:top + 80, right:right + 80] = 1.0
        x, y = wrist or (right + 50.5, 140.5)
        metas.append(meta({sam3.R_ELBOW: (right - 39.5, 140.5), sam3.R_WRIST: (x, y, 0.1 if f in lost else 0.9)}, W, H))
    return masks, metas


def dropped(masks, metas, birth=0):
    return sam3.dropped_regions(masks, metas, 0.5, birth)


def test_a_hand_sized_region_dropped_for_one_frame_and_held_on_both_sides_is_refined():
    masks, metas = dropout_scene()
    got = dropped(masks, metas)
    assert list(got) == [3] and got[3][1] == [2, 4]
    # the first point is the region's deepest pixel, the first in raster order of the pixels 35 px
    # from its outline (rows 34-45, columns 34-35 of the 80 x 70 region)
    assert got[3][0][0] == (250 + 34.5, 100 + 34.5)


def test_the_points_lie_well_inside_the_region_the_frame_lacks_and_apart():
    masks, metas = dropout_scene()
    points = dropped(masks, metas)[3][0]
    assert 2 <= len(points) <= sam3.MAX_REFINE_POINTS
    for x, y in points:
        # at least half the region's depth (35 px) inside it: 18 px or more from its outline
        assert 250 + 17 < x < 320 - 17 and 100 + 17 < y < 180 - 17, (x, y)
        assert masks[2, int(y), int(x)] and masks[4, int(y), int(x)] and not masks[3, int(y), int(x)]
    for i, (x, y) in enumerate(points):
        assert all(np.hypot(x - u, y - v) > 17.5 for u, v in points[:i]), (x, y)


def test_region_points_on_a_strip_go_along_its_middle_row_capped():
    # a 5 x 15 strip, 3 px deep along its middle row (columns 2-12): the points are the pixels of that
    # row 2 px apart (more than half the depth), from the left; the pixels a row off lie within
    # 1.5 px of one of them
    strip = np.ones((5, 15), bool)
    assert sam3.region_points(strip, (100, 50)) == [(50 + c + 0.5, 102.5) for c in (2, 4, 6, 8, 10, 12)]
    # a 5 x 60 strip has 28 such pixels: the first MAX_REFINE_POINTS are taken
    points = sam3.region_points(np.ones((5, 60), bool), (0, 0))
    assert sam3.MAX_REFINE_POINTS == 16 and points == [(c + 0.5, 2.5) for c in range(2, 33, 2)]


def test_a_region_under_the_hand_floor_is_not_refined():
    # the same 5600 px region: 1.43% of her 640 x 600 rectangle and the square (under LOSS_HAND,
    # 1.5%), 1.64% of a 600 x 560 one; each holds a whole block of its grid
    assert dropped(*dropout_scene(body=(640, 600))) == {}
    assert list(dropped(*dropout_scene(body=(600, 560)))) == [3]


@pytest.mark.parametrize("lost", [(3, 4, 5, 6), (0, 1, 2)])
def test_an_open_run_is_not_refined(lost):
    # dropped to the clip's last frame, or from its first: no frame holds it on the other side
    assert dropped(*dropout_scene(lost=lost)) == {}


def test_a_region_not_held_on_both_sides_is_not_refined():
    # from frame 4 on the square is 160 px lower: no frame after the drop holds the region again
    assert dropped(*dropout_scene(moved=(4, 5, 6))) == {}


def test_a_run_of_up_to_eight_frames_is_refined_on_each():
    got = dropped(*dropout_scene(lost=range(1, 9), frames=11))
    assert list(got) == list(range(1, 9)) and {tuple(h) for _, h in got.values()} == {(0, 9)}
    assert dropped(*dropout_scene(lost=range(1, 10), frames=11)) == {}


def test_without_the_body_in_the_region_nothing_is_refined():
    # her wrist drawn on her rectangle: no limb of hers crosses the region, which could be background
    assert dropped(*dropout_scene(wrist=(100.5, 140.5))) == {}


@pytest.mark.parametrize("birth, empty, refined", [
    (0, (), {3}),
    (3, (), {3}),          # the birth frame can be refined
    (4, (), set()),        # never a frame before the birth
    (0, (3,), set()),      # never a frame whose mask is empty
    (-1, (), set()),       # no track, nothing
])
def test_which_frames_the_mask_trigger_can_refine(birth, empty, refined):
    assert set(dropped(*dropout_scene(empty=empty), birth=birth)) == refined


# --- demotion and the second pass's memory ------------------------------------------------------

def test_demotion_takes_pass_1_s_conditioning_frames_within_16_of_a_refined_frame():
    g = 40
    conditioning = {t: f"out {t}" for t in (g - 17, g - 16, g, g + 16, g + 17)}
    kept, demoted = sam3.demote(conditioning, {g: "refined"})
    assert sam3.DEMOTION_WINDOW == 16
    # g itself is refined, not demoted: its refine replaces pass 1's conditioning of it
    assert kept == {g - 17: f"out {g - 17}", g + 17: f"out {g + 17}"} and demoted == [g - 16, g + 16]
    # refined frames 10 and 70: 23 and 24 are 13 and 14 frames from 10, 56 and 57 14 and 13 from 70
    kept, demoted = sam3.demote(conditioning, {10: "refined", 70: "refined"})
    assert demoted == [g - 17, g - 16, g + 16, g + 17] and list(kept) == [g]


def test_the_conditioning_frames_read_are_the_closest_on_both_sides():
    cond = {t: t for t in (0, 16, 32, 48, 64)}
    selected, unselected = sam3.closest_conditioning(cond, 30, 4)
    # the closest before (16), the closest at or after (32), then the nearest of the rest (48, then 0)
    assert sorted(selected) == [0, 16, 32, 48] and unselected == {64: 64}
    selected, _ = sam3.closest_conditioning(cond, 32, 2)
    assert sorted(selected) == [16, 32]                  # "at or after" takes the frame itself
    selected, _ = sam3.closest_conditioning({t: t for t in (0, 16, 32, 48)}, 24, 3)
    assert sorted(selected) == [0, 16, 32]               # 0 and 48 tie: the earlier
    selected, unselected = sam3.closest_conditioning(cond, 70, 2)
    assert sorted(selected) == [48, 64] and sorted(unselected) == [0, 16, 32]
    assert sam3.closest_conditioning(cond, 30, 5) == (cond, {})


def stored_outputs(scores):
    return {t: {"memory_score": s, "obj_ptr": t, "maskmem_features": t} for t, s in scores.items()}


def test_the_second_pass_view_reads_an_unselected_conditioning_frame_as_ordinary_memory():
    """Without memory selection, by distance: of the conditioning frames 10, 12, 14 and 20, frame 15
    selects 14 and 20 (keep 2) and reads 10 and 12, 5 and 3 frames back, as ordinary memory."""
    cond = {t: {"obj_ptr": f"c{t}", "maskmem_features": f"c{t}"} for t in (10, 12, 14, 20)}
    stored = stored_outputs({11: 0.5, 13: 0.5})
    view = sam3.pass_two_view(cond, stored, 15, 2, 5, selection=False)
    assert sorted(view["cond_frame_outputs"]) == [14, 20]
    assert view["non_cond_frame_outputs"] == {10: cond[10], 12: cond[12], **stored}
    # a conditioning frame after the frame is never ordinary memory
    view = sam3.pass_two_view(cond, stored, 11, 2, 5, selection=False)
    assert sorted(view["cond_frame_outputs"]) == [10, 12] and view["non_cond_frame_outputs"] == stored


def outputs(*frames):
    return {t: f"out {t}" for t in frames}


@pytest.mark.parametrize("selection", [False, True])
@pytest.mark.parametrize("keep, first", [(2, 31), (3, 26)])
def test_the_first_influenced_frame_is_the_first_whose_conditioning_frames_change(selection, keep, first):
    """Pass 1's conditioning frames 0, 10, 30 and 50, frame 66 refined: 50 lies 16 from it and is
    demoted. Keeping 2, frames 11-30 read 10 and 30 either way, and 31 is the first to read 66 (30
    before it, 66 after). Keeping 3, the third is the nearest of the rest: 0 up to frame 25 either
    way (on 25, 0 and 50 tie and the earlier wins), but on 26 pass 1's 50 (24 away) beats 0 (26
    away), while without 50 the view keeps 0 (66 is 40 away): the demotion changes frame 26's view
    five frames before any frame reads 66. The unselected conditioning frames before a frame, read
    as ordinary memory without memory selection, move neither."""
    baseline = outputs(0, 10, 30, 50)
    conditioning = {**outputs(0, 10, 30), 66: "refined"}
    assert sam3.first_influenced(conditioning, baseline, {66: "refined"}, 0, 80, keep, 15, selection) == first


@pytest.mark.parametrize("selection", [False, True])
def test_a_refined_frame_pass_1_conditioned_on_counts_although_the_frames_are_the_same(selection):
    """Frame 30 refined, a conditioning frame of pass 1 too, nothing within 16 of it: frames 11-30
    read 10 and 30 in both passes, and 11 is the first of them, reading 30's refine."""
    baseline = outputs(0, 10, 30, 50)
    conditioning = {**outputs(0, 10, 50), 30: "refined"}
    assert sam3.first_influenced(conditioning, baseline, {30: "refined"}, 0, 80, 2, 15, selection) == 11


def test_the_first_influenced_frame_is_never_before_the_birth():
    # the birth (5) and the anchor on 20 lie within 16 of frame 12, both demoted: the birth reads 12
    assert sam3.first_influenced({12: "refined"}, outputs(5, 20), {12: "refined"}, 5, 40, 2, 15, False) == 5
    # nothing changes: none
    assert sam3.first_influenced(outputs(5, 20), outputs(5, 20), {}, 5, 40, 2, 15, False) == 40


@pytest.mark.parametrize("refined, kept, demoted", [
    ((44, 45, 46), (0, 10, 80), [40, 60]),     # 40 is 4 from 44, 60 14 from 46
    ((58, 59, 60), (0, 10, 40, 80), []),       # 40 is 18 from 58, 80 20 from 60; the anchor on 60 is refined
    ((0, 1, 2), (40, 60, 80), [10]),           # from the birth on: 10 is 8 from 2; the birth is refined
])
def test_adjacent_refined_frames_demote_what_lies_within_16_of_any_of_them(refined, kept, demoted):
    conditioning = outputs(0, 10, 40, 60, 80)          # pass 1's: the birth on 0 and four anchors
    got_kept, got_demoted = sam3.demote(conditioning, {g: "refined" for g in refined})
    assert got_kept == outputs(*kept) and got_demoted == demoted


def test_between_adjacent_refined_frames_the_view_takes_the_closest_on_both_sides():
    cond = {t: t for t in (0, 20, 21, 22, 60)}
    selected, unselected = sam3.closest_conditioning(cond, 30, 4)
    assert sorted(selected) == [20, 21, 22, 60] and unselected == {0: 0}     # 22 before, 60 after, then 21, 20
    selected, unselected = sam3.closest_conditioning(cond, 10, 4)
    assert sorted(selected) == [0, 20, 21, 22] and unselected == {60: 60}    # 0 before, 20 after, then 21, 22
    # keeping 2: 22 before, 60 after however far; 0, 20 and 21, before the frame, are ordinary memory
    view = sam3.pass_two_view(cond, {}, 23, 2, 5, selection=False)
    assert sorted(view["cond_frame_outputs"]) == [22, 60] and sorted(view["non_cond_frame_outputs"]) == [0, 20, 21]


@pytest.mark.parametrize("selection", [False, True])
@pytest.mark.parametrize("keep, first", [(2, 21), (3, 21), (4, 0)])
def test_the_first_influenced_frame_with_adjacent_refined_frames(selection, keep, first):
    """Pass 1's conditioning frames 0, 10, 20, 70 and 90, frames 74-76 refined: 70 and 90 lie within
    16 of them and are demoted. Keeping 2 or 3, frames up to 20 read 0, 10 and 20 either way, and 21
    is the first to read 74 (20 before it, 74 after, where pass 1 had 70). Keeping 4, frame 0 reads
    74 in place of pass 1's 70 already."""
    baseline = outputs(0, 10, 20, 70, 90)
    refined = {g: "refined" for g in (74, 75, 76)}
    kept, demoted = sam3.demote(baseline, refined)
    assert demoted == [70, 90]
    conditioning = dict(sorted({**kept, **refined}.items()))
    assert sam3.first_influenced(conditioning, baseline, refined, 0, 100, keep, 15, selection) == first


@pytest.mark.parametrize("selection", [False, True])
@pytest.mark.parametrize("keep", [2, 4])
def test_a_refined_birth_is_the_first_influenced_frame(selection, keep):
    """Born on 5, frames 5 and 6 refined: the anchor on 20 lies 14 from 6 and is demoted, 50 is kept.
    The birth reads its own refine, so the second pass's output covers every frame from it on."""
    baseline = outputs(5, 20, 50)
    refined = {5: "refined", 6: "refined"}
    kept, demoted = sam3.demote(baseline, refined)
    assert kept == outputs(50) and demoted == [20]
    conditioning = dict(sorted({**kept, **refined}.items()))
    assert sam3.first_influenced(conditioning, baseline, refined, 5, 60, keep, 15, selection) == 5


def test_the_second_pass_reads_its_own_frames_as_prompt_mode_reads_its_propagated_ones():
    stored = stored_outputs({1: 0.5, 3: 0.5, 5: 0.0, 6: 0.5, 7: 0.0, 8: 0.5})
    cond = {2: "c2", 4: "c4", 20: "c20"}
    for selection in (False, True):
        view = sam3.pass_two_view(cond, stored, 9, 2, 3, selection)
        assert view["cond_frame_outputs"] == {4: "c4", 20: "c20"}
        prompt = sam3.memory_view({"cond_frame_outputs": {}, "non_cond_frame_outputs": stored}, 9, 3)
        assert view["non_cond_frame_outputs"] == (prompt["non_cond_frame_outputs"] if selection else {2: "c2", **stored})


# --- the capture: what segment_by_prompt hands over, read-only ----------------------------------
# The scripted run of test_sam3_1_multiplex_ab: the person born on frame 2, re-anchored on 16 and 32.

def prompt_run(rig, monkeypatch, cfg, make=FakeTracker, capture=None, logits=None):
    """segment_by_prompt on the stand-ins the rig installed, with a fresh tracker from `make`:
    (masks, log, result)."""
    tracker = make()
    monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (None, None, tracker, None))
    result = {}
    masks = sam3.segment_by_prompt(FakeModel(), object(), torch.zeros(RIG_N, RIG_H, RIG_W, 3), "p", cfg,
                                   result=result, logits=logits, capture=capture)
    return masks, tracker.log, result


def test_the_capture_changes_nothing_segment_by_prompt_does(rig, monkeypatch):
    cfg = config()
    base, base_log, base_result = rig(cfg, ring=2, speck=True)
    for capture, dump in (({"raw": {}}, None), ({"raw": {}}, {}), (None, {})):
        masks, log, result = prompt_run(rig, monkeypatch, cfg, lambda: FakeTracker(ring=2, speck=True), capture, dump)
        assert torch.equal(masks, base) and log == base_log and result == base_result


def test_the_capture_holds_the_track_s_birth_its_conditioning_frames_as_created_and_its_raw_logits(rig, monkeypatch):
    cfg = config()   # max_conditioning_frames 2 with the birth kept: pass 1 drops the anchor on 16 at 32
    dump = {}
    rig(cfg, logits=dump)
    capture = {"raw": {}}
    prompt_run(rig, monkeypatch, cfg, capture=capture)
    assert capture["birth"] == 2 and sorted(capture["cond"]) == [2, 16, 32]
    assert all("pred_masks_high_res" not in out and "maskmem_features" in out for out in capture["cond"].values())
    assert sorted(capture["raw"]) == list(range(3, RIG_N))     # every propagated frame, anchors included
    for f, raw in capture["raw"].items():
        assert raw.device.type == "cpu" and dump["raw"][f] and torch.equal(raw[0, 0].to(torch.float16), dump["logits"][f])


def test_a_false_start_resets_the_capture(rig, monkeypatch):
    rig(config(), **PROBATION)
    capture = {"raw": {}}
    _, _, result = prompt_run(rig, monkeypatch, config(), lambda: FakeTracker(empty=range(3, 10)), capture)
    assert result["false starts"] == 1
    assert capture["birth"] == 12 and sorted(capture["cond"]) == [12, 16, 32] and min(capture["raw"]) == 13


# --- the mode on the scripted tracker --------------------------------------------------------------

class Backbone:
    """The vision backbone as prompt_pose reads it: the scripted tracker makes the features itself,
    so only the trunk the second pass times is here."""
    def __init__(self):
        self.trunk = torch.nn.Identity()


class Run:
    """What pp_rig's run returns."""


def refined_box(real):
    """The stand-in refine's mask logits on frame `real`: the person's box, two rows taller."""
    return box(*position(real), h=10, ring=1)[None, None]


@pytest.fixture
def pp_rig(rig, monkeypatch):
    """segment_by_prompt_pose on test_sam3_1_multiplex_ab's stand-ins, the refine stood in for (the
    person's box two rows taller) and, when `chosen` is given, the rule too. `make` builds the
    tracker; prompt mode is run on one of its own first, as the baseline. `size` is the (H, W) of
    the frames prompt_pose is given (the baseline stays on the rig's)."""
    def run(cfg=None, chosen=None, metas=None, logits=None, make=None, size=(RIG_H, RIG_W), **rig_kwargs):
        cfg = cfg or config()
        tracker_kwargs = {k: rig_kwargs.pop(k) for k in ("ring", "speck", "scores", "empty") if k in rig_kwargs}
        make = make or (lambda: FakeTracker(**tracker_kwargs))
        out = Run()
        out.prompt, out.prompt_log, out.prompt_result = rig(cfg, tracker=make(), **rig_kwargs)
        out.tracker = tracker = make()
        monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (None, None, tracker, Backbone()))
        detect, out.detected = sam3.detect_person, []
        monkeypatch.setattr(sam3, "detect_person", lambda *a: out.detected.append(a[2]) or detect(*a))
        out.refines = []

        def refine(tracker, backbone, frame, trunk_out, vision_feats, vision_pos, feat_sizes, points, mux, previous):
            real = vision_feats[0]
            out.refines.append({"frame": real, "points": points, "previous": previous, "log": len(tracker.log)})
            logits_ = refined_box(real)
            return ({"pred_masks": logits_, "pred_masks_high_res": logits_, "object_score_logits": torch.tensor([[7.0]]),
                     "obj_ptr": 0, "maskmem_features": ("cond", "refined", real), "maskmem_pos_enc": [0]},
                    {"points": list(points)[:16], "stability": 0.99, "fallback": False})

        monkeypatch.setattr(sam3, "refine_with_points", refine)
        if chosen is not None:
            monkeypatch.setattr(sam3, "refine_points", lambda *args: dict(chosen))
        metas = metas or [meta({}, size[1], size[0])] * RIG_N
        out.result = {}
        out.masks = sam3.segment_by_prompt_pose(FakeModel(), object(), torch.zeros(RIG_N, *size, 3), "p", cfg,
                                                metas, 0.5, [NOTHING] * RIG_N, result=out.result, logits=logits)
        out.log = tracker.log
        out.second = tracker.log[out.refines[0]["log"]:] if out.refines else []    # the second pass's calls
        return out
    return run


def at_defaults(best=None, **tracker_kwargs):
    """pp_rig's arguments for the config defaults (test_sam3_1_multiplex_ab's defaults run): frames
    in -1..1 read back by the tracker, a propagation decoder for the best_iou pointer and memory
    selection (on frame f it selects mask best.get(f, 0)), and core's MultiplexState."""
    return dict(make=lambda: DefaultsTracker(signed=True, best=best or {}, **tracker_kwargs), prep=range_prep,
                core_mux=True)


def tracked(entries):
    """{frame: the conditioning frames it read} of the ("track", ...) entries of a forward pass."""
    return {e[1]: e[3] for e in entries if e[0] == "track"}


def shown(logits_, cfg):
    return sam3.to_frame_size(sam3.clean_channel_logits(logits_, cfg.fill_hole_area), RIG_H, RIG_W)


@pytest.mark.parametrize("defaults", [False, True])
def test_with_no_frame_to_refine_the_result_is_prompt_mode_s_tensor(pp_rig, caplog, monkeypatch, defaults):
    returned = []
    by_prompt = sam3.segment_by_prompt
    monkeypatch.setattr(sam3, "segment_by_prompt", lambda *a, **k: returned.append(by_prompt(*a, **k)) or returned[-1])
    with caplog.at_level("INFO"):
        out = pp_rig(sam3.SAM3Config() if defaults else config(), **(at_defaults() if defaults else {}))
    assert out.masks is returned[-1]                                     # pass 1's tensor itself
    assert torch.equal(out.masks, out.prompt) and out.log == out.prompt_log and out.result == out.prompt_result
    assert out.refines == [] and out.detected == list(range(RIG_N))      # no refine, no second pass
    assert "prompt_pose: no frame needed points; the mask is prompt mode's" in caplog.text


@pytest.mark.parametrize("defaults", [False, True])
def test_the_second_pass_tracks_again_from_the_birth_around_the_kept_and_refined_frames(pp_rig, defaults):
    """Frame 36 refined: the anchor on 32 is within 16 of it and is demoted; the birth (2) and the
    anchor on 16 are kept - on the earlier baseline 16 although pass 1 had let it go at 32
    (max_conditioning_frames 2). The second pass tracks every other frame from the birth on with the
    tracker alone, each reading the max_conditioning_frames conditioning frames closest to it on both
    sides (2 of the 3; at the defaults' 4, all three), its memory from the decoder's raw logits. At
    the defaults it reads its frames in -1..1, builds each pointer from the best-IoU mask's token and
    ranks its memory (memory selection); the baseline reads its memory by distance."""
    cfg = sam3.SAM3Config() if defaults else config()
    rig_kwargs = at_defaults(speck=True, best={20: 2}) if defaults else dict(speck=True)
    out = pp_rig(cfg, chosen={36: [right_hand(0)]}, **rig_kwargs)
    assert [r["frame"] for r in out.refines] == [36]
    retracked = [f for f in range(3, RIG_N) if f not in (16, 36)]
    reads = tracked(out.second)
    assert list(reads) == retracked
    assert (reads[3], reads[20], reads[32], reads[37]) == \
        (((2, 16, 36),) * 4 if defaults else ((2, 16), (16, 36), (16, 36), (16, 36)))
    # frame 18 reads 17 and 15-12 by distance (16 is a conditioning frame); ranked, 11 fills the slot 16 leaves
    assert out.tracker.reads[18] == ((17, 15, 14, 13, 12, 11) if defaults else (17, 15, 14, 13, 12))
    if defaults:   # every frame the backbone was given in -1..1; frame 21 reads mask 2's token (3), not token 0's
        assert all(value == float(torch.tensor(real / 100) * 2 - 1) for real, value in out.tracker.seen)
        assert pointers(out.second)[21] == 3.0
    assert out.detected == list(range(RIG_N))                            # the detector ran in pass 1 only
    encodes = [e for e in out.second if e[0] == "encode"]
    assert [e[1] for e in encodes] == retracked
    assert {e[2] for e in encodes} <= {60, 68}                            # the raw box: 56 or 64 px and the 4 px speck
    assert {e[2] for e in out.prompt_log if e[0] == "encode"} <= {56, 64}  # pass 1 encodes the cleaned mask
    for f in (0, 1, 2, 16):                                               # before the birth, and the kept frames
        assert torch.equal(out.masks[f], out.prompt[f]), f
    assert torch.equal(out.masks[36], shown(refined_box(36), cfg))
    assert not torch.equal(out.masks[36], out.prompt[36])
    assert out.result["refined frames"] == 1 and out.result["demoted"] == 1 and out.result["re-tracked"] == 35
    assert out.result["frames segmented"] == RIG_N


class Drifts(FakeTracker):
    """The scripted tracker, whose mask on a frame it tracks a second time, the second pass's, is
    the person's box one column to the right (drifted): the second pass's drift from the first."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visited = set()

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, *args, **kwargs):
        out = super().track_step(frame_idx, is_init_cond_frame, current_vision_feats, *args, **kwargs)
        real = current_vision_feats[0]
        if real in self.visited:
            out["pred_masks"] = out["pred_masks_high_res"] = drifted(real)
        self.visited.add(real)
        return out


def drifted(real):
    y, x = position(real)
    return box(y, x + 1, ring=1)[None, None]


def test_the_frames_the_refine_cannot_reach_keep_the_first_pass(pp_rig, caplog):
    """Frame 36 refined, max_conditioning_frames 2: the anchor on 32 is demoted, frames 2-16 read
    the birth and the anchor on 16 as they would with no refine, and 17 is the first to read 36 (16
    before it, 36 after). The second pass still tracks from the birth, but on a tracker whose second
    pass drifts, frames 0-16 show pass 1's masks bit for bit, and from 17 on every frame but the
    refined one shows the second pass."""
    cfg = config()
    with caplog.at_level("INFO"):
        out = pp_rig(cfg, chosen={36: [right_hand(0)]}, make=Drifts)
    assert list(tracked(out.second)) == [f for f in range(3, RIG_N) if f not in (16, 36)]
    for f in range(17):
        assert torch.equal(out.masks[f], out.prompt[f]), f
    for f in range(17, RIG_N):
        expected = shown(refined_box(36) if f == 36 else drifted(f), cfg)
        assert torch.equal(out.masks[f], expected) and not torch.equal(out.masks[f], out.prompt[f]), f
    assert out.result["kept from the first pass"] == 17
    assert "prompt_pose: frames 0-16 keep the first pass: the refine cannot reach them" in caplog.text


def test_a_refine_every_frame_reads_keeps_only_the_frames_before_the_birth(pp_rig, caplog):
    """At the defaults' max_conditioning_frames 4 the birth, the anchor on 16 and the refine on 36
    are all read from the birth on: only frames 0 and 1, tracked backwards, keep the first pass."""
    with caplog.at_level("INFO"):
        out = pp_rig(sam3.SAM3Config(), chosen={36: [right_hand(0)]}, **at_defaults())
    assert out.result["kept from the first pass"] == 2
    assert "prompt_pose: frames 0-1 keep the first pass: the refine cannot reach them" in caplog.text


def test_a_demoted_anchor_pass_1_had_let_go_is_tracked_again(pp_rig):
    out = pp_rig(config(), chosen={20: [right_hand(0)]})
    reads = tracked(out.second)
    assert 16 in reads and 32 in reads and 2 not in reads                 # 16 and 32 demoted, the birth kept
    assert (reads[16], reads[25]) == ((2, 20), (2, 20))


def test_refined_frames_stay_conditioning_frames_whatever_their_distance(pp_rig):
    out = pp_rig(config(), chosen={20: [right_hand(0)], 30: [right_hand(0)]})
    reads = tracked(out.second)
    assert 20 not in reads and 30 not in reads and reads[25] == (20, 30)
    assert out.result["refined frames"] == 2 and out.result["demoted"] == 2   # 16 and 32


def test_adjacent_refined_frames_are_each_refined_and_read_as_conditioning_frames(pp_rig, rig):
    """Frames 20, 21 and 22 refined, each from its own points (the cap is per refine) and pass 1's
    raw logits of its frame: the anchors on 16 and 32 lie within 16 of them and are demoted, the
    birth (2) is kept. Keeping 2, the frames before 20 read the birth and 20, those after 22 read 21
    and 22; the birth already reads 20, so every frame from it on shows the second pass."""
    cfg = config()
    dump = {}
    rig(cfg, logits=dump)
    chosen = {20: [right_hand(j) for j in range(20)], 21: [right_hand(j) for j in range(3)], 22: [right_hand(0)]}
    out = pp_rig(cfg, chosen=chosen)
    assert [(r["frame"], len(r["points"])) for r in out.refines] == [(20, 20), (21, 3), (22, 1)]
    for refine in out.refines:
        assert torch.equal(refine["previous"][0, 0].to(torch.float16), dump["logits"][refine["frame"]])
    retracked = [f for f in range(3, RIG_N) if f not in (20, 21, 22)]
    reads = tracked(out.second)
    assert list(reads) == retracked
    assert {reads[f] for f in range(3, 20)} == {(2, 20)} and {reads[f] for f in range(23, RIG_N)} == {(21, 22)}
    for g in (20, 21, 22):
        assert torch.equal(out.masks[g], shown(refined_box(g), cfg))
    assert (out.result["refined frames"], out.result["points"], out.result["demoted"], out.result["re-tracked"],
            out.result["kept from the first pass"]) == (3, 16 + 3 + 1, 2, 34, 2)


def test_a_refine_on_the_birth_starts_from_the_mask_pass_1_conditioned_it_with(pp_rig, rig, caplog):
    """Frame 2, the birth, refined: pass 1 did not propagate it, so the refine gets the birth's
    conditioning logits, the mask pass 1 shows there (Meta's refine looks the frame's conditioning
    output up). The refine replaces the birth's conditioning; the anchor on 16, 14 frames away, is
    demoted, 32 is kept. The birth reads its own refine: on a tracker whose second pass drifts,
    frames 0 and 1 (tracked backwards) keep pass 1's mask, the birth shows the refine, the anchor
    on 32 pass 1's mask and every other frame the second pass."""
    cfg = config()
    dump = {}
    rig(cfg, logits=dump)
    record = {}
    with caplog.at_level("INFO"):
        out = pp_rig(cfg, chosen={2: [right_hand(0)]}, make=Drifts, logits=record)
    (refine,) = out.refines
    assert refine["frame"] == 2 and not dump["raw"][2]
    assert torch.equal(refine["previous"][0, 0].to(torch.float16), dump["logits"][2])
    assert set(refine["previous"].unique().tolist()) == {-10.0, 10.0}
    reads = tracked(out.second)
    assert list(reads) == [f for f in range(3, RIG_N) if f != 32] and set(reads.values()) == {(2, 32)}
    for f in (0, 1, 32):
        assert torch.equal(out.masks[f], out.prompt[f]), f
    assert torch.equal(out.masks[2], shown(refined_box(2), cfg)) and not torch.equal(out.masks[2], out.prompt[2])
    for f in (f for f in range(3, RIG_N) if f != 32):
        assert torch.equal(out.masks[f], shown(drifted(f), cfg)), f
    assert (out.result["refined frames"], out.result["demoted"], out.result["re-tracked"],
            out.result["kept from the first pass"]) == (1, 1, 36, 2)
    assert "prompt_pose: frames 0-1 keep the first pass: the refine cannot reach them" in caplog.text
    assert "conditioning frames demoted 16, kept 32" in caplog.text
    assert record["cut"][2] == "prompt" and record["raw"][2]
    for f in range(RIG_N):
        again = reproduce(record["logits"][f], "prompt", RIG_H, RIG_W, 0.0, None, 0, cfg, record["raw"][f])
        assert torch.equal(again, out.masks[f]), f


@pytest.mark.parametrize("defaults", [False, True])
def test_every_refine_gets_the_first_pass_s_raw_logits_of_its_frame(pp_rig, rig, defaults):
    """No switch: prompt_pose's capture keeps pass 1's raw logits, and each refine is handed those
    of its own frame, a CPU copy of the decoder's logits before the cleaning (the tracker's 2x2
    speck still in them). The refine clamps them to +/-32 (tests/models/test_sam3_1_multiplex_refine.py)."""
    cfg = sam3.SAM3Config() if defaults else config()
    kwargs = at_defaults(speck=True) if defaults else {}
    make = kwargs.pop("make", lambda: FakeTracker(speck=True))
    dump = {}
    rig(cfg, logits=dump, tracker=make(), **kwargs)
    out = pp_rig(cfg, chosen={20: [right_hand(0)], 36: [right_hand(0)]}, make=make, **kwargs)
    assert [r["frame"] for r in out.refines] == [20, 36]
    for refine in out.refines:
        g, previous = refine["frame"], refine["previous"]
        assert previous.device.type == "cpu" and previous.shape == (1, 1, 16, 16) and dump["raw"][g]
        assert torch.equal(previous[0, 0].to(torch.float16), dump["logits"][g])
        assert (previous[0, 0][SPECK] == 5.0).all()                        # before the cleaning


LOSES = 20


class LosesTheArm(FakeTracker):
    """The scripted tracker, whose mask on the frames `losing` (LOSES) drops the right part of the
    person (the low-res columns from x + `kept` on, x + 5 where the three right-hand keypoints below
    lie)."""
    losing = (LOSES,)
    kept = 5

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, *args, **kwargs):
        out = super().track_step(frame_idx, is_init_cond_frame, current_vision_feats, *args, **kwargs)
        real = current_vision_feats[0]
        if real in self.losing:
            _, x = position(real)
            out["pred_masks"][0, 0, :, x + self.kept:] = -10.0
        return out


def hand_on_the_person(real):
    """Three right-hand keypoints near the right edge of the person's box on frame `real`, in frame
    pixels (the box's low-res columns x..x+7 are frame columns 2x..2x+15)."""
    _, x = position(real)
    return {right_hand(j): (2 * x + 13.5, 12.5 + 2 * j) for j in range(3)}


def test_the_rule_on_a_tracked_clip_refines_the_one_frame_that_lost_the_hand(pp_rig):
    metas = [meta(hand_on_the_person(f), RIG_W, RIG_H) for f in range(RIG_N)]
    out = pp_rig(config(), metas=metas, make=LosesTheArm)
    (refine,) = out.refines
    points = hand_on_the_person(LOSES)
    # the points reach the decoder at x * 1008, y * 1008 of pose_data's normalised position
    assert refine["frame"] == LOSES
    assert refine["points"] == pytest.approx([(x / RIG_W * 1008, y / RIG_H * 1008) for x, y in points.values()])
    assert (out.result["refined frames"], out.result["points"], out.result["demoted"], out.result["re-tracked"]) == \
        (1, 3, 2, 36)


class LosesTheArmForThreeFrames(LosesTheArm):
    losing = (20, 21, 22)


def test_the_rule_on_a_tracked_clip_refines_each_frame_of_a_loss_over_several(pp_rig):
    metas = [meta(hand_on_the_person(f), RIG_W, RIG_H) for f in range(RIG_N)]
    out = pp_rig(config(), metas=metas, make=LosesTheArmForThreeFrames)
    assert [r["frame"] for r in out.refines] == [20, 21, 22]
    for refine in out.refines:
        points = hand_on_the_person(refine["frame"]).values()
        assert refine["points"] == pytest.approx([(x / RIG_W * 1008, y / RIG_H * 1008) for x, y in points])
    assert (out.result["refined frames"], out.result["points"], out.result["demoted"], out.result["re-tracked"]) == \
        (3, 9, 2, 34)


def test_a_pose_the_mask_holds_on_every_frame_refines_nothing(pp_rig, monkeypatch):
    """The rule itself, on a pose whose hand the mask holds on every frame from the birth to the
    last: no frame is refined, and the result is pass 1's tensor itself."""
    returned = []
    by_prompt = sam3.segment_by_prompt
    monkeypatch.setattr(sam3, "segment_by_prompt", lambda *a, **k: returned.append(by_prompt(*a, **k)) or returned[-1])
    metas = [meta(hand_on_the_person(f), RIG_W, RIG_H) for f in range(RIG_N)]
    out = pp_rig(config(), metas=metas)
    assert out.refines == [] and out.masks is returned[-1] and torch.equal(out.masks, out.prompt)
    assert out.log == out.prompt_log and out.result == out.prompt_result


def test_the_logits_record_reproduces_prompt_pose_s_masks(pp_rig):
    cfg = config()
    dump = {}
    out = pp_rig(cfg, chosen={36: [right_hand(0)]}, logits=dump, ring=2, speck=True)
    assert (dump["cut"][2], dump["cut"][16], dump["cut"][32], dump["cut"][36]) == ("birth", "anchor", "prompt", "prompt")
    assert set(dump["cut"]) == {"prompt", "birth", "anchor"}
    assert [f for f in range(RIG_N) if not dump["raw"][f]] == [2]        # the kept birth shows its conditioning mask
    assert [a["frame"] for a in dump["anchors"] if a["fired"]] == [16, 32]  # the first pass's slots
    for f in range(RIG_N):
        again = reproduce(dump["logits"][f], "prompt", RIG_H, RIG_W, 0.0, None, 0, cfg, dump["raw"][f])
        assert torch.equal(again, out.masks[f]), f


def test_the_sink_names_the_mode_and_gets_the_record(pp_rig):
    cfg = config()
    got = {}
    pp_rig(cfg, chosen={36: [right_hand(0)]})   # installs the stand-ins
    masks = sam3.track((FakeModel(), object()), torch.zeros(RIG_N, RIG_H, RIG_W, 3),
                       pose_data=pose_data([meta({}, RIG_W, RIG_H)] * RIG_N), mode="prompt_pose", config=cfg,
                       logits_sink=lambda logits_, info: got.update(logits=logits_, info=info))
    info = got["info"]
    assert info["mode"] == "prompt_pose" and info["raw"][36] and info["cut"][36] == "prompt"
    assert info["fill_hole_area"] == cfg.fill_hole_area and "anchors" in info
    assert torch.equal(masks[36], shown(refined_box(36), cfg))


# The mask-driven trigger on the tracked clip, on 512 x 512 frames (32 px per low-res pixel), where
# the final's 32 px blocks fit: on LOSES the tracker drops the person's low-res columns from x + 4
# on, and the frames around it hold them. Her right elbow and wrist are drawn on her box, the wrist
# on the columns dropped; on LOSES the pose loses the wrist too.
BIG = (512, 512)


class DropsTheHand(LosesTheArm):
    kept = 4


def forearm_on_the_person(real, lost=()):
    """Her right elbow and wrist on frame `real` in 512 x 512 pixels, the wrist under the draw
    threshold on `lost`."""
    y, x = position(real)
    elbow, wrist = ((x + 2) * 32 + 16, 200.5), ((x + 6) * 32 + 16, 200.5)
    return meta({sam3.R_ELBOW: elbow, sam3.R_WRIST: (*wrist, 0.1 if real in lost else 0.9)}, *BIG[::-1])


def first_pass(monkeypatch):
    """A list that gets, on every run of segment_by_prompt (the baseline's, then prompt_pose's pass 1),
    a copy of its masks and the tensor itself."""
    got = []
    by_prompt = sam3.segment_by_prompt

    def keep(*a, **k):
        masks = by_prompt(*a, **k)
        got.append((masks.clone(), masks))
        return masks

    monkeypatch.setattr(sam3, "segment_by_prompt", keep)
    return got


def test_the_mask_trigger_on_a_tracked_clip_refines_the_frame_that_dropped_the_region(pp_rig, monkeypatch, caplog):
    passes = first_pass(monkeypatch)
    cfg = config()
    metas = [forearm_on_the_person(f, lost=(LOSES,)) for f in range(RIG_N)]
    with caplog.at_level("INFO"):
        out = pp_rig(cfg, metas=metas, make=DropsTheHand, size=BIG)
    (refine,) = out.refines
    assert refine["frame"] == LOSES and 1 <= len(refine["points"]) <= 16
    pass_1, _ = passes[-1]
    for x, y in refine["points"]:
        # each point, back in frame pixels, is on the region: held on 19 and 21, dropped on 20
        col, row = int(x / 1008 * BIG[1]), int(y / 1008 * BIG[0])
        assert pass_1[LOSES - 1, row, col] > 0 and pass_1[LOSES + 1, row, col] > 0 and pass_1[LOSES, row, col] == 0
    assert (f"prompt_pose: frame {LOSES} refined from {len(refine['points'])} point(s) in a region the mask dropped "
            f"(held on frames {LOSES - 1} and {LOSES + 1}), with the first pass mask") in caplog.text
    # the second pass: the birth kept, the anchors on 16 and 32 demoted, every other frame from 3 re-tracked
    assert list(tracked(out.second)) == [f for f in range(3, RIG_N) if f != LOSES]
    refined = sam3.clean_channel_logits(refined_box(LOSES), cfg.fill_hole_area)
    assert torch.equal(out.masks[LOSES], sam3.to_frame_size(refined, *BIG))
    assert (out.result["refined frames"], out.result["refined for a dropped region"], out.result["points"],
            out.result["demoted"], out.result["re-tracked"]) == (1, 1, len(refine["points"]), 2, 36)


def test_the_two_triggers_join_their_frames_and_a_frame_both_pick_gets_the_pose_points_first(pp_rig, caplog):
    metas = [forearm_on_the_person(f, lost=(LOSES,)) for f in range(RIG_N)]
    # three right-hand keypoints on the part of her the mask keeps: the pose rule alone picks nothing
    metas[LOSES] = meta({**{right_hand(j): (200.5 + j, 300.5) for j in range(3)}, sam3.R_ELBOW: (208.5, 200.5),
                         sam3.R_WRIST: (336.5, 200.5, 0.1)}, *BIG[::-1])
    (alone,) = pp_rig(config(), metas=metas, make=DropsTheHand, size=BIG).refines
    assert alone["frame"] == LOSES
    with caplog.at_level("INFO"):
        out = pp_rig(config(), chosen={LOSES: [right_hand(j) for j in range(3)], 36: [right_hand(0)]}, metas=metas,
                     make=DropsTheHand, size=BIG)
    assert [r["frame"] for r in out.refines] == [LOSES, 36]
    # the pose's points first, then the region's (past 16 the refine keeps the first 8 and the last 8)
    pose = [((200.5 + j) / 512 * 1008, 300.5 / 512 * 1008) for j in range(3)]
    assert out.refines[0]["points"] == pytest.approx(pose + alone["points"])
    assert ("point(s) on the right forearm and hand and in a region the mask dropped (held on frames "
            f"{LOSES - 1} and {LOSES + 1})") in caplog.text
    assert out.result["refined frames"] == 2 and out.result["refined for a dropped region"] == 1


def test_a_clip_whose_mask_drops_nothing_is_prompt_mode_s_tensor_with_the_body_drawn(pp_rig, monkeypatch, caplog):
    passes = first_pass(monkeypatch)
    metas = [forearm_on_the_person(f) for f in range(RIG_N)]
    with caplog.at_level("INFO"):
        out = pp_rig(config(), metas=metas, size=BIG)
    pass_1, returned = passes[-1]
    assert out.refines == [] and out.masks is returned and torch.equal(out.masks, pass_1)
    assert out.log == out.prompt_log and "refined for a dropped region" not in out.result
    assert "prompt_pose: no frame needed points; the mask is prompt mode's" in caplog.text


# --- the entry: what prompt_pose mode reads and ignores -------------------------------------------

@pytest.fixture
def stub(monkeypatch):
    """segment_by_prompt_pose / segment_by_prompt / segment_by_pose stand-ins that record what they
    were handed."""
    calls = []

    def by_prompt_pose(model, clip, images, prompt, config, pose_metas, draw_threshold, hidden, result=None, **kwargs):
        calls.append(("prompt_pose", prompt, pose_metas, draw_threshold, hidden))
        return torch.zeros(images.shape[:3])

    monkeypatch.setattr(sam3, "segment_by_prompt_pose", by_prompt_pose)
    monkeypatch.setattr(sam3, "segment_by_prompt", lambda model, clip, images, *a, **k: torch.zeros(images.shape[:3]))
    monkeypatch.setattr(sam3, "segment_by_pose", lambda model, images, *a, **k: torch.zeros(images.shape[:3]))
    return calls


def two_frames(**rules):
    frames = [meta(HAND, W, H)] * 2
    data = pose_data(frames, 0.4, **rules)
    data["detections"] = [{"bbox": [1.0, 1.0, 7.0, 7.0], "score": 0.9, "persons": 1}] * 2
    return data


def not_used(caplog):
    return [r.getMessage() for r in caplog.records if "not used" in r.getMessage()]


def test_prompt_pose_needs_pose_data_and_a_prompt(stub):
    with pytest.raises(ValueError, match="prompt_pose mode adds points from the pose; connect pose_data"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), mode="prompt_pose")
    with pytest.raises(ValueError, match="empty"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), pose_data=two_frames(), mode="prompt_pose", prompt=" ")
    with pytest.raises(ValueError, match="covers 2 pose frames, the images are 3"):
        sam3.track((None, None), torch.zeros(3, 8, 8, 3), pose_data=two_frames(), mode="prompt_pose")


def test_prompt_pose_reads_the_pose_and_the_prompt_and_ignores_the_rest_in_one_line(stub, caplog):
    cfg = sam3.SAM3Config(birth_threshold=0.6, clear_on_anchor=True, reseed_interval=5, assoc_iou=0.2,
                          pose_point_distance=0.1)
    data = two_frames(forearm_limit=2.0)
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), pose_data=data, bboxes=[2, 2, 6, 6],
                   positive_coords='[{"x": 1, "y": 1}]', negative_coords="[]", mode="prompt_pose", prompt="a dog",
                   max_objects=3, object_index=2, config=cfg)
    (line,) = not_used(caplog)
    for name in ("bboxes", "positive_coords", "negative_coords", "max_objects 3", "object_index 2",
                 "sam3_config.reseed_interval", "sam3_config.assoc_iou"):
        assert name in line, name
    for read in ("birth_threshold", "clear_on_anchor", "pose_point_distance", "prompt 'a dog'"):
        assert read not in line, read
    ((kind, prompt, metas, threshold, hidden),) = stub
    assert (kind, prompt, threshold) == ("prompt_pose", "a dog", 0.4) and metas is data["pose_metas_original"]
    assert hidden == sam3.prompt_pose_inputs(data, 2)[2]


@pytest.mark.parametrize("mode", ["prompt", "box_keypoint"])
def test_the_other_modes_name_a_changed_prompt_pose_field(stub, caplog, mode):
    cfg = sam3.SAM3Config(pose_point_distance=0.1)
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), pose_data=two_frames() if mode == "box_keypoint" else None,
                   mode=mode, config=cfg)
    (line,) = not_used(caplog)
    assert "sam3_config.pose_point_distance" in line


def test_the_prompt_pose_field_comes_last_and_leaves_every_other_default():
    fields = dataclasses.fields(sam3.SAM3Config)
    assert [f.name for f in fields if f.metadata["tooltip"].startswith("[prompt_pose] ")] == ["pose_point_distance"]
    distance = fields[-1]
    assert distance.name == "pose_point_distance" and "experimental" not in distance.metadata
    assert (distance.default, distance.metadata["min"], distance.metadata["max"], distance.metadata["step"]) == \
        (0.07, 0.0, 0.5, 0.005)
    assert sam3.MODES == ("prompt", "box_keypoint", "prompt_pose") and sam3.MODE_PROMPT_POSE == "prompt_pose"
