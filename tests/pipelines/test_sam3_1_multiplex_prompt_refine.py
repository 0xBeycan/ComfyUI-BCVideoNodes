"""SAM 3.1 Multiplex prompt mode's refine: the detection that finds a part of her the mask drops for a
few frames from the mask alone (guard.mask.dropped_parts, each criterion on synthetic masks), and
prompt mode on the scripted tracker of test_sam3_1_multiplex_ab (segment_by_prompt_refined, the
refine stood in for as in test_sam3_1_multiplex_prompt_pose): the frame it refines, its points, the
second pass it shares with prompt_pose, the track's tensor itself where nothing is dropped, and
prompt_pose, which does not run it. No model is loaded.

Needs ComfyUI importable (the ComfyUI root on PYTHONPATH), like the other SAM 3.1 Multiplex tests:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/pipelines/test_sam3_1_multiplex_prompt_refine.py
"""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import sam3  # noqa: E402
from test_sam3_1_multiplex_ab import FakeModel, FakeTracker, config, reproduce, rig  # noqa: E402,F401
from test_sam3_1_multiplex_ab import H as RIG_H  # noqa: E402
from test_sam3_1_multiplex_ab import N as RIG_N  # noqa: E402
from test_sam3_1_multiplex_ab import W as RIG_W  # noqa: E402
from test_sam3_1_multiplex_prompt_pose import (BIG, LOSES, Backbone, DropsTheHand, Run, at_defaults,  # noqa: E402,F401
                                               forearm_on_the_person, pp_rig, refined_box, stand_in_refine, tracked)
from test_sam3_1_multiplex_prompt_pose import first_pass as track_spy  # noqa: E402


# --- the detection: a part of her the mask drops, from the mask alone -------------------------------
# prompt_pose's dropout scene: her rectangle (400 x 200, rows and columns from 40) and a square at its
# right (columns 240-320), here given per frame as (top row, height), None where the mask drops it. On a
# frame without the square the Wan Animate final covers her rectangle grown by 10 px and cut into
# blocks, columns 30-250, so each frame around a drop of the 80 x 80 square at row 100 loses its part
# from column 250 on: rows 100-180, 80 x 70 = 5600 px, 6.5% of her 86400, holding a whole block of the
# frames' grid (columns 261-294, rows 126-158).

def parts_scene(squares, body=(400, 200)):
    """[frames, H, W] booleans of the scene: frame f holds the square at squares[f] = (top, height)."""
    rows, cols = body
    H, W = 40 + rows + 40, 40 + cols + 80 + 40
    masks = np.zeros((len(squares), H, W), bool)
    for f, square in enumerate(squares):
        masks[f, 40:40 + rows, 40:40 + cols] = True
        if square is not None:
            top, height = square
            masks[f, top:top + height, 40 + cols:40 + cols + 80] = True
    return masks


def found(masks):
    """dropped_parts on `masks`: (first and last frame of the run, the frames around it, the region's
    top-left pixel, its shape, its pixels) per region."""
    return [(run.start, run.stop - 1, anchors, corner, region.shape, int(region.sum()))
            for run, anchors, region, corner, _ in sam3.dropped_parts(masks)]


SQUARE = (100, 80)


def test_a_hand_sized_part_dropped_for_one_frame_and_held_on_both_sides_is_found():
    masks = parts_scene([SQUARE] * 3 + [None] + [SQUARE] * 3)
    assert found(masks) == [(3, 3, (2, 4), (100, 250), (80, 70), 5600)]
    (*_, share), = sam3.dropped_parts(masks)
    assert share == 5600 / 86400


def test_a_part_that_moves_between_the_two_frames_is_found_where_both_of_them_hold_it():
    # the square 40 px lower on frame 4 and lower again on 5 and 6: each of frames 2 and 4 loses its
    # 5600 px part, and the region is their overlap, rows 140-180 - held on no frame after 4, and
    # without a whole block of its own
    masks = parts_scene([SQUARE] * 3 + [None, (140, 80), (180, 80), (220, 80)])
    assert found(masks) == [(3, 3, (2, 4), (140, 250), (40, 70), 2800)]


def test_parts_that_do_not_overlap_are_not_found():
    # on frame 4 the square is 160 px lower: frames 2 and 4 lose parts that share no pixel
    assert found(parts_scene([SQUARE] * 3 + [None] + [(260, 80)] * 3)) == []


def test_each_part_must_be_hand_sized():
    # the 5600 px parts: 1.43% of her 640 x 600 rectangle and the square (under LOSS_HAND, 1.5%),
    # 1.64% of a 600 x 560 one
    squares = [SQUARE] * 3 + [None] + [SQUARE] * 3
    assert found(parts_scene(squares, body=(640, 600))) == []
    assert len(found(parts_scene(squares, body=(600, 560)))) == 1


def test_each_part_must_hold_a_whole_block():
    # on frame 4 the square is 24 px tall: its part, rows 100-124, is 2.05% of her mask but holds no
    # whole block of any grid around (their rows 94-126 and 126-158 stick out of it)
    assert found(parts_scene([SQUARE] * 3 + [None] + [(100, 24)] * 3)) == []


def test_a_part_that_moved_away_is_not_found():
    # on frames 3-6 the square is 80 px higher, next to where it was: the limb moved and came back, and
    # the mask holds it near the part on every frame of the run (the gain rule); dropped instead, the
    # same run is found
    moved = [SQUARE] * 3 + [(20, 80)] * 4 + [SQUARE] * 2
    assert found(parts_scene(moved)) == []
    dropped = [SQUARE] * 3 + [None] * 4 + [SQUARE] * 2
    assert found(parts_scene(dropped)) == [(3, 6, (2, 7), (100, 250), (80, 70), 5600)]


def test_a_run_of_up_to_eight_frames_is_found_and_an_open_one_is_not():
    assert [f[:3] for f in found(parts_scene([SQUARE] + [None] * 8 + [SQUARE] * 2))] == [(1, 8, (0, 9))]
    assert found(parts_scene([SQUARE] + [None] * 9 + [SQUARE])) == []
    assert found(parts_scene([SQUARE] * 4 + [None] * 3)) == []        # dropped to the last frame
    assert found(parts_scene([None] * 3 + [SQUARE] * 4)) == []        # from the first


def test_the_points_of_a_moving_part_lie_where_both_frames_hold_it_and_the_run_does_not():
    masks = parts_scene([SQUARE] * 3 + [None, (140, 80), (180, 80), (220, 80)])
    frames = sam3.region_frames(torch.from_numpy(masks).float(), 0, sam3.dropped_parts)
    assert list(frames) == [3] and frames[3][1] == [2, 4]
    points = frames[3][0]
    assert 2 <= len(points) <= sam3.MAX_REFINE_POINTS
    for x, y in points:
        assert 140 <= y < 180 and 250 <= x < 320, (x, y)
        assert masks[2, int(y), int(x)] and masks[4, int(y), int(x)] and not masks[3, int(y), int(x)]


# --- prompt mode on the scripted tracker ------------------------------------------------------------
# The mask-driven clip of test_sam3_1_multiplex_prompt_pose: 512 x 512 frames, the tracker drops the
# person's low-res columns from x + 4 on on frame LOSES (20), and the frames around it hold them; the
# person moves one low-res column (32 px) a frame, so the part each of frames 19 and 21 loses is a
# different one, and the region is where they overlap.

@pytest.fixture
def prompt_rig(rig, monkeypatch):
    """segment_by_prompt_refined (prompt mode) on test_sam3_1_multiplex_ab's stand-ins, the refine
    stood in for (stand_in_refine). `make` builds the tracker; the track alone (segment_by_prompt) is
    run on one of its own first, as the baseline, on the rig's frames. `size` is the (H, W) of the
    frames prompt mode is given."""
    def run(cfg=None, make=FakeTracker, size=(RIG_H, RIG_W), logits=None, **rig_kwargs):
        cfg = cfg or config()
        out = Run()
        out.prompt, out.prompt_log, out.prompt_result = rig(cfg, tracker=make(), **rig_kwargs)
        out.tracker = tracker = make()
        monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (None, None, tracker, Backbone()))
        stand_in_refine(out, monkeypatch)
        out.result = {}
        out.masks = sam3.segment_by_prompt_refined(FakeModel(), object(), torch.zeros(RIG_N, *size, 3), "p", cfg,
                                                   result=out.result, logits=logits)
        out.log = tracker.log
        out.second = tracker.log[out.refines[0]["log"]:] if out.refines else []    # the second pass's calls
        return out
    return run


def test_prompt_mode_refines_the_frame_that_dropped_a_part_of_her_and_tracks_again(prompt_rig, monkeypatch, caplog):
    passes = track_spy(monkeypatch)
    cfg = config()
    with caplog.at_level("INFO"):
        out = prompt_rig(cfg, make=DropsTheHand, size=BIG)
    (refine,) = out.refines
    assert refine["frame"] == LOSES and 1 <= len(refine["points"]) <= sam3.MAX_REFINE_POINTS
    track, _ = passes[-1]
    for x, y in refine["points"]:
        # each point, back in frame pixels, is on the region: held on 19 and 21, dropped on 20
        col, row = int(x / 1008 * BIG[1]), int(y / 1008 * BIG[0])
        assert track[LOSES - 1, row, col] > 0 and track[LOSES + 1, row, col] > 0 and track[LOSES, row, col] == 0
    assert (f"prompt: frame {LOSES} refined from {len(refine['points'])} point(s) in a region the mask dropped "
            f"(held on frames {LOSES - 1} and {LOSES + 1}), with the first pass mask") in caplog.text
    # the second pass: the birth kept, the anchors on 16 and 32 demoted, every other frame from 3 re-tracked
    assert list(tracked(out.second)) == [f for f in range(3, RIG_N) if f != LOSES]
    refined = sam3.clean_channel_logits(refined_box(LOSES), cfg.fill_hole_area)
    assert torch.equal(out.masks[LOSES], sam3.to_frame_size(refined, *BIG))
    assert not torch.equal(out.masks[LOSES], track[LOSES])
    for f in range(3):                                  # the backward fill and the kept birth
        assert torch.equal(out.masks[f], track[f]), f
    assert {k: out.result.get(k) for k in ("refined frames", "points", "stability fallbacks", "demoted", "re-tracked",
                                           "frames segmented")} == \
        {"refined frames": 1, "points": len(refine["points"]), "stability fallbacks": None, "demoted": 2,
         "re-tracked": 36, "frames segmented": RIG_N}


def test_prompt_mode_and_prompt_pose_refine_the_same_region_with_the_same_second_pass(prompt_rig, pp_rig):
    """Where the pose has her forearm in the region too, prompt_pose's mask trigger picks the same frame
    and the same points, and the one refine action gives both modes the same tensor."""
    prompt = prompt_rig(config(), make=DropsTheHand, size=BIG)
    metas = [forearm_on_the_person(f, lost=(LOSES,)) for f in range(RIG_N)]
    pose = pp_rig(config(), metas=metas, make=DropsTheHand, size=BIG)
    assert [r["frame"] for r in pose.refines] == [r["frame"] for r in prompt.refines] == [LOSES]
    assert pose.refines[0]["points"] == prompt.refines[0]["points"]
    assert torch.equal(pose.masks, prompt.masks) and pose.second == prompt.second


def test_prompt_pose_does_not_run_prompt_mode_s_refine(prompt_rig, pp_rig, monkeypatch):
    """With no body in the pose, prompt_pose refines nothing on the clip prompt mode refines: its pass
    1 is the track alone, so no region is refined twice and the clip is tracked again at most once."""
    assert [r["frame"] for r in prompt_rig(config(), make=DropsTheHand, size=BIG).refines] == [LOSES]
    passes = track_spy(monkeypatch)
    pose = pp_rig(config(), make=DropsTheHand, size=BIG)
    _, track = passes[-1]
    assert pose.refines == [] and pose.masks is track


@pytest.mark.parametrize("defaults", [False, True])
def test_with_nothing_dropped_prompt_mode_is_the_track_s_tensor(prompt_rig, monkeypatch, caplog, defaults):
    passes = track_spy(monkeypatch)
    cfg = sam3.SAM3Config() if defaults else config()
    with caplog.at_level("INFO"):
        out = prompt_rig(cfg, **(at_defaults() if defaults else {}))
    refine_lines = [line for line in caplog.text.splitlines() if "prompt:" in line]
    _, track = passes[-1]
    assert out.masks is track                                            # the track's tensor itself
    assert torch.equal(out.masks, out.prompt) and out.log == out.prompt_log and out.result == out.prompt_result
    assert out.refines == [] and refine_lines == []


def test_the_logits_record_reproduces_prompt_mode_s_masks_after_a_refine(prompt_rig, monkeypatch):
    cfg = config()
    dump = {}
    out = prompt_rig(cfg, make=DropsTheHand, size=BIG, logits=dump)
    assert dump["cut"][LOSES] == "prompt" and dump["raw"][LOSES] and dump["cut"][2] == "birth"
    for f in range(RIG_N):
        again = reproduce(dump["logits"][f], "prompt", *BIG, 0.0, None, 0, cfg, dump["raw"][f])
        assert torch.equal(again, out.masks[f]), f
    # the entry: the prompt mode track() runs is this one, and the sink gets the record
    got = {}
    masks = sam3.track((FakeModel(), object()), torch.zeros(RIG_N, *BIG, 3), config=cfg,
                       logits_sink=lambda logits_, info: got.update(info=info))
    assert got["info"]["mode"] == "prompt" and got["info"]["cut"][LOSES] == "prompt" and got["info"]["raw"][LOSES]
    assert torch.equal(masks, out.masks)
