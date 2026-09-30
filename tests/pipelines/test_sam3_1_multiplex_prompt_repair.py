"""SAM 3.1 Multiplex prompt mode's repairs of its track, from the mask alone: the rule that finds a part
lost for good (prompt.lost_for_good, each criterion on hand-written masks), the one propagation both the
backward fill and the re-track run (prompt.propagate_from), the detection of a part dropped for a few
frames (guard.mask.dropped_parts, on synthetic masks), and prompt mode on the scripted tracker of
test_sam3_1_multiplex_ab (refine.segment_by_prompt_repaired, the refine stood in for as in
test_sam3_1_multiplex_prompt_pose): the frames tracked again after a part lost for good, the union on a
refined frame, both repairs on one clip, the track's tensor itself where nothing is found, the logits
record, and prompt_pose, which runs the track alone. No model is loaded.

Needs ComfyUI importable (the ComfyUI root on PYTHONPATH), like the other SAM 3.1 Multiplex tests:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/pipelines/test_sam3_1_multiplex_prompt_repair.py
"""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import sam3  # noqa: E402
from test_sam3_1_multiplex_ab import FakeModel, FakeTracker, box, config, position, reproduce, rig  # noqa: E402,F401
from test_sam3_1_multiplex_ab import H as RIG_H  # noqa: E402
from test_sam3_1_multiplex_ab import N as RIG_N  # noqa: E402
from test_sam3_1_multiplex_ab import W as RIG_W  # noqa: E402
from test_sam3_1_multiplex_prompt_pose import (BIG, LOSES, Backbone, Run, at_defaults, pp_rig,  # noqa: E402,F401
                                               prompt_run, refined_box, stand_in_refine)
from test_sam3_1_multiplex_prompt_pose import first_pass as track_spy  # noqa: E402


# --- the rule for a part lost for good, on hand-written masks ---------------------------------------
# A 20 x 20 frame: the body, 90 px, on every frame from the birth, and the arm beside it, 10 px, on the
# frames given: the arm is 0.1 of the 100 px mask.

BODY = (slice(0, 10), slice(0, 9))
ARM = (slice(10, 12), slice(0, 5))


def scene(frames, arm_on, arm=ARM, body=BODY, empty=()):
    masks = torch.zeros(frames, 20, 20)
    for f in range(frames):
        if f in empty:
            continue
        masks[f][body] = 1.0
        if f in arm_on:
            masks[f][arm] = 1.0
    return masks


def test_a_part_lost_to_the_end_of_the_clip_is_lost_for_good():
    assert (sam3.LOST_SHARE, sam3.BACK_AREA) == (0.08, 0.92)
    assert sam3.lost_for_good(scene(12, range(6)), 0) == [(6, 11, 0.1)]


def test_the_lost_piece_must_be_8_percent_of_the_mask_and_4_connected():
    # an 8 px arm (row 10, columns 0-7) on 92 px of her: 0.08 of the mask, at the line
    masks = scene(12, range(6), arm=(slice(10, 11), slice(0, 8)))
    masks[:, 15, 0:2] = 1.0
    assert sam3.lost_for_good(masks, 0) == [(6, 11, 0.08)]
    # 10 px lost in two pieces of 5 that touch only at a corner: 0.05 each, no loss
    masks = scene(12, range(6), arm=(slice(10, 11), slice(0, 5)))
    masks[:6, 11, 5:10] = 1.0
    assert sam3.lost_for_good(masks, 0) == []
    masks[:6, 10, 5] = 1.0                                 # joined on a side: one piece of 11 px
    assert sam3.lost_for_good(masks, 0) == [(6, 11, 11 / 101)]


def test_the_area_must_stay_at_or_under_92_percent_to_the_end_of_the_stretch():
    # the arm gone from frame 6, and on frame 10 a patch of 3 px elsewhere: 93 px against frame 5's 100
    masks = scene(12, range(6))
    masks[10, 15, 0:3] = 1.0
    assert sam3.lost_for_good(masks, 0) == []
    masks[10, 15, 2] = 0.0                                 # 92 px: at the line
    assert sam3.lost_for_good(masks, 0) == [(6, 11, 0.1)]


def test_a_loss_the_area_comes_back_from_before_the_end_of_the_stretch_is_left_alone():
    # the arm gone on frames 6-9 and back from 10 on: a part that returns is not lost for good, however
    # long it is gone (a part dropped for a few frames is the refine's)
    assert sam3.lost_for_good(scene(14, set(range(6)) | set(range(10, 14))), 0) == []
    assert sam3.lost_for_good(scene(40, set(range(6)) | {39}), 0) == []


def test_the_stretch_ends_at_an_empty_frame_and_the_next_one_is_judged_on_its_own():
    # the arm gone from 6 to 9, the frames 10-11 empty, the arm back on 12-15: lost for good in the
    # first stretch, whose last frame is 9; the second stretch keeps it
    masks = scene(16, set(range(6)) | set(range(12, 16)), empty=(10, 11))
    assert sam3.lost_for_good(masks, 0) == [(6, 9, 0.1)]
    # lost again from 14 on in the second stretch: each stretch has its own
    masks = scene(20, set(range(6)) | set(range(12, 14)), empty=(10, 11))
    assert sam3.lost_for_good(masks, 0) == [(6, 9, 0.1), (14, 19, 0.1)]


def test_only_the_first_loss_of_a_stretch_counts_and_only_from_the_birth_on():
    # a second, smaller loss inside the stretch already lost belongs to the first
    masks = scene(12, range(6))
    masks[9:, 0:3, 0:9] = 0.0
    assert sam3.lost_for_good(masks, 0) == [(6, 11, 0.1)]
    # the frame before must be the birth or later: the frames before it are the backward pass's
    assert sam3.lost_for_good(scene(12, range(6)), 6) == []
    assert sam3.lost_for_good(scene(12, range(6)), 5) == [(6, 11, 0.1)]
    assert sam3.lost_for_good(scene(12, range(6)), -1) == []


# --- the one propagation: forwards to the end of a stretch, backwards to frame 0 -------------------

def test_propagate_from_tracks_forwards_under_the_frames_own_index_and_backwards_mirrored(rig):
    rig(config())                                          # installs the stand-ins
    for start, stop, order, index in ((5, 9, [6, 7, 8], lambda f: f), (5, -1, [4, 3, 2, 1, 0], lambda f: 2 * RIG_N - f)):
        tracker = FakeTracker()
        emitted = []
        sam3.propagate_from(tracker, None, None, torch.zeros(RIG_N, 3, RIG_H, RIG_W), box(*position(start))[None, None],
                            start, stop, lambda f, current, raw: emitted.append(f), torch.device("cpu"),
                            torch.float32, config())
        assert emitted == order
        (cond,) = [e for e in tracker.log if e[0] == "cond"]
        assert cond[1:3] == (index(start), start)          # the seed conditions its own frame, nothing else
        assert [(e[1], e[2], e[3]) for e in tracker.log if e[0] == "track"] == \
            [(index(f), f, (index(start),)) for f in order]


# --- the detection of a part dropped for a few frames, from the mask alone -------------------------
# Her rectangle (400 x 200, rows and columns from 40) and a square at its right (columns 240-320), given
# per frame as (top row, height), None where the mask drops it. On a frame without the square the Wan
# Animate final covers her rectangle grown by 10 px and cut into blocks, columns 30-250, so each frame
# around a drop of the 80 x 80 square at row 100 loses its part from column 250 on: rows 100-180,
# 80 x 70 = 5600 px, 6.5% of her 86400, holding a whole block of the frames' grid.

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
    # the square 40 px lower on frame 4 and lower again on 5 and 6: the region is the overlap of the two
    # frames' parts, rows 140-180
    masks = parts_scene([SQUARE] * 3 + [None, (140, 80), (180, 80), (220, 80)])
    assert found(masks) == [(3, 3, (2, 4), (140, 250), (40, 70), 2800)]


def test_parts_that_do_not_overlap_are_not_found():
    assert found(parts_scene([SQUARE] * 3 + [None] + [(260, 80)] * 3)) == []


def test_each_part_must_be_hand_sized():
    # the 5600 px parts: 1.43% of her 640 x 600 rectangle and the square (under 1.5%), 1.64% of 600 x 560
    squares = [SQUARE] * 3 + [None] + [SQUARE] * 3
    assert found(parts_scene(squares, body=(640, 600))) == []
    assert len(found(parts_scene(squares, body=(600, 560)))) == 1


def test_each_part_must_hold_a_whole_block():
    assert found(parts_scene([SQUARE] * 3 + [None] + [(100, 24)] * 3)) == []


def test_a_part_that_moved_away_is_not_found():
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
    for x, y in frames[3][0]:
        assert 140 <= y < 180 and 250 <= x < 320, (x, y)
        assert masks[2, int(y), int(x)] and masks[4, int(y), int(x)] and not masks[3, int(y), int(x)]


# --- prompt mode on the scripted tracker -------------------------------------------------------------
# At config(memory_gap=0) the scripted clip keeps its box: after each anchor the box is a row short for
# two frames only, so nothing is lost for good (at the EARLIER baseline's memory_gap 7 the anchor on 32
# holds the memory off to the clip's end, test_sam3_1_multiplex_ab). The person moves one low-res
# column a frame; the part a tracker below drops is the box's low-res columns from x + `kept` on.

def steady():
    return config(memory_gap=0)


class LosesForGood(FakeTracker):
    """The scripted tracker, whose masks from frame `lost_from` on drop the part - unless its memory has
    the frame before as a conditioning frame, as a re-track seeded there has - and on `dropped` drop it
    whatever the memory."""
    lost_from, kept, dropped = 30, 4, ()

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes,
                   mask_inputs, output_dict, num_frames, **kwargs):
        out = super().track_step(frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                                 feat_sizes, mask_inputs, output_dict, num_frames, **kwargs)
        real = current_vision_feats[0]
        reseeded = (self.lost_from - 1) in output_dict["cond_frame_outputs"]
        if frame_idx == real and (real in self.dropped or (real >= self.lost_from and not reseeded)):
            _, x = position(real)
            out["pred_masks"][0, 0, :, x + self.kept:] = -10.0
        return out


class DropsOnce(LosesForGood):
    """Drops the part on frame LOSES only."""
    lost_from, dropped = RIG_N, (LOSES,)


class DropsAndLoses(LosesForGood):
    """Drops the part on frame LOSES, loses it for good from 30 on, and after a re-track from 29 drops it
    on 33 once."""
    dropped = (LOSES,)

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes,
                   mask_inputs, output_dict, num_frames, **kwargs):
        out = super().track_step(frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                                 feat_sizes, mask_inputs, output_dict, num_frames, **kwargs)
        if current_vision_feats[0] == 33 and (self.lost_from - 1) in output_dict["cond_frame_outputs"]:
            _, x = position(33)
            out["pred_masks"][0, 0, :, x + self.kept:] = -10.0
        return out


def part(real, size, kept=LosesForGood.kept):
    """The frame pixels of the part the trackers drop on frame `real` at frame size `size` (H, W): the
    box's rows, its columns from x + kept on (a low-res pixel is size / 16 px)."""
    y, x = position(real)
    ry, rx = size[0] // 16, size[1] // 16
    return slice((y + 1) * ry, (y + 6) * ry), slice((x + kept) * rx + rx // 2, (x + 7) * rx)


@pytest.fixture
def repair_rig(rig, monkeypatch):
    """segment_by_prompt_repaired (prompt mode) on test_sam3_1_multiplex_ab's stand-ins, the refine stood
    in for (stand_in_refine). `make` builds the tracker; the track alone (segment_by_prompt) is run on one of
    its own first, as the baseline, on frames of `size` (H, W), with its logits record `out.track_dump`.
    `rig_kwargs` go to the rig (prep, core_mux)."""
    def run(cfg=None, make=FakeTracker, size=(RIG_H, RIG_W), logits=None, **rig_kwargs):
        cfg = cfg or steady()
        out = Run()
        rig(cfg, **rig_kwargs)                                # installs the stand-ins
        track_tracker = make()
        monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (None, None, track_tracker, Backbone()))
        out.track_dump = {}
        out.prompt = sam3.segment_by_prompt(FakeModel(), object(), torch.zeros(RIG_N, *size, 3), "p", cfg,
                                            result={}, logits=out.track_dump)
        out.prompt_log = list(track_tracker.log)
        out.tracker = tracker = make()
        monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (None, None, tracker, Backbone()))
        stand_in_refine(out, monkeypatch)
        out.result = {}
        out.masks = sam3.segment_by_prompt_repaired(FakeModel(), object(), torch.zeros(RIG_N, *size, 3), "p", cfg,
                                                    result=out.result, logits=logits)
        out.log = tracker.log
        out.after = tracker.log[len(out.prompt_log):]       # the tracker's calls after the track
        return out
    return run


def track_raw(out, f):
    """The track's raw logits of frame f, from its logits record, [1, 1, 16, 16]."""
    assert out.track_dump["raw"][f]
    return out.track_dump["logits"][f].float()[None, None]


def test_the_frames_after_a_part_lost_for_good_are_tracked_again_from_the_frame_before(repair_rig, caplog):
    cfg = steady()
    with caplog.at_level("INFO"):
        out = repair_rig(cfg, make=LosesForGood)
    (t, end, share), = sam3.lost_for_good(out.prompt, 2)
    assert (t, end) == (30, RIG_N - 1)
    # the second track: its own memory, seeded on 29 with the track's cleaned mask of it, then 30 to the end
    (cond,) = [e for e in out.after if e[0] == "cond"]
    seed = sam3.clean_channel_logits(track_raw(out, 29), cfg.fill_hole_area)
    assert cond[1:4] == (29, 29, int((seed > 0).sum()))
    assert [(e[1], e[3]) for e in out.after if e[0] == "track"] == [(f, (29,)) for f in range(30, RIG_N)]
    for f in range(30):
        assert torch.equal(out.masks[f], out.prompt[f]), f
    for f in range(30, RIG_N):                               # the part is back on every frame to the end
        assert not out.prompt[f][part(f, (RIG_H, RIG_W))].any() and out.masks[f][part(f, (RIG_H, RIG_W))].all(), f
    assert (f"prompt: frames 30-{RIG_N - 1} tracked again from frame 29's mask: it lost a part for good "
            f"({share:.2f} of its mask)") in caplog.text
    assert out.refines == []
    assert {k: out.result.get(k) for k in ("tracked again", "refined frames", "frames segmented")} == \
        {"tracked again": 10, "refined frames": None, "frames segmented": RIG_N}


def test_a_refined_frame_shows_its_mask_and_the_refine_s_together_and_nothing_is_tracked_again(repair_rig, caplog):
    cfg = steady()
    with caplog.at_level("INFO"):
        out = repair_rig(cfg, make=DropsOnce, size=BIG)
    (refine,) = out.refines
    assert refine["frame"] == LOSES and 1 <= len(refine["points"]) <= sam3.MAX_REFINE_POINTS
    assert torch.equal(refine["previous"], track_raw(out, LOSES).to(refine["previous"].dtype))
    for x, y in refine["points"]:
        # each point, back in frame pixels, is on the region: held on 19 and 21, dropped on 20
        col, row = int(x / 1008 * BIG[1]), int(y / 1008 * BIG[0])
        assert out.prompt[LOSES - 1, row, col] > 0 and out.prompt[LOSES + 1, row, col] > 0
        assert out.prompt[LOSES, row, col] == 0
    own = sam3.clean_channel_logits(track_raw(out, LOSES), cfg.fill_hole_area)
    refined = sam3.clean_channel_logits(refined_box(LOSES), cfg.fill_hole_area)
    assert torch.equal(out.masks[LOSES], sam3.to_frame_size(torch.maximum(own, refined), *BIG))
    union = out.masks[LOSES] > 0      # the union, at the decoder's resolution, holds both masks
    assert (union | ~(out.prompt[LOSES] > 0)).all() and (union | ~(sam3.to_frame_size(refined, *BIG) > 0)).all()
    for f in range(RIG_N):
        if f != LOSES:
            assert torch.equal(out.masks[f], out.prompt[f]), f
    assert out.after == []                                   # no demotion, no second pass: the tracker is idle
    assert (f"prompt: frame {LOSES} refined from {len(refine['points'])} point(s) in a region the mask dropped "
            f"(held on frames {LOSES - 1} and {LOSES + 1}), with the tracked mask") in caplog.text
    assert {k: out.result.get(k) for k in ("tracked again", "refined frames", "points", "frames segmented")} == \
        {"tracked again": None, "refined frames": 1, "points": len(refine["points"]), "frames segmented": RIG_N}


def test_both_repairs_on_one_clip_the_refine_reading_the_second_track(repair_rig, caplog):
    """The part dropped on 20, lost for good from 30, and dropped by the second track on 33: the frames
    30-39 are tracked again, then 20 and 33 refined, 33 from the second track's logits."""
    cfg = steady()
    with caplog.at_level("INFO"):
        out = repair_rig(cfg, make=DropsAndLoses, size=BIG)
    assert [r["frame"] for r in out.refines] == [LOSES, 33]
    # the second track's raw logits of 33: the box (8 rows, two memories read) without the part
    y, x = position(33)
    second = box(y, x, ring=1)
    second[:, x + DropsAndLoses.kept:] = -10.0
    assert torch.equal(out.refines[1]["previous"], second[None, None])
    assert torch.equal(out.refines[0]["previous"], track_raw(out, LOSES))
    own = sam3.clean_channel_logits(second[None, None], cfg.fill_hole_area)
    refined = sam3.clean_channel_logits(refined_box(33), cfg.fill_hole_area)
    assert torch.equal(out.masks[33], sam3.to_frame_size(torch.maximum(own, refined), *BIG))
    for f in list(range(30, 33)) + list(range(34, RIG_N)):
        assert out.masks[f][part(f, BIG)].all(), f
    assert "prompt: frames 30-39 tracked again from frame 29's mask" in caplog.text
    assert caplog.text.index("tracked again from frame 29") < caplog.text.index(f"prompt: frame {LOSES} refined")
    assert (out.result["tracked again"], out.result["refined frames"]) == (10, 2)


@pytest.mark.parametrize("defaults", [False, True])
def test_with_nothing_found_prompt_mode_is_the_track_s_tensor(repair_rig, monkeypatch, caplog, defaults):
    passes = track_spy(monkeypatch)
    cfg = sam3.SAM3Config() if defaults else steady()
    with caplog.at_level("INFO"):
        out = repair_rig(cfg, **(at_defaults() if defaults else {}))
    _, track = passes[-1]
    assert out.masks is track and torch.equal(out.masks, out.prompt)     # the track's tensor itself
    assert out.after == [] and out.refines == []
    assert not [line for line in caplog.text.splitlines() if "prompt: frame" in line]
    assert "tracked again" not in out.result and "refined frames" not in out.result


def test_the_logits_record_reproduces_prompt_mode_s_masks_after_both_repairs(repair_rig):
    cfg = steady()
    dump = {}
    out = repair_rig(cfg, make=DropsAndLoses, size=BIG, logits=dump)
    assert dump["cut"][2] == "birth" and not dump["raw"][2]
    assert all(dump["cut"][f] == "prompt" and dump["raw"][f] for f in range(30, RIG_N) if f != 33)
    assert [f for f in (LOSES, 33) if dump["cut"][f] == "prompt" and not dump["raw"][f]] == [LOSES, 33]
    for f in range(RIG_N):
        again = reproduce(dump["logits"][f], "prompt", *BIG, 0.0, None, 0, cfg, dump["raw"][f])
        assert torch.equal(again, out.masks[f]), f
    # the entry: the prompt mode track() runs is this one, and the sink gets the record
    got = {}
    masks = sam3.track((FakeModel(), object()), torch.zeros(RIG_N, *BIG, 3), config=cfg,
                       logits_sink=lambda logits_, info: got.update(info=info))
    assert got["info"]["mode"] == "prompt" and got["info"]["cut"][33] == "prompt" and not got["info"]["raw"][33]
    assert torch.equal(masks, out.masks)


def test_prompt_mode_keeps_the_conditioning_frames_masks_alone(rig, monkeypatch):
    """What prompt mode's capture keeps of a conditioning frame: its mask logits, what a refine of it or
    a re-track from it reads; prompt_pose's keeps the outputs whole (test_sam3_1_multiplex_prompt_pose)."""
    cfg = config()
    base, _, _ = rig(cfg)
    capture = {"raw": {}}
    masks, _, _ = prompt_run(rig, monkeypatch, cfg, capture=capture)
    assert torch.equal(masks, base) and sorted(capture["cond"]) == [2, 16, 32]
    assert all(list(out) == ["pred_masks"] for out in capture["cond"].values())


def test_prompt_pose_runs_the_track_alone(repair_rig, pp_rig, monkeypatch):
    """On a clip prompt mode repairs, prompt_pose with no pose to add points from returns its pass 1, the
    track's tensor itself: the part lost for good stays lost and nothing is tracked again."""
    assert repair_rig(steady(), make=DropsAndLoses, size=BIG).result["tracked again"] == 10
    passes = track_spy(monkeypatch)
    pose = pp_rig(steady(), make=DropsAndLoses, size=BIG)
    _, track = passes[-1]
    assert pose.refines == [] and pose.masks is track
    assert not pose.masks[RIG_N - 1][part(RIG_N - 1, BIG)].any()
