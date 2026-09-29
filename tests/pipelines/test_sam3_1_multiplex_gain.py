"""SAM 3.1 Multiplex prompt mode's re-track from a gain: the rule that finds the frame where the
track gained a piece it keeps, shortly after its birth (gain_frame, on hand-written masks), and the
one backward pass from that frame on the scripted tracker of test_sam3_1_multiplex_ab: the frames
before it tracked again from its mask, the frames from it on untouched, the logits record and
prompt_pose's capture taking the frames tracked again as frames before a birth, and a clip it does
not fire on left as it was. No model is loaded.

Needs ComfyUI importable (the ComfyUI root on PYTHONPATH), like the other SAM 3.1 Multiplex tests:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/pipelines/test_sam3_1_multiplex_gain.py
"""
import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import sam3  # noqa: E402
from test_sam3_1_multiplex_ab import (H, N, W, FakeTracker, PointerTracker, box, config, reproduce,  # noqa: E402,F401
                                      rig)
from test_sam3_1_multiplex_prompt_pose import pp_rig, prompt_run, right_hand, tracked  # noqa: E402,F401


# --- the rule on hand-written masks -------------------------------------------------------------
# A 20 x 20 frame: the body, 80 px, on every frame from the birth; from frame g on the arm beside it,
# 20 px: 0.2 of the mask. Nothing before the birth, as before the backward pass fills it.

BODY = (slice(0, 10), slice(0, 8))
ARM = (slice(2, 6), slice(8, 13))


def gaining(frames, g, arm=ARM, birth=0):
    masks = torch.zeros(frames, 20, 20)
    for f in range(birth, frames):
        masks[f][BODY] = 1.0
        if f >= g:
            masks[f][arm] = 1.0
    return masks


def test_a_piece_that_lasts_right_after_the_birth_is_a_gain():
    assert (sam3.GAIN_SHARE, sam3.KEPT_TENTHS, sam3.KEPT_FRAMES, sam3.GAIN_WINDOW) == (0.15, 8, 5, 16)
    assert sam3.gain_frame(gaining(12, 3), 0) == (3, 0.2)


def test_the_largest_new_piece_must_be_15_percent_of_the_mask():
    assert sam3.gain_frame(gaining(12, 3, arm=(slice(2, 4), slice(8, 15))), 0) is None        # 14 of 94 px
    assert sam3.gain_frame(gaining(12, 3, arm=(slice(2, 5), slice(8, 13))), 0) == (3, 15 / 95)
    # 20 px gained in two pieces of 10 (0.1 of the mask each) is no gain
    masks = gaining(12, 3, arm=(slice(0, 2), slice(8, 13)))
    masks[3:, 4:6, 8:13] = 1.0
    assert sam3.gain_frame(masks, 0) is None
    masks[3:, 2:4, 8:13] = 1.0                                                              # joined: 30 px
    assert sam3.gain_frame(masks, 0) == (3, 30 / 110)


@pytest.mark.parametrize("after", [1, 2, 3, 4, 5])
def test_the_piece_must_stay_80_percent_inside_on_each_of_the_5_frames_after(after):
    masks = gaining(14, 3)
    masks[3 + after, 2:6, 12] = 0.0                  # 16 of the 20 px stay: 0.8
    assert sam3.gain_frame(masks, 0) == (3, 0.2)
    masks[3 + after, 2, 11] = 0.0                    # 15 stay: 0.75
    assert sam3.gain_frame(masks, 0) is None


def test_what_follows_the_5_frames_does_not_count():
    masks = gaining(14, 3)
    masks[9:][:, ARM[0], ARM[1]] = 0.0               # the arm gone from the sixth frame after on
    assert sam3.gain_frame(masks, 0) == (3, 0.2)


def test_a_gain_needs_5_frames_after_it():
    assert sam3.gain_frame(gaining(8, 3), 0) is None
    assert sam3.gain_frame(gaining(9, 3), 0) == (3, 0.2)


def test_only_a_gain_up_to_16_frames_after_the_birth_counts():
    assert sam3.gain_frame(gaining(30, 18, birth=2), 2) == (18, 0.2)
    assert sam3.gain_frame(gaining(30, 19, birth=2), 2) is None
    # the birth frame's whole mask is new against the empty frame before it: no gain
    assert sam3.gain_frame(gaining(30, 2, birth=2), 2) is None
    assert sam3.gain_frame(torch.zeros(30, 20, 20), -1) is None


def test_of_two_gains_the_later_one_counts():
    masks = gaining(20, 3)
    masks[6:, 10:14, 0:5] = 1.0                      # a second piece, 20 px under the body, from frame 6
    assert sam3.gain_frame(masks, 0) == (6, 20 / 120)


# --- the backward pass from a gain, on the scripted tracker ---------------------------------------
# A person standing still: the box (64 low-res px) on every frame; forwards the tracker finds the
# arm beside it (32 px) on its own from frame `gain` on, while the detection never has it.

STILL = (4, 2)
ARM_LOW = (slice(4, 12), slice(10, 14))


def person(arm):
    logits = box(*STILL)
    if arm:
        logits[ARM_LOW] = 10.0
    return logits[None, None]


def still_person(birth):
    """The person's detection, without the arm, from frame `birth` on."""
    return lambda real: [(box(*STILL), 0.9)] if real >= birth else []


class RegainsTheArm(FakeTracker):
    """The scripted tracker on the still person: forwards the box, with the arm from frame `gain`
    on; backwards (a mirrored index) with the arm where the conditioning frame it tracks from has
    it."""
    def __init__(self, gain, **kwargs):
        super().__init__(**kwargs)
        self.gain = gain

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes,
                   mask_inputs, output_dict, num_frames, **kwargs):
        out = super().track_step(frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
                                 feat_sizes, mask_inputs, output_dict, num_frames, **kwargs)
        real = current_vision_feats[0]
        if frame_idx == real:
            arm = real >= self.gain
        else:
            arm = any(bool((cond["pred_masks"][0, 0][ARM_LOW] > 0).all())
                      for cond in output_dict["cond_frame_outputs"].values())
        out["pred_masks"] = out["pred_masks_high_res"] = person(arm)
        return out


class RegainsWithPointers(RegainsTheArm, PointerTracker):
    """RegainsTheArm with test_sam3_1_multiplex_ab's propagation decoder, for the best_iou pointer."""


def shows(arm):
    return sam3.to_frame_size(person(arm), H, W)


def backwards(log):
    """The backward pass's calls in the tracker's log: its conditioning frames as (index, frame, mask
    px), and the frames it tracked, in order."""
    return ([e[1:4] for e in log if e[0] == "cond" and e[1] >= N],
            [e[2] for e in log if e[0] == "track" and e[1] >= N])


def no_gain(monkeypatch):
    """The run as it was before the gain rule: the rule never fires."""
    monkeypatch.setattr(sam3, "gain_frame", lambda masks, birth: None)


@pytest.mark.parametrize("birth, gain", [(0, 11), (2, 18)])
def test_the_frames_before_a_gain_are_tracked_again_backwards_from_it(rig, monkeypatch, caplog, birth, gain):
    """The frames before the gain show the backward pass from the gain frame's mask, arm and all:
    one pass, conditioned on that mask (the box and the arm, 96 px) under the mirrored index, over
    every frame before it once - with a birth on frame 2 the frames before the birth, the birth and
    the anchor on 16 (born on 2, the gain on 18) among them. The frames from the gain on are the
    forward pass's, as they are without the rule."""
    with caplog.at_level("INFO"):
        masks, log, result = rig(config(), detections=still_person(birth), tracker=RegainsTheArm(gain))
    assert backwards(log) == ([(2 * N - gain, gain, 96)], list(range(gain - 1, -1, -1)))
    for f in range(N):
        assert torch.equal(masks[f], shows(arm=True)), f
    assert (result["gained on frame"], result["tracked backwards"], result.get("tracked from frame", 0)) == \
        (gain, gain, birth)
    assert (f"prompt: frames 0-{gain - 1} tracked backwards from frame {gain} (the track was born on frame {birth}): "
            f"it gained a piece frame {gain - 1} lacked (0.33 of its mask)") in caplog.text

    no_gain(monkeypatch)
    before, before_log, _ = rig(config(), detections=still_person(birth), tracker=RegainsTheArm(gain))
    assert torch.equal(masks[gain:], before[gain:])
    assert all(torch.equal(before[f], shows(arm=False)) for f in range(gain))
    assert backwards(before_log)[0] == ([(2 * N - birth, birth, 64)] if birth else [])


def test_the_record_and_the_capture_take_the_frames_tracked_again_as_frames_before_a_birth(rig, monkeypatch):
    """Born on 2, re-anchored on 16, the gain on 18. In the logits record every frame before 18 is a
    "prompt" frame whose logits are the backward pass's before the cleaning, as the frames before a
    birth are; the re-anchor slots stay the forward pass's. prompt_pose's capture hands 18 over as
    the birth, with no conditioning frame and no raw logits before it."""
    dump = {}
    masks, _, _ = rig(config(), detections=still_person(2), tracker=RegainsTheArm(18), logits=dump)
    assert dump["cut"] == ["prompt"] * N and dump["raw"] == [True] * N
    assert [(a["frame"], a["fired"]) for a in dump["anchors"]] == [(16, True), (32, False)]
    for f in range(N):
        again = reproduce(dump["logits"][f], "prompt", H, W, 0.0, None, 0, config(), dump["raw"][f])
        assert torch.equal(again, masks[f]), f
    capture = {"raw": {}}
    prompt_run(rig, monkeypatch, config(), lambda: RegainsTheArm(18), capture)
    assert capture["birth"] == 18 and capture["cond"] == {} and sorted(capture["raw"]) == list(range(18, N))
    no_gain(monkeypatch)
    capture = {"raw": {}}
    prompt_run(rig, monkeypatch, config(), lambda: RegainsTheArm(18), capture)
    assert capture["birth"] == 2 and sorted(capture["cond"]) == [2, 16] and sorted(capture["raw"]) == list(range(3, N))


def test_best_iou_counts_every_frame_once(rig, caplog):
    """Born on 0, the gain on 11: frames 0-10 are the backward pass's, 11-39 the forward pass's, 40
    propagated frames with the mask each selected, the birth frame's among them."""
    dump = {}
    with caplog.at_level("INFO"):
        rig(config(obj_ptr_token=sam3.BEST_IOU), detections=still_person(0), tracker=RegainsWithPointers(11, best={}),
            core_mux=True, logits=dump)
    assert dump["mask_index"] == [0] * N
    assert "the decoder selected a mask other than token 0 on 0 of 40 propagated frame(s)" in caplog.text


def test_where_the_rule_does_not_fire_the_run_is_as_it_was(rig, monkeypatch):
    """The arm found on frame 19, 17 frames after the birth: no gain. The frames before the birth
    are tracked backwards from the birth detection (the box, 64 px), and the masks, the tracker's
    calls, the counts and the logits record are those of the run without the rule."""
    runs = []
    for rule in (True, False):
        if not rule:
            no_gain(monkeypatch)
        dump = {}
        runs.append(rig(config(), detections=still_person(2), tracker=RegainsTheArm(19), logits=dump) + (dump,))
    (masks, log, result, dump), (before, before_log, before_result, before_dump) = runs
    assert torch.equal(masks, before) and log == before_log and result == before_result
    assert backwards(log) == ([(2 * N - 2, 2, 64)], [1, 0]) and "gained on frame" not in result
    assert all(torch.equal(masks[f], shows(arm=f >= 19)) for f in range(N))
    for key in ("cut", "raw", "anchors"):
        assert dump[key] == before_dump[key], key
    assert all(torch.equal(a, b) for a, b in zip(dump["logits"], before_dump["logits"]))


def test_prompt_pose_starts_its_second_pass_at_the_gain(pp_rig):
    """Born on 2, the gain on 18, frame 30 refined: the capture's birth is 18, so the second pass
    tracks from 18 around the refined frame (pass 1's conditioning frames, 2 and 16, lie before
    it), and frames 0-17 keep pass 1's masks, the backward pass's from the gain."""
    out = pp_rig(config(), chosen={30: [right_hand(0)]}, make=lambda: RegainsTheArm(18), detections=still_person(2))
    assert list(tracked(out.second)) == [f for f in range(18, N) if f != 30]
    assert torch.equal(out.masks[:18], out.prompt[:18])
    assert torch.equal(out.prompt[:18], shows(arm=True).expand(18, H, W))
    assert out.result["kept from the first pass"] == 18 and "demoted" not in out.result
