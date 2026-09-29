"""prompt_pose's refine, `refine_with_points`, on a scripted stand-in for core's tracker: the decode
reads no memory (the interactive neck on the cached trunk plus interactivity_no_mem_embed), the
points are Meta's (capped at 16: the first 8 and the last 8), the dense prompt is the previous mask
clamped to +/-32 or nothing, one point decodes three masks, two or more take token 0 unless Meta's
stability fallback picks the best-IoU mask (the pointer staying token 0's), and the frame's memory
is a conditioning memory from the 1008 mask. No model is loaded.

The adapter imports ComfyUI's tracker helpers when called, so this runs where ComfyUI is
importable (the ComfyUI root on PYTHONPATH):

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/models/test_sam3_1_multiplex_refine.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from sam3_1_multiplex_fakes import sam3  # noqa: E402

C, SIDE, LOW = 8, 4, 16          # channels and side of the low-res features, side of the decoder's masks
CPU = torch.device("cpu")


class Trunk(torch.nn.Module):
    """The backbone's trunk: records a call, which the refine must not make."""
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, frame):
        self.calls += 1
        return frame


class Backbone:
    """The vision backbone: its interactive neck returns two high-res maps and the low-res one."""
    def __init__(self):
        torch.manual_seed(0)
        self.trunk = Trunk()
        self.calls = []
        self.features = [torch.randn(1, C, 4 * SIDE, 4 * SIDE), torch.randn(1, C, 2 * SIDE, 2 * SIDE),
                         torch.randn(1, C, SIDE, SIDE)]

    def __call__(self, frame, tracker_mode=None, cached_trunk=None, tracker_only=False):
        self.calls.append((tracker_mode, cached_trunk, tracker_only))
        return None, None, list(self.features), None


def logits(above, between, value=5.0):
    """[1, 1, LOW, LOW] mask logits: `above` pixels at `value` (over +0.05), `between` at 0 (within
    +/-0.05), the rest at -5: token 0's stability is above / (above + between)."""
    out = torch.full((LOW * LOW,), -5.0)
    out[:above] = value
    out[above:above + between] = 0.0
    return out.view(1, 1, LOW, LOW)


class Heads:
    """Core's tracker as the refine drives it: `_forward_sam_heads` answers token 0's mask
    (`single`) or the best-IoU one of the three (`multi`), with token 0's pointer (ones) or the best
    mask's (twos), and records what it was given; `_encode_new_memory` records its arguments."""
    def __init__(self, single, multi, score=3.0):
        self.single, self.multi, self.score = single, multi, score
        self.interactivity_no_mem_embed = torch.arange(C, dtype=torch.float32).view(1, 1, C) / 10
        self.heads, self.encoded = [], []

    def _forward_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, box_inputs=None,
                           high_res_features=None, multimask_output=False):
        self.heads.append({"features": backbone_features, "points": point_inputs, "mask": mask_inputs,
                           "box": box_inputs, "high_res": high_res_features, "multimask": multimask_output})
        low = self.multi if multimask_output else self.single
        high = torch.nn.functional.interpolate(low, size=(4 * LOW, 4 * LOW), mode="bilinear", align_corners=False)
        pointer = torch.full((1, C), 2.0 if multimask_output else 1.0)
        return low, high, pointer, torch.tensor([[self.score]])

    def _encode_new_memory(self, pix_feat, pred_masks_high_res, object_score_logits, is_mask_from_pts=False,
                           multiplex_state=None, is_conditioning=False, cond_obj_mask=None):
        self.encoded.append({"pix_feat": pix_feat, "mask": pred_masks_high_res, "score": object_score_logits,
                             "from_points": is_mask_from_pts, "conditioning": is_conditioning,
                             "mux": multiplex_state})
        return "memory", ["position"]


def refine(tracker, points, previous=None, backbone=None):
    backbone = backbone or Backbone()
    vision_feats = [torch.randn(1, SIDE * SIDE, C)]                    # [1, HW, C], the propagation features
    vision_pos = [torch.randn(1, SIDE * SIDE, C)]
    mux = sam3.MultiplexState(1, 16, CPU, torch.float32)
    output, info = sam3.refine_with_points(tracker, backbone, "frame", "trunk", vision_feats, vision_pos,
                                           [(SIDE, SIDE)], points, mux, previous)
    return output, info, backbone, vision_feats, vision_pos, mux


STABLE = logits(98, 2)       # stability 98 / 100 = 0.98: token 0 is kept
UNSTABLE = logits(97, 3)     # 97 / 100 = 0.97: the fallback takes the best-IoU mask
MULTI = logits(120, 0, value=7.0)
POINTS = [(10.0 * k, 5.0 * k) for k in range(1, 6)]


def test_the_refine_decodes_on_the_interactive_neck_of_the_cached_trunk_with_no_memory():
    tracker = Heads(STABLE, MULTI)
    output, info, backbone, vision_feats, vision_pos, _ = refine(tracker, POINTS)
    assert backbone.calls == [("interactive", "trunk", True)] and backbone.trunk.calls == 0
    (call,) = tracker.heads
    low = backbone.features[-1]
    # interactivity_no_mem_embed added to every position of the low-res map, as core's conditioning path adds it
    assert torch.equal(call["features"], low + tracker.interactivity_no_mem_embed.view(1, C, 1, 1))
    assert call["high_res"] == backbone.features[:2] and call["box"] is None
    assert output["image_features"] is vision_feats[-1] and output["image_pos_enc"] is vision_pos[-1]


def test_the_points_are_positives_as_given_and_capped_at_the_first_8_and_the_last_8():
    tracker = Heads(STABLE, MULTI)
    points = [(float(k), float(100 + k)) for k in range(22)]
    _, info, *_ = refine(tracker, points)
    sent = points[:8] + points[-8:]
    assert sam3.MAX_REFINE_POINTS == 16 and info["points"] == sent
    coords, labels = tracker.heads[0]["points"]["point_coords"], tracker.heads[0]["points"]["point_labels"]
    assert coords.dtype == torch.float32 and coords.tolist() == [[list(p) for p in sent]]
    assert labels.dtype == torch.int32 and labels.tolist() == [[1] * 16]
    tracker = Heads(STABLE, MULTI)
    _, info, *_ = refine(tracker, POINTS)
    assert info["points"] == POINTS and tracker.heads[0]["points"]["point_coords"].tolist() == [[list(p) for p in POINTS]]


def test_the_dense_prompt_is_nothing_or_the_previous_mask_clamped_to_32():
    tracker = Heads(STABLE, MULTI)
    refine(tracker, POINTS)
    assert tracker.heads[0]["mask"] is None                  # pose_refine_with_mask off: the points alone
    previous = torch.linspace(-100.0, 100.0, LOW * LOW).view(1, 1, LOW, LOW)
    tracker = Heads(UNSTABLE, MULTI)
    refine(tracker, POINTS, previous=previous)
    assert len(tracker.heads) == 2                           # the fallback decodes with the same prompt
    for call in tracker.heads:
        assert torch.equal(call["mask"], previous.clamp(-32.0, 32.0))
        assert call["mask"].min() == -32.0 and call["mask"].max() == 32.0


def test_one_point_decodes_three_masks_and_keeps_the_best():
    tracker = Heads(UNSTABLE, MULTI)
    output, info, _, _, _, mux = refine(tracker, POINTS[:1])
    assert [call["multimask"] for call in tracker.heads] == [True]
    assert torch.equal(output["pred_masks"], MULTI) and torch.equal(output["obj_ptr"], mux.mux(torch.full((1, C), 2.0)))
    assert info["stability"] is None and info["fallback"] is False


@pytest.mark.parametrize("token_0, fallback, stability", [(STABLE, False, 0.98), (UNSTABLE, True, 0.97),
                                                         (logits(0, 0), False, 1.0)])
def test_two_or_more_points_keep_token_0_unless_its_stability_is_under_0_98(token_0, fallback, stability):
    """Meta's rule (mask_decoder.py _dynamic_multimask_via_stability, delta 0.05, threshold 0.98):
    token 0's pixels over +0.05 against those over -0.05, 1.0 where there are none; under 0.98 the
    best-IoU mask of the three is taken, and the object pointer stays token 0's."""
    tracker = Heads(token_0, MULTI)
    output, info, _, _, _, mux = refine(tracker, POINTS)
    assert [call["multimask"] for call in tracker.heads] == ([False, True] if fallback else [False])
    assert torch.equal(output["pred_masks"], MULTI if fallback else token_0)
    assert info["fallback"] is fallback
    assert info["stability"] == stability
    assert torch.equal(output["obj_ptr"], mux.mux(torch.ones(1, C)))                # token 0's pointer
    assert torch.equal(output["object_score_logits"], torch.tensor([[3.0]]))


@pytest.mark.parametrize("token_0", [STABLE, UNSTABLE])
def test_the_memory_is_a_conditioning_memory_from_the_1008_mask(token_0):
    tracker = Heads(token_0, MULTI)
    output, _, _, vision_feats, _, mux = refine(tracker, POINTS)
    (encoded,) = tracker.encoded
    assert encoded["conditioning"] is True and encoded["from_points"] is False and encoded["mux"] is mux
    assert encoded["mask"] is output["pred_masks_high_res"] and encoded["mask"].shape[-1] == 4 * LOW
    assert torch.equal(encoded["score"], output["object_score_logits"])
    # the propagation features, as core's track_step encodes a conditioning frame's memory
    assert torch.equal(encoded["pix_feat"], vision_feats[-1].permute(0, 2, 1).reshape(1, C, SIDE, SIDE))
    assert (output["maskmem_features"], output["maskmem_pos_enc"]) == ("memory", ["position"])
