"""The SAM3 A/B switches (specs/mask-process.md A1-A10, M4) on a scripted stand-in for core's
tracker and SAM 3.1's detector. No model is loaded: the stand-ins answer the primitives
segment_by_prompt / segment_by_prompt_multi / segment_by_pose drive (`_condition_with_masks`,
`track_step`, `_deferred_memory_encode`, the detections, the box_keypoint decoder) with small
masks that depend on what the policy fed them - which frames are conditioning, which spatial
memories the lookup would read - and log every call.

Two kinds of test per switch: with the defaults the run is exactly the one recorded from the
code before the switches existed (the GOLDEN digests below: mask per frame and the call log),
and with the switch on, the path it names changes and nothing else is asked of it.

Needs ComfyUI importable (the pod, or ComfyUI's root on PYTHONPATH), like test_sam3.py."""
import hashlib

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

sam3 = pytest.importorskip("preprocess.sam3")

N, H, W, LOW = 40, 32, 32, 16
SPECK = (slice(14, 16), slice(0, 2))             # low-res pixels of FakeTracker's speck, clear of the person's ring
SPECK_FRAME = (slice(28, 32), slice(0, 4))       # and the frame pixels it covers


def box(y, x, h=8, w=8, hole=None, value=10.0, ring=0):
    """[LOW, LOW] logits: +value in the box, -value elsewhere, an optional 1-pixel hole, and a
    border `ring` pixels wide of -0.5 around the box: outside the mask at a cut of 0, inside it
    at -1 (where bilinear upsampling keeps it between -1 and 0, which takes two pixels)."""
    out = torch.full((LOW, LOW), -value)
    if ring:
        out[max(y - ring, 0):y + h + ring, max(x - ring, 0):x + w + ring] = -0.5
    out[y:y + h, x:x + w] = value
    if hole is not None:
        out[hole] = -value
    return out


def position(real):
    """Where the person is on frame `real`, in low-res pixels."""
    return 4, 2 + (real % 6)


class FakeTracker:
    """Core's tracker primitives, scripted. The propagated mask is the person's box, one row
    shorter when fewer than two spatial memories are readable, so memory policy shows in the
    output; the object score comes from `scores` (default 5.0)."""
    image_size = 16
    num_maskmem = 7
    max_obj_ptrs_in_encoder = 16
    num_multiplex = 16

    def __init__(self, scores=None, empty=(), ring=1, speck=False):
        self.ring = ring
        self.speck = speck
        self.log = []
        self.reads = {}   # forward frame -> the real frames its readable spatial memories came from
        self.scores = scores or {}
        self.empty = set(empty)

    def _compute_backbone_frame(self, backbone_fn, frame, frame_idx=None):
        return [frame], None, [(4, 4)], None, frame

    def _condition_with_masks(self, masks, frame_idx, vision_feats, vision_pos, feat_sizes, high_res, output_dict,
                              N, mux, backbone, frame, trunk_out, threshold=0.5):
        m = masks if masks.dim() == 4 else masks.unsqueeze(1)
        # as core: resized to the tracker's input size, then thresholded
        binary = torch.nn.functional.interpolate(m.float(), size=(self.image_size, self.image_size),
                                                 mode="bilinear", align_corners=False)[0, 0] > threshold
        current = {"pred_masks": (binary.float() * 20 - 10)[None, None], "object_score_logits": torch.tensor([[10.0]]),
                   "obj_ptr": 0, "maskmem_features": ("cond", int(binary.sum())), "maskmem_pos_enc": [0],
                   "pred_masks_high_res": (binary.float() * 20 - 10)[None, None]}
        output_dict["cond_frame_outputs"][frame_idx] = current
        self.log.append(("cond", frame_idx, vision_feats[0], int(binary.sum()),
                         hashlib.md5(binary.numpy().tobytes()).hexdigest()[:8]))
        return current

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds, feat_sizes,
                   mask_inputs, output_dict, num_frames, propagation_high_res=None, multiplex_state=None,
                   run_mem_encoder=True, **kwargs):
        real = current_vision_feats[0]
        non_cond = output_dict["non_cond_frame_outputs"]
        cond = tuple(sorted(output_dict["cond_frame_outputs"]))
        mem = tuple(frame_idx - k for k in range(1, self.num_maskmem)
                    if non_cond.get(frame_idx - k) is not None and non_cond[frame_idx - k].get("maskmem_features") is not None)
        ptrs = tuple(frame_idx - k for k in range(1, self.max_obj_ptrs_in_encoder) if non_cond.get(frame_idx - k) is not None)
        y, x = position(real)
        logits = box(y, x, h=8 if len(mem) >= 2 else 7, ring=self.ring)
        if real in self.empty:
            logits = torch.full((LOW, LOW), -10.0)
        elif self.speck:
            logits[SPECK] = 5.0   # a 2x2 speck the decoder leaves in the background
        self.log.append(("track", frame_idx, real, cond, mem, ptrs))
        if frame_idx == real:
            self.reads[real] = tuple(non_cond[t]["maskmem_features"][2] for t in mem
                                     if non_cond[t]["maskmem_features"][0] == "mem")
        return {"pred_masks": logits[None, None], "pred_masks_high_res": logits[None, None],
                "object_score_logits": torch.tensor([[self.scores.get(real, 5.0)]]), "obj_ptr": 0,
                "maskmem_features": None, "maskmem_pos_enc": None}

    def _deferred_memory_encode(self, current, N_obj, vision_feats, feat_sizes, mux, device, cond_obj_mask=None):
        current["maskmem_features"] = ("mem", int((current["pred_masks"] > 0).sum()), vision_feats[0])
        current["maskmem_pos_enc"] = [0]
        self.log.append(("encode", vision_feats[0], int((current["pred_masks"] > 0).sum())))


class FakeModel:
    class model:
        @staticmethod
        def get_dtype():
            return torch.float32


def person_detection(real, score=0.9, hole=True):
    y, x = position(real)
    return box(y, x, hole=(y + 3, x + 3) if hole else None), score


def default_detections(real):
    """The person on every frame from frame 2, with a pinhole; on the anchor frames 16 and 32 a
    second, weaker detection of the same person with a different pinhole."""
    if real < 2:
        return []
    dets = [person_detection(real)]
    if real in (16, 32):
        y, x = position(real)
        dets.append((box(y, x, hole=(y + 5, x + 1)), 0.85))
    return dets


@pytest.fixture
def rig(monkeypatch):
    """Patches the model-facing calls; returns run(config, detections=..., **tracker_kwargs)."""
    def run(config=None, detections=default_detections, multi=0, **tracker_kwargs):
        tracker = FakeTracker(**tracker_kwargs)

        def detect(detector, backbone, trunk_out, embedding, text_mask, config):
            found = detections(trunk_out)
            if not found:
                return torch.zeros(0, LOW, LOW), torch.zeros(0)
            return torch.stack([m for m, _ in found]), torch.tensor([s for _, s in found])

        monkeypatch.setattr(sam3.mm, "load_model_gpu", lambda model: None)
        monkeypatch.setattr(sam3.mm, "get_torch_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (None, None, tracker, None))
        monkeypatch.setattr(sam3, "encode_prompt", lambda *a: (None, None))
        monkeypatch.setattr(sam3, "detect_person", detect)
        monkeypatch.setattr(sam3, "_prep_frame", lambda frames, idx, device, dtype, size: idx.start)
        monkeypatch.setattr(sam3, "MultiplexState", lambda *a: object())
        images = torch.zeros(N, H, W, 3)
        config = config or sam3.SAM3Config()
        result = {}
        if multi:
            masks = sam3.segment_by_prompt_multi(FakeModel(), object(), images, "p", config, multi, -1, result=result)
        else:
            masks = sam3.segment_by_prompt(FakeModel(), object(), images, "p", config, result=result)
        run.tracker = tracker
        return masks, tracker.log, result
    return run


def digest(masks, log):
    return (hashlib.md5(masks.numpy().tobytes()).hexdigest(),
            hashlib.md5(repr(log).encode()).hexdigest())


def probation_detections(real):
    """A detection on frame 0 only, then nothing until frame 12: the track born on frame 0
    goes unmatched through its probation window."""
    return [person_detection(real)] if real == 0 or real >= 12 else []


PROBATION = dict(detections=probation_detections, empty=range(3, 10))


def two_people(real):
    """The person, and from frame 5 a second one in the bottom-left corner."""
    dets = default_detections(real)
    if real >= 5:
        dets.append((box(10, 0, h=5, w=4), 0.7))
    return dets


class FakeSam3:
    """What segment_by_pose reads off the model: the tracker, and a backbone dict."""
    def __init__(self, tracker):
        self.tracker = tracker
        self.detector = type("D", (), {"backbone": {"vision_backbone": None}})()


def pose_metas_and_boxes():
    """Every frame: the box around the person and five confident keypoints inside it."""
    metas, boxes = [], []
    for real in range(N):
        y, x = position(real)
        kps = np.zeros((20, 3))
        for k in (0, 1, 2, 5, 8):
            kps[k] = [(2 * x + 8) / W, (2 * y + 4 + k) / H, 0.9]
        metas.append({"keypoints_body": kps})
        boxes.append(np.array([2 * x, 2 * y, 2 * x + 16, 2 * y + 16, 0.9]))
    return metas, boxes


@pytest.fixture
def pose_rig(monkeypatch):
    """segment_by_pose on the scripted tracker, the decoder answering every prompt with the
    person's box and a ring of -0.5 around it; returns run(config, **kwargs)."""
    def run(config=None, ring=1, **kwargs):
        tracker = FakeTracker(ring=ring)
        decoded = []

        def decode(model, frame, point_inputs, box_inputs, refine):
            decoded.append(len(decoded))
            y, x = int(box_inputs[0, 0, 1] * H / sam3.SAM3_SIZE) // 2, int(box_inputs[0, 0, 0] * W / sam3.SAM3_SIZE) // 2
            return box(y, x, ring=ring)[None, None]

        monkeypatch.setattr(sam3.mm, "load_model_gpu", lambda model: None)
        monkeypatch.setattr(sam3.mm, "get_torch_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(sam3.mm, "intermediate_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (FakeSam3(tracker), None, tracker, None))
        monkeypatch.setattr(sam3, "decode", decode)
        monkeypatch.setattr(sam3, "_prep_frame", lambda frames, idx, device, dtype, size: idx.start)
        monkeypatch.setattr(sam3, "MultiplexState", lambda *a: object())
        metas, boxes = pose_metas_and_boxes()
        result = {}
        masks = sam3.segment_by_pose(FakeModel(), torch.zeros(N, H, W, 3), boxes, metas, config or sam3.SAM3Config(),
                                     0.3, result=result, **kwargs)
        return masks, tracker.log, result
    return run


# Recorded from preprocess/sam3.py at 03c0a10, before any switch existed: (masks, call log).
GOLDEN = {
    "single": ("a858ad7c2face423c09e9158f957570c", "4a0fd6c7b53c73f5170becab17596c90"),
    "multi": ("a858ad7c2face423c09e9158f957570c", "4a0fd6c7b53c73f5170becab17596c90"),
    "probation": ("2186e143fc68a74c651bde14c3e3fc0c", "83c4bfb98b200ddfea08edae2961e26d"),
    "two": ("5070fe57e5756268e1a0d421e2aae4a9", "ef69dcd98ca5b3b8aaa3aa4f1ca9a814"),
    "pose": ("b63a1e0bf4572057a6ae39a909a0ff19", "fc19fae5abe7b537b36f6ba76dd6ca7b"),
}
SCENARIOS = {"single": {}, "multi": {"multi": 2}, "probation": PROBATION,
             "two": {"multi": 2, "detections": two_people}}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_the_defaults_are_the_code_before_the_switches(rig, name):
    masks, log, _ = rig(**SCENARIOS[name])
    assert digest(masks, log) == GOLDEN[name]


def test_box_keypoint_defaults_are_the_code_before_the_switches(pose_rig):
    masks, log, _ = pose_rig()
    assert digest(masks, log) == GOLDEN["pose"]


# --- the switches --------------------------------------------------------------------------

def forward(log, real):
    """The forward pass's track_step entry for frame `real`: (cond frames, readable memories)."""
    (entry,) = [e for e in log if e[0] == "track" and e[1] == real and e[2] == real]
    return entry[3], entry[4]


def conditioned(log, real):
    """The mask sum and hash the tracker was conditioned with on frame `real` (forward pass)."""
    (entry,) = [e for e in log if e[0] == "cond" and e[1] == real]
    return entry[3], entry[4]


def config(**kwargs):
    return sam3.SAM3Config(**kwargs)


def test_a1_memory_around_an_anchor(rig):
    _, log, _ = rig(config(anchor_memory="ours"))
    assert forward(log, 17)[1] == () and forward(log, 22)[1] == ()        # cleared, then held off
    assert forward(log, 23)[1] == () and forward(log, 25)[1] == (24,)   # 17-23 are held off
    _, log, _ = rig(config(anchor_memory="clear_past"))
    assert forward(log, 17)[1] == () and forward(log, 19)[1] == (18, 17)  # cleared, encoded again at once
    _, log, _ = rig(config(anchor_memory="meta"))
    assert forward(log, 17)[1] == (15, 14, 13, 12, 11)                    # nothing cleared (16 is conditioning)


def anchor_taller(real):
    """As default_detections, but on the anchor frames the detection is one row taller than the
    propagated mask: still the same person (IoU 0.89), a different mask."""
    if real in (16, 32):
        y, x = position(real)
        return [(box(y, x, h=9), 0.9)]
    return default_detections(real)


def test_a2_an_anchor_frame_shows_the_detection_or_the_propagated_mask(rig):
    masks, log, _ = rig(detections=anchor_taller)
    assert masks[16].sum() == 2 * 9 * 2 * 8 and ("encode", 16, 64) not in log
    masks, log, _ = rig(config(anchor_output="meta"), detections=anchor_taller)
    assert masks[16].sum() == masks[15].sum()        # the propagated mask, the size of every tracked one
    assert ("encode", 16, 64) in log                 # whose memory the conditioning entry keeps
    assert conditioned(log, 16)[0] == 72             # the tracker is still conditioned on the detection


def test_a3_conditioning_frames(rig):
    _, log, _ = rig(config(recondition_every=4))
    assert forward(log, 30)[0] == (2, 28)
    _, log, _ = rig(config(recondition_every=4, conditioning_frames="meta"))
    assert forward(log, 30)[0] == (16, 20, 24, 28)


def test_a4_memory_selection_skips_frames_the_person_was_absent(rig):
    gone = {20: -5.0, 21: -5.0, 22: -5.0}
    rig(config(recondition_every=100), scores=gone)
    assert rig.tracker.reads[24] == (23, 22, 21, 20, 19, 18)
    rig(config(recondition_every=100, memory_selection="meta"), scores=gone)
    assert rig.tracker.reads[24] == (23, 19, 18, 17, 16, 15)
    # the frame before is read even when absent
    assert rig.tracker.reads[22][0] == 21


def test_a5_the_tracker_score_gate(rig):
    low = {16: 0.5}
    assert rig(scores=low)[2]["reconditioned"] == 2
    assert rig(config(anchor_score_gate="meta"), scores=low)[2]["reconditioned"] == 1


def two_agreeing(real):
    """On the anchor frames two detections agree with the track: the better one the track's
    size, the weaker one a row taller."""
    if real in (16, 32):
        y, x = position(real)
        return [(box(y, x), 0.95), (box(y, x, h=9), 0.85)]
    return default_detections(real)


def test_a6_which_detection_re_anchors_the_track(rig):
    _, log, _ = rig(detections=two_agreeing)
    assert conditioned(log, 16)[0] == 64      # the best-scoring
    _, log, _ = rig(config(anchor_matching="meta"), detections=two_agreeing)
    assert conditioned(log, 16)[0] == 72      # the last one Meta's loop sees


def test_a6_several_tracks_one_detection():
    scores = torch.tensor([0.9])
    overlap = torch.tensor([[0.85, 0.95]])    # one detection agreeing with two tracks
    assert sam3.anchor_detections(scores, overlap, [0, 1], config()) == {0: 0, 1: 0}
    assert sam3.anchor_detections(scores, overlap, [0, 1], config(anchor_matching="meta")) == {1: 0}
    assert sam3.anchor_detections(scores, overlap, [0], config(anchor_matching="meta")) == {0: 0}


def test_a7_an_empty_track_does_not_count_against_its_probation(rig):
    _, _, result = rig(**PROBATION)
    assert result["false starts"] == 1
    _, log, result = rig(config(unmatched_counting="meta"), **PROBATION)
    assert "false starts" not in result and "tracked backwards" not in result   # the frame-0 track lived


def test_a6_a7_are_read_with_several_objects(rig):
    _, _, ours = rig(multi=2, **PROBATION)
    _, _, meta = rig(config(unmatched_counting="meta"), multi=2, **PROBATION)
    assert ours["false starts"] == 1 and "false starts" not in meta


def test_a9_the_tracker_gets_the_raw_detection(rig):
    masks, log, _ = rig()
    assert conditioned(log, 2)[0] == 64 and masks[2].sum() == 256
    masks, log, _ = rig(config(seed_cleaning="meta"))
    assert conditioned(log, 2)[0] == 63       # the pinhole reaches the tracker
    assert masks[2].sum() == 256              # the birth frame's output is still cleaned
    assert conditioned(log, 16)[0] == 63


def test_m4_one_threshold_cuts_prompt_mode(rig):
    masks, _, _ = rig(ring=2)
    wide, _, _ = rig(config(uniform_mask_threshold=True), ring=2)   # mask_threshold -1.0 takes in the ring
    assert (wide[10] >= masks[10]).all() and wide[10].sum() > masks[10].sum()
    same, _, _ = rig(config(uniform_mask_threshold=True, mask_threshold=0.0), ring=2)
    assert torch.equal(same, masks)


def test_m4_one_threshold_cuts_box_keypoints_propagated_frames(pose_rig):
    masks, _, _ = pose_rig(ring=2)
    wide, _, _ = pose_rig(config(uniform_mask_threshold=True), ring=2)
    assert torch.equal(wide[0], masks[0])                        # the prompted frame was at -1.0 already
    assert wide[10].sum() > masks[10].sum()                      # the propagated ones were at 0
    tight, _, _ = pose_rig(config(uniform_mask_threshold=True, mask_threshold=0.0), ring=2)
    assert tight[0].sum() < masks[0].sum() and torch.equal(tight[10], masks[10])


def reproduce(low, how, H, W, threshold, mask_threshold, tracker_size, cfg, raw=False):
    """A frame's mask from its dumped logits, by the recipe `track` documents. `raw`: the logits
    are the ones before prompt mode's output cleaning (info["raw"]), cleaned here relative to
    the threshold."""
    low = low.float()[None, None]
    if how == "prompt":
        if raw:
            low = sam3.shown_logits(low, threshold, cfg.fill_hole_area)
        return sam3.to_frame_size(low, H, W, threshold)
    if how == "prompted":
        cut = torch.nn.functional.interpolate(low, size=(H, W), mode="bilinear", align_corners=False)[0, 0] > mask_threshold
    else:
        high = torch.nn.functional.interpolate(low, size=(tracker_size, tracker_size), mode="bilinear", align_corners=False)
        cut = torch.nn.functional.interpolate((high > threshold).float(), size=(H, W), mode="bilinear",
                                              align_corners=False)[0, 0] > 0.5
    return torch.from_numpy(sam3.clean_mask(cut.numpy(), cfg)).float()


@pytest.mark.parametrize("seed_cleaning", ["ours", "meta"])
@pytest.mark.parametrize("uniform", [False, True])
def test_the_logits_dump_reproduces_prompt_mode(rig, monkeypatch, uniform, seed_cleaning):
    got = {}
    cfg = config(uniform_mask_threshold=uniform, seed_cleaning=seed_cleaning)
    rig(cfg, ring=2, speck=True)   # installs the stand-ins
    masks = sam3.track((FakeModel(), object()), torch.zeros(N, H, W, 3), config=cfg,
                       logits_sink=lambda logits, info: got.update(logits=logits, info=info))
    info = got["info"]
    assert info["mode"] == "prompt" and info["size"] == (H, W)
    assert [info["cut"][f] for f in (2, 16, 32)] == ["birth", "anchor", "anchor"] and set(info["cut"]) == {"prompt", "birth", "anchor"}
    assert info["threshold"] == (-1.0 if uniform else 0.0) and info["fill_hole_area"] == cfg.fill_hole_area
    # the logits before the output's cleaning, except where the frame shows the conditioning
    # mask itself: the birth frame (seed_cleaning ours) and the anchors
    assert [f for f in range(N) if not info["raw"][f]] == ([2] if seed_cleaning == "ours" else []) + [16, 32]
    assert got["logits"][10][SPECK].min() == 5.0
    for f in range(N):
        assert got["logits"][f].dtype == torch.float16 and got["logits"][f].shape == (LOW, LOW)
        again = reproduce(got["logits"][f], "prompt", H, W, info["threshold"], None, 0, cfg, info["raw"][f])
        assert torch.equal(again, masks[f]), f


def test_the_module_sink_is_used_when_none_is_passed(rig, monkeypatch):
    got = []
    rig()
    monkeypatch.setattr(sam3, "LOGITS_SINK", lambda logits, info: got.append(len(logits)))
    sam3.track((FakeModel(), object()), torch.zeros(N, H, W, 3))
    assert got == [N]


@pytest.mark.parametrize("uniform", [False, True])
def test_the_logits_dump_reproduces_box_keypoint_mode(pose_rig, uniform):
    cfg = config(uniform_mask_threshold=uniform, mask_threshold=-1.0)
    dump = {}
    masks, _, _ = pose_rig(cfg, ring=2, logits=dump)
    assert dump["cut"][0] == "prompted" and set(dump["cut"][1:]) == {"propagated"}
    for f in range(N):
        again = reproduce(dump["logits"][f], dump["cut"][f], H, W, sam3.output_cut(cfg), cfg.mask_threshold,
                          FakeTracker.image_size, cfg)
        assert torch.equal(again, masks[f]), f


def test_no_logits_are_collected_unless_asked(rig):
    masks, log, _ = rig()
    assert digest(masks, log) == GOLDEN["single"]


@pytest.fixture
def fake_segments(monkeypatch):
    calls = []
    monkeypatch.setattr(sam3, "segment_by_prompt", lambda *a, **k: calls.append("single") or torch.zeros(2, 8, 8))
    monkeypatch.setattr(sam3, "segment_by_prompt_multi", lambda *a, **k: calls.append("multi") or torch.zeros(2, 8, 8))
    return calls


def not_used(caplog):
    return " ".join(r.getMessage() for r in caplog.records if "not used" in r.getMessage())


def test_the_single_object_switches_are_named_unused_with_several_objects(fake_segments, caplog):
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), max_objects=2,
                   config=config(anchor_output="meta", anchor_matching="meta", unmatched_counting="meta"),
                   logits_sink=lambda *a: None)
    line = not_used(caplog)
    assert "sam3_config.anchor_output (max_objects > 1)" in line and "logits sink" in line
    assert "anchor_matching" not in line and "unmatched_counting" not in line


def test_mask_threshold_is_read_in_prompt_mode_only_when_uniform(fake_segments, caplog):
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), config=config(mask_threshold=0.0))
    assert "sam3_config.mask_threshold" in not_used(caplog)
    caplog.clear()
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), config=config(mask_threshold=0.0, uniform_mask_threshold=True))
    assert "mask_threshold" not in not_used(caplog)


@pytest.mark.parametrize("mask_threshold", [-1.0, 0.0])
@pytest.mark.parametrize("uniform", [False, True])
def test_a_carried_frame_whose_prompt_failed_is_not_re_seeded_forever(monkeypatch, mask_threshold, uniform):
    # box_keypoint: frame 0 is prompted; every later prompt decodes to -0.5 everywhere (empty at
    # a cut of 0, full at -1) and the tracked mask misses the keypoints. Carrying the track on
    # after an empty prompt then broke out at the carried frame itself and came back to it
    # unchanged, forever (seen on a GPU run with mask_threshold 0).
    calls = []

    def decode(model, frame, point_inputs, box_inputs, refine):
        calls.append(1)
        if len(calls) > 200:
            raise RuntimeError("segment_by_pose keeps re-seeding the same frame")
        return torch.full((1, 1, LOW, LOW), 5.0 if len(calls) == 1 else -0.5)

    def propagate(model, frames_chw, first_mask, device, dtype, H_, W_, threshold=0.0, logits_out=None):
        mask = torch.as_tensor(first_mask).bool().cpu().numpy() if not isinstance(first_mask, np.ndarray) else first_mask
        out = [np.asarray(mask, bool).copy() for _ in range(len(frames_chw) - 1)]
        if logits_out is not None:
            logits_out.extend(torch.zeros(LOW, LOW) for _ in out)
        return out

    monkeypatch.setattr(sam3.mm, "load_model_gpu", lambda model: None)
    monkeypatch.setattr(sam3.mm, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sam3.mm, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (FakeSam3(FakeTracker()), None, FakeTracker(), None))
    monkeypatch.setattr(sam3, "_prep_frame", lambda frames, idx, device, dtype, size: idx.start)
    monkeypatch.setattr(sam3, "MultiplexState", lambda *a: object())
    monkeypatch.setattr(sam3, "decode", decode)
    monkeypatch.setattr(sam3, "propagate", propagate)
    monkeypatch.setattr(sam3, "keypoint_recall", lambda *a, **k: 0.0)
    metas, boxes = pose_metas_and_boxes()
    config = sam3.SAM3Config(mask_threshold=mask_threshold, uniform_mask_threshold=uniform)
    result = {}
    masks = sam3.segment_by_pose(FakeModel(), torch.zeros(N, H, W, 3), boxes, metas, config, 0.3, result=result)
    assert masks.shape[0] == N and bool(masks[1:].any())
    if mask_threshold == 0.0:
        assert result.get("kept at low recall"), result


def test_only_the_carried_frame_itself_is_exempt_from_the_recall_re_seed(monkeypatch):
    # frame 1 loses the keypoints and is re-seeded; its prompt is empty, so the track is carried
    # on and frame 1 is kept at low recall.
    # The first window runs to its end without a stop; frame 6 opens the next window with a
    # prompt nobody tried, so losing the keypoints there must re-seed, not be kept.
    calls = []

    def decode(model, frame, point_inputs, box_inputs, refine):
        calls.append(1)
        return torch.full((1, 1, LOW, LOW), -5.0 if len(calls) == 2 else 5.0)

    def propagate(model, frames_chw, first_mask, device, dtype, H_, W_, threshold=0.0, logits_out=None):
        mask = np.asarray(torch.as_tensor(first_mask).bool().cpu().numpy(), bool)
        out = [mask.copy() for _ in range(len(frames_chw) - 1)]
        if logits_out is not None:
            logits_out.extend(torch.zeros(LOW, LOW) for _ in out)
        return out

    metas, boxes = pose_metas_and_boxes()
    frame_of = {id(m["keypoints_body"]): k for k, m in enumerate(metas)}
    monkeypatch.setattr(sam3.mm, "load_model_gpu", lambda model: None)
    monkeypatch.setattr(sam3.mm, "get_torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sam3.mm, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sam3, "_multiplex_parts", lambda model: (FakeSam3(FakeTracker()), None, FakeTracker(), None))
    monkeypatch.setattr(sam3, "_prep_frame", lambda frames, idx, device, dtype, size: idx.start)
    monkeypatch.setattr(sam3, "MultiplexState", lambda *a: object())
    monkeypatch.setattr(sam3, "decode", decode)
    monkeypatch.setattr(sam3, "propagate", propagate)
    monkeypatch.setattr(sam3, "is_anchor", lambda *a, **k: False)
    monkeypatch.setattr(sam3, "keypoint_recall",
                        lambda mask, kps, *a, **k: 0.0 if frame_of[id(kps)] in (1, 6) else 1.0)
    result = {}
    sam3.segment_by_pose(FakeModel(), torch.zeros(N, H, W, 3), boxes, metas,
                         sam3.SAM3Config(reseed_interval=5), 0.3, result=result)
    assert result.get("kept at low recall") == 1, result
    assert result.get("re-seeded early") == 2, result
    assert len(calls) == 3, result


def eager_propagate(sam3_parts, frames_chw, first_mask, device, dtype, H, W, threshold=0.0, logits_out=None):
    """preprocess/sam3.py propagate() before it became a generator, verbatim apart from the
    module prefix: every frame of the window is tracked, then the list is returned."""
    tracker, backbone = sam3_parts.tracker, sam3_parts.detector.backbone["vision_backbone"]
    backbone_fn = sam3._propagation_backbone(backbone)
    N = frames_chw.shape[0]
    size = tracker.image_size
    idev = sam3.mm.intermediate_device()
    initial = (first_mask[None, None].to(device, dtype) * 2 - 1) * sam3.MASK_LOGIT_SCALE
    output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    mux = sam3.MultiplexState(1, tracker.num_multiplex, device, dtype)
    lookback = max(tracker.num_maskmem, tracker.max_obj_ptrs_in_encoder)
    tracked = []
    with torch.inference_mode():
        for f in range(N):
            frame = sam3._prep_frame(frames_chw, slice(f, f + 1), device, dtype, size)
            vision_feats, vision_pos, feat_sizes, high_res, trunk_out = tracker._compute_backbone_frame(
                backbone_fn, frame, frame_idx=f)
            if f == 0:
                current = tracker._condition_with_masks(
                    initial, 0, vision_feats, vision_pos, feat_sizes, high_res, output_dict, N, mux,
                    backbone, frame, trunk_out)
            else:
                current = tracker.track_step(
                    frame_idx=f, is_init_cond_frame=False, current_vision_feats=vision_feats,
                    current_vision_pos_embeds=vision_pos, feat_sizes=feat_sizes, mask_inputs=None,
                    output_dict=output_dict, num_frames=N, propagation_high_res=high_res,
                    multiplex_state=mux, run_mem_encoder=False)
                if logits_out is not None:
                    logits_out.append(sam3.low_res_logits(current["pred_masks"]))
                current["pred_masks"] = sam3.fill_holes_in_mask_scores(current["pred_masks"], max_area=16)
                if tracker.num_maskmem > 0:
                    tracker._deferred_memory_encode(current, 1, vision_feats, feat_sizes, mux, device)
                output_dict["non_cond_frame_outputs"][f] = current
                for old in list(output_dict["non_cond_frame_outputs"]):
                    if old < f - lookback:
                        del output_dict["non_cond_frame_outputs"][old]
            tracked.append((current["pred_masks_high_res"][0, 0] > threshold).to(idev))
        masks = torch.stack(tracked).float()[:, None]
        masks = torch.nn.functional.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)[:, 0] > 0.5
    return [m.cpu().numpy() for m in masks[1:]]


def windows(log):
    """The call log cut into propagate() windows: each opens with its frame-0 conditioning."""
    out = []
    for entry in log:
        if entry[0] == "cond":
            out.append([])
        out[-1].append(entry)
    return out


def failing_recall(frames):
    """keypoint_recall that fails on the given call numbers: the caller's sequence of calls is
    the same whatever propagate computes past its stop, so both runs fail on the same frames."""
    calls = []
    real = sam3.keypoint_recall

    def recall(*args, **kwargs):
        calls.append(1)
        return 0.0 if len(calls) in frames else real(*args, **kwargs)
    return recall


@pytest.mark.parametrize("name, cfg, fail", [
    ("defaults", {}, ()),
    ("max_propagate", dict(max_propagate=10), ()),
    ("recall", {}, (7, 20, 21)),
    ("uniform", dict(uniform_mask_threshold=True, mask_threshold=-1.0, max_propagate=13), (5,)),
])
def test_propagate_stops_where_the_caller_stops_with_the_same_output(pose_rig, monkeypatch, name, cfg, fail):
    runs = {}
    for how in ("lazy", "eager"):
        if how == "eager":
            monkeypatch.setattr(sam3, "propagate", eager_propagate)
        monkeypatch.setattr(sam3, "keypoint_recall", failing_recall(fail))
        dump = {}
        masks, log, result = pose_rig(config(**cfg), ring=1 if name == "defaults" else 2, logits=dump)
        runs[how] = masks, log, result, dump
    (lazy, lazy_log, lazy_result, lazy_dump), (eager, eager_log, eager_result, eager_dump) = runs["lazy"], runs["eager"]
    assert torch.equal(lazy, eager)
    assert lazy_result == eager_result
    assert lazy_dump["cut"] == eager_dump["cut"]
    assert all(torch.equal(a, b) for a, b in zip(lazy_dump["logits"], eager_dump["logits"]))
    # the same windows, each cut off right after the last frame the caller took
    lazy_windows, eager_windows = windows(lazy_log), windows(eager_log)
    assert len(lazy_windows) == len(eager_windows)
    assert all(e[:len(w)] == w for w, e in zip(lazy_windows, eager_windows))
    tracked = sum(entry[0] == "track" for entry in lazy_log)
    assert tracked == lazy_result.get("propagated", 0) + lazy_result.get("re-seeded early", 0)
    if name != "defaults":
        assert tracked < sum(entry[0] == "track" for entry in eager_log)
    if name == "defaults":
        assert digest(lazy, lazy_log) == GOLDEN["pose"]


@pytest.mark.parametrize("size", [(1280, 720), (832, 480)])
def test_per_frame_resize_is_the_batched_one(size):
    # propagate() resizes each frame on its own now; at the tracker's real size the bits must be
    # those of the batched resize it replaced
    torch.manual_seed(0)
    blobs = torch.nn.functional.avg_pool2d(torch.randn(4, 1, 1008, 1008), 31, 1, 15) > 0
    batched = torch.nn.functional.interpolate(blobs.float(), size=size, mode="bilinear", align_corners=False)[:, 0] > 0.5
    for f in range(4):
        one = torch.nn.functional.interpolate(blobs[f, 0].float()[None, None], size=size, mode="bilinear", align_corners=False)[0, 0] > 0.5
        assert torch.equal(one, batched[f])


# --- M4: specks at a cut away from 0 (item 3) -------------------------------------------------

@pytest.mark.parametrize("scenario", [{}, {"multi": 2}])
def test_m4_a_cleaned_speck_stays_out_at_a_negative_cut(rig, scenario):
    # the speck is removed at 0 (-> -0.1): a cut of -1 must not bring it back, on the tracked
    # frames or on the ones tracked backwards before the birth (frames 0-1)
    at_zero, log_zero, _ = rig(speck=True, ring=2, **scenario)
    wide, log_wide, _ = rig(config(uniform_mask_threshold=True), speck=True, ring=2, **scenario)
    for f in (0, 1, 10, 20):
        assert wide[f][SPECK_FRAME].sum() == 0, f
        assert at_zero[f][SPECK_FRAME].sum() == 0, f
    assert log_wide == log_zero                     # the memory is cleaned as before
    assert wide[10].sum() > at_zero[10].sum()       # and -1 still takes in the ring


def test_m4_speck_cleaning_is_the_same_at_a_cut_of_0(rig):
    masks, log, _ = rig(speck=True)
    same, same_log, _ = rig(config(uniform_mask_threshold=True, mask_threshold=0.0), speck=True)
    assert torch.equal(same, masks) and same_log == log


# --- M4: birth and re-anchor frames (item 4) ---------------------------------------------------

def ringed_detections(real):
    """The person from frame 2 with a pinhole and the same -0.5 ring the tracker's masks have:
    what a cut of -1 takes in on a detection."""
    if real < 2:
        return []
    y, x = position(real)
    return [(box(y, x, hole=(y + 3, x + 3), ring=2), 0.9)]


def test_m4_birth_and_anchor_frames_can_show_the_detection_cut_at_the_threshold(rig):
    uniform = dict(uniform_mask_threshold=True)
    masks, log, _ = rig(config(**uniform), detections=ringed_detections, ring=2)
    shown, shown_log, result = rig(config(m4_anchor_frames="output", **uniform), detections=ringed_detections, ring=2)
    assert result["reconditioned"] == 2
    assert shown_log == log                                  # the tracker is conditioned as before
    for f in (2, 16, 32):
        # before: the binarised conditioning mask, which the cut cannot move - the area dips
        assert masks[f].sum() < masks[f + 1].sum()
        y, x = position(f)
        detection = box(y, x, hole=(y + 3, x + 3), ring=2)[None, None]
        assert torch.equal(shown[f], sam3.to_frame_size(sam3.shown_logits(detection, -1.0, 16), H, W, -1.0))
        assert shown[f].sum() > masks[f].sum()
    others = [f for f in range(N) if f not in (2, 16, 32)]
    assert torch.equal(shown[others], masks[others])


@pytest.mark.parametrize("cfg", [dict(), dict(mask_threshold=0.0)])
def test_m4_anchor_frames_option_is_read_only_on_the_m4_path(rig, cfg):
    masks, log, _ = rig(config(**cfg), detections=ringed_detections, ring=2)
    same, same_log, _ = rig(config(m4_anchor_frames="output", **cfg), detections=ringed_detections, ring=2)
    assert torch.equal(same, masks) and same_log == log


def test_m4_anchor_frames_option_is_named_unused_without_uniform(fake_segments, caplog):
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), config=config(m4_anchor_frames="output"))
    assert "sam3_config.m4_anchor_frames (uniform_mask_threshold off)" in not_used(caplog)
    caplog.clear()
    with caplog.at_level("INFO"):
        sam3.track((None, None), torch.zeros(2, 8, 8, 3), max_objects=2, config=config(m4_anchor_frames="output",
                                                                                     uniform_mask_threshold=True))
    assert "sam3_config.m4_anchor_frames (max_objects > 1)" in not_used(caplog)


@pytest.mark.parametrize("option", ["mask", "output"])
def test_the_logits_dump_names_birth_and_anchor_frames(rig, option):
    got = {}
    cfg = config(uniform_mask_threshold=True, m4_anchor_frames=option)
    rig(cfg, detections=ringed_detections, ring=2)
    masks = sam3.track((FakeModel(), object()), torch.zeros(N, H, W, 3), config=cfg,
                       logits_sink=lambda logits, info: got.update(logits=logits, info=info))
    info = got["info"]
    assert info["cut"][2] == "birth" and info["cut"][16] == info["cut"][32] == "anchor"
    assert {info["cut"][f] for f in range(N) if f not in (2, 16, 32)} == {"prompt"}
    for f in (2, 16, 32):
        y, x = position(f)
        if option == "output":   # the logits the output was cut from: the detection's, before cleaning
            assert info["raw"][f] and torch.equal(got["logits"][f], box(y, x, hole=(y + 3, x + 3), ring=2).half())
        else:                    # the conditioning mask
            assert not info["raw"][f] and set(got["logits"][f].unique().tolist()) == {-10.0, 10.0}
    for f in range(N):
        again = reproduce(got["logits"][f], "prompt", H, W, info["threshold"], None, 0, cfg, info["raw"][f])
        assert torch.equal(again, masks[f]), f
