"""G17 (refactor plan 8.2): SAM 3.1 Multiplex pinned beyond the GOLDEN digests of
test_sam3_1_multiplex_ab.py. Recorded once from the production code at 3ca2c3b with
BCV_GOLDEN_RECORD=1 into tests/goldens/test_sam3_1_multiplex_golden.json (tests/golden.py); a
value that changes fails the step that changed it.

On that file's scripted stand-ins for core's tracker and SAM 3.1's parts (FakeTracker, FakeSam3,
the scripted detections), extended with a tracker whose tracks follow the people of a scripted
scene, and with stand-ins that keep everything they are handed:
- `track` end to end in prompt, multi-object and box_keypoint mode: the masks, the result dict,
  every call the stand-ins received (the tracker's log, the frames given to `_prep_frame` and to
  the decoder, the prompts and the boxes), the ProgressBar totals and updates, the log (the "not
  used" lines and the closing `log.step` line with the result counts and the coverage string),
  the logits sink, and every error text;
- prompt mode under the A/B switches (A6, A7); several objects: the cap, a duplicate,
  two people crossing, one under another, object_index;
- box_keypoint mode's branches GOLDEN["pose"] does not reach: the anchor re-seed, undetected
  frames, temporal off, refine off, max_propagate, the annexed background, the carried frame;
- the primitives on stand-ins of the model's parts: `detect_person`, `encode_prompt`, `decode`,
  `_multiplex_parts`, and the checkpoint loader's cache and download;
- the config's choice error, the mask operations on irregular shapes and the prompt geometry on
  seeded random inputs.

The mask cleaning and the point sampling run cv2, so these goldens are build-dependent: they hold
in the ComfyUI venv they were recorded in. Needs ComfyUI importable, like the other SAM 3.1
Multiplex tests."""
import itertools
import json
import logging
import os

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from golden import check, digest, log_text  # noqa: E402
from sam3_1_multiplex_fakes import sam3  # noqa: E402
from test_sam3_1_multiplex_ab import (  # noqa: E402
    H, LOW, N, W, FakeSam3, FakeTracker, anchor_taller, box, default_detections, position, probation_detections,
    two_agreeing, two_people)

MARK = 64.0   # frame f carries f / MARK in its first value, so a stand-in can tell which frame it holds


def golden(key, value):
    check(__file__, key, value)


def clip(n=N, channels=3, seed=0):
    """Seeded [n, H, W, channels] images whose first value marks the frame number."""
    images = torch.rand(n, H, W, channels, generator=torch.Generator().manual_seed(seed))
    images[:, 0, 0, 0] = torch.arange(n) / MARK
    return images


def model_of(dtype):
    """The loaded model as the segment functions read it: its dtype."""
    inner = type("Inner", (), {"get_dtype": staticmethod(lambda: dtype)})
    return type("Model", (), {"model": inner})()


class Clip:
    """The checkpoint's text encoder; the prompt stand-in only records that one was passed."""


class Mux:
    """MultiplexState, numbered in creation order so a track can be told apart."""

    def __init__(self, number, args):
        self.number, self.args = number, args

    def __repr__(self):
        return f"Mux({self.number})"


def progress_bar(bars):
    """A ProgressBar class whose bars append (total, updates) to `bars`."""
    class Recorder:
        def __init__(self, total):
            self.updates = []
            bars.append((total, self.updates))

        def update(self, value):
            self.updates.append(value)
    return Recorder


def pack_records(caplog):
    return [r for r in caplog.records if r.name in ("BCVideoNodes", "root")]


# --- a tracker that follows a scripted scene ---------------------------------------------------

class Person:
    """One person of a scene: `path(real)` is (y, x, h, w) on the [LOW, LOW] grid, or None where
    the person is hidden; `detected(real)` whether the detector finds them there."""

    def __init__(self, path, value=10.0, score=0.9, object_score=5.0, ring=0, detected=lambda real: True):
        self.path, self.value, self.score, self.object_score, self.ring = path, value, score, object_score, ring
        self.detected = detected

    def logits(self, real):
        where = self.path(real)
        if where is None:
            return None
        y, x, h, w = where
        return box(y, x, h=h, w=w, value=self.value, ring=self.ring)


class Scene:
    def __init__(self, *people):
        self.people = people

    def detections(self, real):
        """The people found on frame `real`, best score first."""
        found = [(p.logits(real), p.score) for p in self.people if p.detected(real) and p.path(real) is not None]
        return sorted(found, key=lambda d: -d[1])


class SceneTracker(FakeTracker):
    """FakeTracker whose tracks follow a scene: a track (told apart by its multiplex state)
    follows the person its first conditioning mask overlapped most, and its propagated mask is
    that person's logits on the frame being tracked, empty where the person is hidden."""

    def __init__(self, scene, **kwargs):
        super().__init__(**kwargs)
        self.scene = scene
        self.person = {}

    def _condition_with_masks(self, masks, frame_idx, vision_feats, vision_pos, feat_sizes, high_res, output_dict,
                              N, mux, backbone, frame, trunk_out, threshold=0.5):
        current = super()._condition_with_masks(masks, frame_idx, vision_feats, vision_pos, feat_sizes, high_res,
                                                output_dict, N, mux, backbone, frame, trunk_out, threshold)
        if mux.number not in self.person:
            binary = current["pred_masks"][0, 0] > 0
            overlap = [-1.0 if logits is None else float((binary & (logits > 0)).sum())
                       for logits in (p.logits(vision_feats[0]) for p in self.scene.people)]
            self.person[mux.number] = overlap.index(max(overlap))
        return current

    def track_step(self, **kwargs):
        out = super().track_step(**kwargs)
        person = self.scene.people[self.person[kwargs["multiplex_state"].number]]
        logits = person.logits(kwargs["current_vision_feats"][0])
        score = person.object_score
        if logits is None:
            logits, score = torch.full((LOW, LOW), -10.0), -5.0
        out["pred_masks"] = out["pred_masks_high_res"] = logits[None, None]
        out["object_score_logits"] = torch.tensor([[score]])
        return out


def moving(y, x0, x1, h, w, hidden=()):
    """A path from column x0 to x1 over the clip, hidden on the `hidden` frames."""
    return lambda real: None if real in hidden else (y, x0 + round((x1 - x0) * real / (N - 1)), h, w)


def still(y, x, h, w):
    return lambda real: (y, x, h, w)


def walking(real):
    """The A/B file's person: position(real), 8 x 8."""
    return (*position(real), 8, 8)


# three people side by side, the best-scoring first
THREE = Scene(Person(moving(1, 1, 3, 5, 4), score=0.9), Person(still(1, 8, 5, 4), score=0.8, object_score=4.0),
              Person(still(9, 4, 5, 6), score=0.7, object_score=3.0))
# a second detection on frame 3 whose track then lies on the first person's: a duplicate
GHOST = Scene(Person(walking), Person(lambda real: (12, 12, 3, 3) if real <= 3 else walking(real), score=0.8,
                                      object_score=4.0, detected=lambda real: real == 3))
# two people crossing once both are out of probation (they overlap from frame 15, fully on 18-21);
# each is hidden for two frames first, the second one more recently
CROSSING = Scene(Person(moving(5, 0, 12, 6, 4, hidden=(4, 5)), ring=1),
                 Person(moving(5, 12, 0, 6, 4, hidden=(8, 9)), value=8.0, score=0.85, object_score=3.0, ring=1))
# a small person that walks into a bigger, stronger one and out again
SHRINK = Scene(Person(still(2, 2, 10, 10)),
               Person(lambda real: (5, 5, 3, 3) if 12 <= real <= 30 else (13, 13, 3, 3), value=5.0, score=0.8,
                      object_score=3.0))
# a second person the detector finds from frame 6 on
LATE = Scene(Person(walking), Person(still(12, 12, 3, 3), score=0.7, object_score=3.0, detected=lambda real: real >= 6))


# --- box_keypoint mode's pose ------------------------------------------------------------------

# the prompt keypoints, then the four limb ends between them
BODY = (0, 1, 2, 5, 8, 11, 10, 13, 18, 19, 3, 6, 9, 12)


def pose_data(kp_count=len(BODY), undetected=(), lost=(), unsure=(), margin=10, seed=3):
    """pose_data of the A/B file's person: each frame's box `margin` pixels around it (score -1
    on `undetected` frames) and `kp_count` confident body keypoints inside it, below it on the
    `lost` frames, at confidence 0 on the `unsure` ones."""
    rng = np.random.default_rng(seed)
    metas, detections = [], []
    for real in range(N):
        y, x = position(real)
        kps = np.zeros((20, 3))
        for k in BODY[:kp_count]:
            if real in lost:
                px, py = 2 * x + rng.uniform(2, 12), 2 * y + 17 + rng.uniform(0, 2.5)
            else:
                px, py = 2 * x + rng.uniform(2, 12), 2 * y + 2 + rng.uniform(0, 10)
            kps[k] = [px / W, py / H, 0.0 if real in unsure else rng.uniform(0.5, 1.0)]
        metas.append({"keypoints_body": kps})
        detections.append({"bbox": [2 * x - margin, 2 * y - margin, 2 * x + 16 + margin, 2 * y + 16 + margin],
                           "score": -1.0 if real in undetected else 0.9})
    return {"pose_metas_original": metas, "detections": detections, "pose_config": {"min_keypoint_conf": 0.3}}


# --- the rig -----------------------------------------------------------------------------------

class Rig:
    """Installs the stand-ins on the names the code reads, and keeps what they were handed."""

    def __init__(self, monkeypatch, caplog):
        self.caplog = caplog
        self.calls, self.bars = [], []
        self.tracker, self.detections, self.ring, self.empty_decodes = None, None, 1, ()
        self.numbers, self.result, self.lines = itertools.count(), None, []
        monkeypatch.setattr(sam3.mm, "load_model_gpu", lambda model: self.calls.append(("load_model_gpu", type(model).__name__)))
        monkeypatch.setattr(sam3.mm, "get_torch_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(sam3.mm, "intermediate_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(sam3, "_multiplex_parts",
                            lambda model: (FakeSam3(self.tracker), "detector", self.tracker, "backbone"))
        monkeypatch.setattr(sam3, "MultiplexState", lambda *args: self.mux(next(self.numbers), args))
        monkeypatch.setattr(sam3, "_prep_frame", self.prep_frame)
        monkeypatch.setattr(sam3, "encode_prompt", self.encode_prompt)
        monkeypatch.setattr(sam3, "detect_person", self.detect)
        monkeypatch.setattr(sam3, "decode", self.decode)
        monkeypatch.setattr(sam3, "ProgressBar", progress_bar(self.bars))
        for name in ("segment_by_prompt", "segment_by_prompt_multi", "segment_by_pose"):
            monkeypatch.setattr(sam3, name, self.keeping_result(getattr(sam3, name)))

    def keeping_result(self, segment):
        """`segment` itself, keeping the result dict `track` hands it (and fills in after)."""
        def run(*args, **kwargs):
            self.result = kwargs.get("result")
            return segment(*args, **kwargs)
        return run

    def mux(self, number, args):
        self.calls.append(("MultiplexState", number, repr(args)))
        return Mux(number, args)

    def prep_frame(self, frames, idx, device, dtype, size):
        real = int(round(float(frames[idx.start, 0, 0, 0]) * MARK))
        self.calls.append(("_prep_frame", idx.start, real, digest(frames[idx]), str(device), str(dtype), size))
        return real

    def encode_prompt(self, clip, detector, prompt, device, dtype):
        self.calls.append(("encode_prompt", type(clip).__name__, detector, prompt, str(device), str(dtype)))
        return ("embedding", prompt), "text_mask"

    def detect(self, detector, backbone, trunk_out, embedding, text_mask, config):
        self.calls.append(("detect_person", detector, backbone, trunk_out, embedding, text_mask))
        found = self.detections(trunk_out)
        if not found:
            return torch.zeros(0, LOW, LOW), torch.zeros(0)
        return torch.stack([m for m, _ in found]), torch.tensor([s for _, s in found])

    def decode(self, model, frame, point_inputs, box_inputs, refine):
        """The person's box around the prompt box's centre, or nothing on the `empty_decodes`
        calls; digests the decoder input frame and the prompt."""
        n = sum(1 for c in self.calls if c[0] == "decode")
        points = None if point_inputs is None else (digest(point_inputs["point_coords"]),
                                                    digest(point_inputs["point_labels"]))
        self.calls.append(("decode", type(model).__name__, digest(frame),
                           None if box_inputs is None else digest(box_inputs), points, refine))
        if n in self.empty_decodes:
            return torch.full((1, 1, LOW, LOW), -5.0)
        size = sam3.SAM3_1_MULTIPLEX_SIZE
        cx = float(box_inputs[0, :, 0].float().mean()) * W / size
        cy = float(box_inputs[0, :, 1].float().mean()) * H / size
        return box(round(cy) // 2 - 4, round(cx) // 2 - 4, ring=self.ring)[None, None]

    def track(self, tracker, images, detections=default_detections, model_dtype=torch.float32, text_encoder=True,
              sink=False, ring=1, empty_decodes=(), **kwargs):
        """One `track` run; what it produced and everything the stand-ins saw, as golden values.
        A ValueError or TypeError is kept as "error"."""
        self.tracker, self.detections, self.ring, self.empty_decodes = tracker, detections, ring, empty_decodes
        self.calls.clear()
        self.bars.clear()
        self.numbers, self.result = itertools.count(), None
        received = []
        if sink:
            kwargs["logits_sink"] = lambda logits, info: received.append((logits, info))
        self.caplog.clear()
        masks, error = None, None
        try:
            masks = sam3.track((model_of(model_dtype), Clip() if text_encoder else None), images, **kwargs)
        except (ValueError, TypeError) as e:
            error = f"{type(e).__name__}: {e}"
        records = pack_records(self.caplog)
        self.lines = log_text(records).split("\n")
        out = {"masks": None if masks is None else digest(masks), "result": repr(self.result),
               "tracker": digest(repr(tracker.log)),
               "reads": digest(repr(tracker.reads)), "calls": digest(repr(self.calls)), "progress": repr(self.bars),
               "log": digest(log_text(records)), "closing": log_text(records[-1:]), "error": error}
        if sink:
            out["sink"] = [(digest(repr([None if low is None else digest(low) for low in logits])), digest(repr(info)))
                           for logits, info in received]
        return out


@pytest.fixture
def rig(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="BCVideoNodes")
    return Rig(monkeypatch, caplog)


def config(**kwargs):
    return sam3.SAM3_1MultiplexConfig(**kwargs)


# --- prompt mode, one object: the defaults on each scenario, and the A/B switches ----------------

GONE = {20: -5.0, 21: -5.0, 22: -5.0}
SINGLE = {
    "a2_ours_taller": dict(detections=anchor_taller),
    "a3_ours_every_4": dict(config=dict(recondition_every=4)),
    "a4_ours_absent": dict(config=dict(recondition_every=100), tracker=dict(scores=GONE)),
    "a6_ours_two_agreeing": dict(detections=two_agreeing),
    "a6_meta": dict(config=dict(anchor_matching="meta"), detections=two_agreeing),
    "a7_meta": dict(config=dict(unmatched_counting="meta"), detections=probation_detections,
                    tracker=dict(empty=range(3, 10))),
    "probation_no_sink": dict(detections=probation_detections, tracker=dict(empty=range(3, 10)), sink=False),
    "no_person_no_sink": dict(detections=lambda real: [], sink=False),
    "rgba_fp16": dict(channels=4, model_dtype=torch.float16),
}


@pytest.mark.parametrize("name", list(SINGLE))
def test_prompt_mode_one_object(rig, name):
    kwargs = dict(SINGLE[name])
    tracker = FakeTracker(**kwargs.pop("tracker", {}))
    cfg = config(**kwargs.pop("config", {}))
    images = clip(channels=kwargs.pop("channels", 3), seed=1)
    out = rig.track(tracker, images, config=cfg, sink=kwargs.pop("sink", True), **kwargs)
    assert out["error"] is None
    golden(f"prompt/{name}", out)


# --- prompt mode, several objects --------------------------------------------------------------

MULTI = {
    "cap_2_of_3": dict(scene=THREE, max_objects=2),
    "cap_3_of_3": dict(scene=THREE, max_objects=3),
    "cap_3_object_1": dict(scene=THREE, max_objects=3, object_index=1),
    "cap_2_object_0": dict(scene=THREE, max_objects=2, object_index=0),
    "duplicate": dict(scene=GHOST, max_objects=2),
    "crossing": dict(scene=CROSSING, max_objects=2),
    "crossing_meta": dict(scene=CROSSING, max_objects=2, config=dict(anchor_matching="meta", unmatched_counting="meta")),
    "shrink": dict(scene=SHRINK, max_objects=2),
    "late": dict(scene=LATE, max_objects=2),
    "late_object_1": dict(scene=LATE, max_objects=2, object_index=1),
    "probation": dict(max_objects=2, detections=probation_detections, tracker=dict(empty=range(3, 10))),
    "two_meta": dict(max_objects=2, detections=two_people, config=dict(anchor_matching="meta",
                                                                       unmatched_counting="meta")),
}


@pytest.mark.parametrize("name", list(MULTI))
def test_prompt_mode_several_objects(rig, name):
    kwargs = dict(MULTI[name])
    scene = kwargs.pop("scene", None)
    if scene is not None:
        tracker, kwargs["detections"] = SceneTracker(scene), scene.detections
    else:
        tracker = FakeTracker(**kwargs.pop("tracker", {}))
    out = rig.track(tracker, clip(seed=1), config=config(**kwargs.pop("config", {})), **kwargs)
    assert out["error"] is None
    golden(f"multi/{name}", out)


# --- box_keypoint mode -------------------------------------------------------------------------

POSE = {
    "anchor_reseed_annexed": dict(),
    "undetected_and_carried": dict(data=dict(undetected={0, 1, 24, 25, 33}, lost={33})),
    "temporal_off": dict(config=dict(temporal=False), data=dict(undetected={5})),
    "refine_off": dict(config=dict(refine=False)),
    "max_propagate": dict(config=dict(max_propagate=10), data=dict(kp_count=5)),
    "empty_prompt_carried": dict(config=dict(max_propagate=10), data=dict(kp_count=5), empty_decodes={1}),
    "keypoints_lost": dict(data=dict(lost={10, 11})),
    "no_keypoints_on_frame_0": dict(data=dict(unsure={0})),
    "hand_points": dict(hand=True),
    "hand_points_frame_0_undetected": dict(data=dict(undetected={0}), positive_coords='[{"x": 20, "y": 16}]'),
    "supplied_bboxes": dict(bboxes=[4, 8, 28, 24]),
    "fp16_rgba": dict(model_dtype=torch.float16, channels=4),
}


@pytest.mark.parametrize("name", list(POSE))
def test_box_keypoint_mode(rig, name):
    kwargs = dict(POSE[name])
    data = pose_data(**kwargs.pop("data", {}))
    if kwargs.pop("hand", False):
        # a hand-placed negative half a pixel from the nose, which frame 0 prompts as a positive
        nose = data["pose_metas_original"][0]["keypoints_body"][0]
        kwargs["negative_coords"] = json.dumps([{"x": float(nose[0] * W + 0.5), "y": float(nose[1] * H)}])
        kwargs["positive_coords"] = '[{"x": 12, "y": 14}]'
    out = rig.track(FakeTracker(), clip(channels=kwargs.pop("channels", 3), seed=2), mode="box_keypoint",
                    pose_data=data, config=config(**kwargs.pop("config", {})), sink=True, **kwargs)
    assert out["error"] is None
    golden(f"box_keypoint/{name}", out)


# --- what a mode does not read -----------------------------------------------------------------

def test_prompt_mode_names_what_it_does_not_read(rig):
    base = dict(pose_data=pose_data(), bboxes=[2, 2, 6, 6], positive_coords='[{"x": 1, "y": 1}]', negative_coords="[]")
    runs = {
        "one_object": dict(config=config(reseed_interval=5, mask_threshold=0.5, assoc_iou=0.2, negative_points=4),
                           **base),
        "several_objects": dict(config=config(memory_gap=3), max_objects=2, detections=two_people, sink=True),
    }
    for name, kwargs in runs.items():
        out = rig.track(FakeTracker(), clip(seed=1), **kwargs)
        assert out["error"] is None
        out["not_used"] = [line for line in rig.lines if "not used" in line]
        golden(f"not_used/prompt/{name}", out)


def test_box_keypoint_mode_names_what_it_does_not_read(rig):
    out = rig.track(FakeTracker(), clip(seed=2), mode="box_keypoint", pose_data=pose_data(), prompt="a dog",
                    max_objects=3, object_index=2, bboxes=[4, 8, 28, 24],
                    config=config(birth_threshold=0.6, new_object_threshold=0.7, anchor_matching="meta", temporal=False,
                                  reseed_interval=5, negative_points=4))
    assert out["error"] is None
    out["not_used"] = [line for line in rig.lines if "not used" in line]
    golden("not_used/box_keypoint", out)


# --- every error text of track -----------------------------------------------------------------

def without(data, key):
    return {k: v for k, v in data.items() if k != key}


ERRORS = {
    "mode": dict(mode="boxes"),
    "max_objects_zero": dict(max_objects=0),
    "max_objects_float": dict(max_objects=1.0),
    "object_index_below_minus_1": dict(object_index=-2),
    "object_index_float": dict(object_index=0.0),
    "object_index_past_max_objects": dict(max_objects=2, object_index=2),
    "config_type": dict(config={}),
    "images_dims": dict(images=torch.zeros(2, 8, 8)),
    "images_channels": dict(images=torch.zeros(2, 8, 8, 2)),
    "empty_prompt": dict(prompt="  "),
    "no_text_encoder": dict(text_encoder=False),
    "no_text_encoder_several_objects": dict(text_encoder=False, max_objects=2),
    "no_pose_data": dict(mode="box_keypoint"),
    "pose_data_type": dict(mode="box_keypoint", pose_data=[]),
    "pose_data_keys": dict(mode="box_keypoint", pose_data={"pose_config": {}, "detections": []}),
    "pose_data_frames": dict(mode="box_keypoint", pose_data=pose_data(), images=clip(n=2)),
    "pose_data_no_pose_config": dict(mode="box_keypoint", pose_data=without(pose_data(), "pose_config")),
    "coords_not_json": dict(mode="box_keypoint", pose_data=pose_data(), positive_coords="{x"),
    "coords_points_shape": dict(mode="box_keypoint", pose_data=pose_data(), negative_coords='{"points": [[0.5]]}'),
    "coords_list_shape": dict(mode="box_keypoint", pose_data=pose_data(), positive_coords='[{"x": 1}]'),
    "coords_other": dict(mode="box_keypoint", pose_data=pose_data(), positive_coords="7"),
    "coords_outside": dict(mode="box_keypoint", pose_data=pose_data(), negative_coords='[{"x": 32, "y": 1}]'),
    "coords_outside_normalised": dict(mode="box_keypoint", pose_data=pose_data(),
                                      positive_coords='{"points": [[0.5, 1.0]]}'),
    "bboxes": dict(mode="box_keypoint", pose_data=pose_data(), bboxes=[5, 5, 5, 10]),
    "bboxes_not_json": dict(mode="box_keypoint", pose_data=pose_data(), bboxes="not json"),
    "bboxes_short_box": dict(mode="box_keypoint", pose_data=pose_data(), bboxes=[(0, 0, 5)]),
    "bboxes_not_a_box": dict(mode="box_keypoint", pose_data=pose_data(), bboxes=7),
    "bboxes_count": dict(mode="box_keypoint", pose_data=pose_data(), bboxes=[[4, 8, 28, 24]] * 2),
    "object_index_0_nothing_tracked": dict(object_index=0, detections=lambda real: []),
    "object_index_past_tracked": dict(scene=LATE, max_objects=3, object_index=2),
}


@pytest.mark.parametrize("name", list(ERRORS))
def test_track_error_texts(rig, name):
    kwargs = dict(ERRORS[name])
    scene = kwargs.pop("scene", None)
    if scene is not None:
        tracker, kwargs["detections"] = SceneTracker(scene), scene.detections
    else:
        tracker = FakeTracker()
    out = rig.track(tracker, kwargs.pop("images", clip(seed=4)), **kwargs)
    assert out["error"] is not None
    golden(f"error/{name}", out)


def test_config_choice_error_names_the_class():
    for name, value in (("anchor_matching", "theirs"), ("unmatched_counting", None)):
        with pytest.raises(ValueError) as caught:
            config(**{name: value})
        golden(f"config_choice/{name}", str(caught.value))


# --- the primitives on stand-ins of the model's parts --------------------------------------------

class Detector:
    """SAM 3.1's detector as detect_person drives it: `_detect` answers with the given queries."""

    def __init__(self, scores, masks, presence):
        self.answer = (None, scores, masks, {"presence": presence})
        self.calls = []

    def _detect(self, features, positions, embedding, text_mask):
        self.calls.append(([digest(f) for f in features], [digest(p) for p in positions], digest(embedding),
                           digest(text_mask)))
        return self.answer


class Backbone:
    """The detector's vision backbone: two necks, and a position encoding in float64 that
    detect_person casts to the features' dtype."""
    convs = (lambda t: t * 2.0, lambda t: t[..., ::2, ::2] + 1.0)

    @staticmethod
    def position_encoding(features):
        return features.double() * 0.5 - 0.25


def queries(seed, count=10, h=24, w=20):
    """(scores, masks, presence) of `count` queries: boxes of logits at seeded places, with noise."""
    g = torch.Generator().manual_seed(seed)
    masks = torch.full((1, count, h, w), -6.0)
    for q in range(count):
        y, x = int(torch.randint(0, h - 4, (1,), generator=g)), int(torch.randint(0, w - 4, (1,), generator=g))
        bh, bw = int(torch.randint(3, h - y + 1, (1,), generator=g)), int(torch.randint(3, w - x + 1, (1,), generator=g))
        masks[0, q, y:y + bh, x:x + bw] = float(torch.rand(1, generator=g)) * 8 + 0.5
    masks += torch.randn(masks.shape, generator=g) * 0.5
    scores = (torch.randn(1, count, generator=g) * 3).half()
    presence = torch.randn(1, 1, generator=g).half()
    return scores, masks, presence


def test_detect_person_scores_and_nms():
    configs = {"default": {}, "all_nms_0.5": dict(detection_threshold=0.0, nms_iou=0.5),
               "all_nms_1": dict(detection_threshold=0.0, nms_iou=1.0), "strict": dict(detection_threshold=0.99)}
    for seed in range(4):
        g = torch.Generator().manual_seed(50 + seed)
        dtype = torch.float16 if seed == 3 else torch.float32
        trunk_out = torch.rand(1, 3, 16, 12, generator=g).to(dtype)
        embedding, text_mask = torch.randn(1, 5, 8, generator=g), torch.rand(1, 5, generator=g) > 0.3
        for name, kwargs in configs.items():
            detector = Detector(*queries(seed))
            masks, scores = sam3.detect_person(detector, Backbone(), trunk_out, embedding, text_mask, config(**kwargs))
            golden(f"detect_person/seed{seed}/{name}",
                   {"masks": digest(masks), "scores": digest(scores), "calls": digest(repr(detector.calls))})


def test_encode_prompt():
    for with_mask in (True, False):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            g = torch.Generator().manual_seed(7)
            calls = []
            embedding = torch.randn(1, 7, 12, generator=g)
            extra = {"pooled_output": None}
            if with_mask:
                extra["attention_mask"] = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])

            class TextEncoder:
                def tokenize(self, text):
                    calls.append(("tokenize", text))
                    return {"l": [text]}

                def encode_from_tokens_scheduled(self, tokens):
                    calls.append(("encode_from_tokens_scheduled", repr(tokens)))
                    return [[embedding, extra]]

            def resizer(e):
                calls.append(("resizer", digest(e)))
                return e * 3.0 - 1.0

            detector = type("D", (), {"backbone": {"language_backbone": {"resizer": resizer}}})()
            got, mask = sam3.encode_prompt(TextEncoder(), detector, sam3.PROMPT, torch.device("cpu"), dtype)
            golden(f"encode_prompt/{'mask' if with_mask else 'no_mask'}/{dtype}",
                   {"embedding": digest(got), "mask": digest(mask), "calls": digest(repr(calls))})


class Heads:
    """The tracker's SAM heads: logits that depend on everything they are handed."""

    def __init__(self, **embeds):
        for name, value in embeds.items():
            setattr(self, name, value)
        self.calls = []

    def _forward_sam_heads(self, backbone_features, point_inputs, mask_inputs, box_inputs, high_res_features,
                           multimask_output):
        self.calls.append((digest(backbone_features), point_inputs and {k: digest(v) for k, v in point_inputs.items()},
                           None if mask_inputs is None else digest(mask_inputs),
                           None if box_inputs is None else digest(box_inputs),
                           [digest(f) for f in high_res_features], multimask_output))
        logits = backbone_features.float().sum(dim=1, keepdim=True) + len(self.calls)
        if mask_inputs is not None:
            logits = logits + mask_inputs.float() * 0.5
        return None, logits, None, None


class VisionBackbone:
    def __init__(self, feats):
        self.feats, self.calls = feats, []

    def __call__(self, frame, tracker_mode=None):
        self.calls.append((digest(frame), tracker_mode))
        return None, None, self.feats, None


def test_decode_no_mem_embed_refine_and_point_count():
    g = torch.Generator().manual_seed(11)
    feats = [torch.randn(1, 4, 32, 32, generator=g).half(), torch.randn(1, 4, 16, 16, generator=g).half(),
             torch.randn(1, 8, 8, 8, generator=g).half()]
    interactive, plain = torch.randn(1, 1, 8, generator=g), torch.randn(1, 1, 8, generator=g)
    frame = torch.rand(1, 3, 64, 64, generator=g)
    box_inputs = torch.tensor([[[10.0, 12.0], [50.0, 60.0]]])
    embeds = {"interactivity": dict(interactivity_no_mem_embed=interactive, no_mem_embed=plain),
              "interactivity_none": dict(interactivity_no_mem_embed=None, no_mem_embed=plain),
              "no_mem": dict(no_mem_embed=plain), "neither": {}}
    points = {0: None}
    for count in (1, 3):
        points[count] = {"point_coords": torch.rand(1, count, 2, generator=g) * 1008,
                         "point_labels": torch.tensor([[1, 0, 1][:count]], dtype=torch.int32)}
    for name, attrs in embeds.items():
        for refine in (True, False):
            for count, point_inputs in points.items():
                heads, backbone = Heads(**attrs), VisionBackbone(feats)
                model = type("Sam", (), {"tracker": heads})()
                model.detector = type("Detector", (), {"backbone": {"vision_backbone": backbone}})()
                logits = sam3.decode(model, frame, point_inputs, box_inputs, refine)
                golden(f"decode/{name}/refine_{refine}/points_{count}",
                       {"logits": digest(logits), "heads": digest(repr(heads.calls)),
                        "backbone": digest(repr(backbone.calls))})


def parts_model(sam):
    model = type("Model", (), {})()
    model.model = type("Inner", (), {})()
    model.model.diffusion_model = sam
    return model


def test_multiplex_parts_errors():
    def sam(detector=None, tracker=None, name="Checkpoint"):
        return type(name, (), {"detector": detector, "tracker": tracker})()

    def detector(backbone):
        return type("Detector", (), {"backbone": backbone})()

    multiplex = type("Vision", (), {"multiplex": True})()
    single = type("Vision", (), {"multiplex": False})()
    primitives = ("_compute_backbone_frame", "track_step", "_condition_with_masks", "_deferred_memory_encode",
                  "_forward_sam_heads")
    complete = type("Tracker", (), {n: None for n in primitives})()
    partial = type("Tracker", (), {n: None for n in primitives[:3]})()
    cases = {
        "no_detector": sam(name="PlainModel"),
        "backbone_list": sam(detector([multiplex])),
        "no_vision_backbone": sam(detector({"language_backbone": None})),
        "module_dict_not_multiplex": sam(detector(torch.nn.ModuleDict({"vision_backbone": torch.nn.Identity()}))),
        "not_multiplex": sam(detector({"vision_backbone": single})),
        "tracker_missing_two": sam(detector({"vision_backbone": multiplex}), partial),
        "no_tracker": sam(detector({"vision_backbone": multiplex})),
    }
    for name, model in cases.items():
        with pytest.raises(ValueError) as caught:
            sam3._multiplex_parts(parts_model(model))
        golden(f"multiplex_parts/{name}", str(caught.value))
    whole = sam(detector({"vision_backbone": multiplex, "language_backbone": None}), complete)
    parts = sam3._multiplex_parts(parts_model(whole))
    assert parts[0] is whole and parts[1] is whole.detector and parts[2] is complete and parts[3] is multiplex
    golden("multiplex_parts/ok", repr([type(p).__name__ for p in parts]))


def test_load_caches_and_downloads_the_default_checkpoint(monkeypatch, caplog, tmp_path):
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="BCVideoNodes")
    for key in ("name", "model", "clip"):
        monkeypatch.setitem(sam3._loaded, key, None)
    folder = str(tmp_path / "checkpoints")
    present, calls = {}, []

    def relative(path):
        return os.path.relpath(path, tmp_path)

    def get_full_path(kind, name):
        calls.append(("get_full_path", kind, name))
        return present.get(name)

    def get_folder_paths(kind):
        calls.append(("get_folder_paths", kind))
        return [folder]

    def download(url, path):
        calls.append(("download", url, relative(path), os.path.isdir(os.path.dirname(path))))
        present[os.path.basename(path)] = path

    def load_checkpoint(path, **kwargs):
        calls.append(("load_checkpoint_guess_config", relative(path), kwargs))
        name = os.path.basename(path)
        if name == "broken.safetensors":
            raise RuntimeError("not a checkpoint")
        return f"model:{name}", f"clip:{name}", "vae", "clipvision"

    monkeypatch.setattr(sam3.folder_paths, "get_full_path", get_full_path)
    monkeypatch.setattr(sam3.folder_paths, "get_folder_paths", get_folder_paths)
    monkeypatch.setattr(sam3, "download", download)
    monkeypatch.setattr(sam3.comfy_sd, "load_checkpoint_guess_config", load_checkpoint)
    present["other.safetensors"] = os.path.join(folder, "other.safetensors")
    present["broken.safetensors"] = os.path.join(folder, "broken.safetensors")

    steps = []
    for name in (None, None, "other.safetensors", "missing.safetensors", None, "broken.safetensors",
                 "other.safetensors"):
        try:
            got = sam3.load_sam3_1_multiplex() if name is None else sam3.load_sam3_1_multiplex(name)
        except (FileNotFoundError, RuntimeError) as e:
            got = f"{type(e).__name__}: {e}"
        steps.append((name, got, dict(sam3._loaded), len(calls)))
    golden("load/steps", repr(steps))
    golden("load/calls", repr(calls))
    golden("load/log", log_text(pack_records(caplog)))


# --- mask operations on irregular shapes -----------------------------------------------------------

def blobs(rng, shape, count=6):
    """A bool mask of `count` random rectangles with random pinholes: islands and holes."""
    h, w = shape
    mask = np.zeros(shape, bool)
    for _ in range(count):
        y, x = rng.integers(0, h), rng.integers(0, w)
        mask[y:y + rng.integers(1, max(2, h // 2)), x:x + rng.integers(1, max(2, w // 2))] = True
    return mask & ~(rng.random(shape) < 0.03)


SHAPES = ((37, 53), (41, 29), (9, 70), (64, 3))
LOW_SHAPES = ((13, 21), (17, 11), (5, 23), (19, 4))


def test_mask_operations_on_irregular_shapes():
    for seed, (shape, low) in enumerate(zip(SHAPES, LOW_SHAPES)):
        rng = np.random.default_rng(100 + seed)
        mask = blobs(rng, shape)
        values = {
            "drop_islands": [digest(sam3.drop_islands(mask.astype(np.uint8), f)) for f in (0.0, 0.01, 0.2, 1.0)],
            "fill_holes": [digest(sam3.fill_holes(mask.astype(np.uint8), f)) for f in (0.0, 0.01, 0.1, 1.0)],
            "clean_mask": [digest(sam3.clean_mask(mask, config(max_hole_fraction=h, min_island_fraction=i)))
                           for h, i in ((0.01, 0.01), (0.0, 0.0), (0.2, 0.3), (1.0, 1.0))],
        }
        logits = torch.from_numpy(rng.normal(0, 4, (3, *low))).float()
        other = torch.from_numpy(rng.normal(0, 4, (4, *low))).float()
        values["to_frame_size"] = [digest(sam3.to_frame_size(logits[:1, None], *shape, t)) for t in (0.0, -1.0, 0.5)]
        values["clean_logits"] = [digest(sam3.clean_logits(logits, a)) for a in (0, 4, 16)]
        values["low_res_logits"] = [digest(sam3.low_res_logits(logits[:1, None])), digest(sam3.low_res_logits(logits[:1]))]
        values["iou"] = digest(sam3.iou(logits, other))
        binary = other > 0
        values["non_overlapping"] = [digest(sam3.non_overlapping(binary, probs)) for probs in
                                     (torch.tensor([0.9, 0.2, 0.5, 0.7]), torch.tensor([0.5, 0.5, 0.5, 0.5]))]
        values["suppress_shrunk"] = [digest(sam3.suppress_shrunk(other, keep)) for keep in (0.0, 0.3, 0.9)]
        last = torch.from_numpy(rng.integers(-1, 6, 4))
        values["suppress_recently_occluded"] = [digest(sam3.suppress_recently_occluded(other, last, t))
                                                for t in (0.0, 0.1, 0.7)]
        for name, value in values.items():
            golden(f"mask_ops/seed{seed}/{name}", value)


# --- prompt geometry on seeded random inputs -----------------------------------------------------

def test_prompt_geometry_on_seeded_random_inputs(caplog):
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="BCVideoNodes")
    Wg, Hg = 53, 37
    for seed in range(6):
        rng = np.random.default_rng(200 + seed)
        kps = np.column_stack([rng.uniform(-0.1, 1.1, 20), rng.uniform(-0.1, 1.1, 20), rng.uniform(0, 1, 20)])
        if seed % 3 == 1:
            kps[[sam3.R_HIP, sam3.L_HIP], 2] = 0.0                       # the hips out: the step down
            kps[[sam3.R_SHOULDER, sam3.L_SHOULDER], 2] = 0.9
        if seed % 3 == 2:
            kps[[sam3.R_SHOULDER, sam3.L_SHOULDER], :] = [[0.4, 0.3, 0.9], [0.4, 0.35, 0.8]]   # no width: 0.15
            kps[[sam3.R_HIP, sam3.L_HIP], 2] = 0.0
        mask, previous, annexed = blobs(rng, (Hg, Wg)), blobs(rng, (Hg, Wg)), blobs(rng, (Hg, Wg), count=3)
        bbox = np.array([rng.uniform(-10, 20), rng.uniform(-10, 15), rng.uniform(25, 60), rng.uniform(20, 45), 0.9])
        metas = [{"keypoints_body": kps}]
        hand_positive = [(float(rng.uniform(0, Wg)), float(rng.uniform(0, Hg))) for _ in range(2)]
        hand_negative = [tuple(float(v) for v in (kps[0][0] * Wg + 0.3, kps[0][1] * Hg))]
        caplog.clear()
        values = {
            "body_points": repr([sam3.body_points(kps, t, Wg / Hg) for t in (0.0, 0.3, 0.6)]),
            "confident_pixels": repr([sam3.confident_pixels(kps, Wg, Hg, t) for t in (0.0, 0.5)]),
            "keypoint_recall": repr([sam3.keypoint_recall(mask, kps, Wg, Hg, t) for t in (0.0, 0.5, 1.1)]),
            "confident_count": repr([sam3.confident_count(metas[0], t) for t in (0.0, 0.5)]),
            "box_bounds": repr([sam3.box_bounds(bbox, Wg, Hg), sam3.box_bounds([3, 3, 10, 40, 1.0], Wg, Hg)]),
            "spread_points": repr([sam3.spread_points(mask, 3, 5, count) for count in (1, 4, 8, 9)]),
            "background_points": repr([sam3.background_points(previous, bbox, Wg, Hg, 8, m) for m in (0.0, 0.04, 0.2)]),
            "annexed_points": repr([sam3.annexed_points(annexed, bbox, Wg, Hg, n) for n in (0, 3, 8)]),
            "remember_annexed": repr([None if a is None else digest(a) for a in
                                      (sam3.remember_annexed(None, mask, previous, kps, Wg, Hg, 0.03, 0.3),
                                       sam3.remember_annexed(annexed, mask, previous, kps, Wg, Hg, 0.03, 0.3),
                                       sam3.remember_annexed(annexed, mask, previous, kps, Wg, Hg, 0.5, 1.1))]),
            "is_anchor": repr([sam3.is_anchor(0, [bbox], metas, config(**c), t, r)
                               for c in ({}, dict(min_anchor_keypoints=3, min_anchor_conf=0.2))
                               for t in (0.1, 0.5) for r in (0, 12)]
                              + [sam3.is_anchor(0, [None], metas, config(), 0.1),
                                 sam3.is_anchor(0, [bbox * [1, 1, 1, 1, -1]], metas, config(), 0.1)]),
            "clear_of_hand_points": repr(sam3._clear_of_hand_points(
                [(x * Wg, y * Hg) for x, y in sam3.body_points(kps, 0.0, Wg / Hg)],
                sam3.background_points(previous, bbox, Wg, Hg, 8, 0.0), hand_positive, hand_negative, bbox)),
        }
        prompts = []
        for dtype, cfg, prev, ann, extra in ((torch.float32, config(), None, None, ()),
                                              (torch.float32, config(), previous, annexed, ()),
                                              (torch.float16, config(negative_points=3, annexed_points=2), previous,
                                               annexed, (hand_positive, hand_negative)),
                                              (torch.float32, config(), previous, annexed, ((), hand_negative))):
            for bboxes in ([bbox], [None], [bbox * [1, 1, 1, 1, -1]]):
                box_inputs, point_inputs = sam3.prompt_for(0, bboxes, metas, Wg, Hg, torch.device("cpu"), dtype, cfg, 0.3,
                                                           prev, ann, *extra)
                prompts.append((None if box_inputs is None else digest(box_inputs),
                                None if point_inputs is None else {k: digest(v) for k, v in point_inputs.items()}))
        values["prompt_for"] = repr(prompts)
        values["log"] = log_text(pack_records(caplog))
        for name, value in values.items():
            golden(f"geometry/seed{seed}/{name}", value if name == "log" else digest(value))
