"""Unused heavy outputs return empty (nodes/unused_outputs.py), node by node and the handler alone:

- which outputs are heavy: a whole-clip IMAGE / MASK the node makes; the pass-throughs of the
  guards are the input tensors themselves, and a node with one output (not an output node) runs
  only when that output is linked, so it declares none;
- the on_prompt handler's stamp, its stale stamp overwritten, the feature off (stamp removed,
  console line, toast event) when another pack's handler runs after it, still on when the handler
  after it is a stamping one (ComfyUI-BCNodes'), the pack named from its module;
- the node side: no stamp -> every output full; a link the stamp missed -> full and a warning;
- per node: each unlinked heavy output is a new 0-frame tensor of the full output's dtype and
  trailing shape, every other output equals the full run's, and where the output is a step of its
  own the step does not run (pose images not drawn, faces not cut, the driving colored mask not
  rendered, the driving video not blacked out, the WanAnimate SAM track not run).

Fake models, synthetic frames; the full run is the same node call without a stamp. The runtime
behaviour through ComfyUI's executor is in test_unused_outputs_runtime.py.

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_unused_outputs.py
"""
import logging
import sys
import types

import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
folder_paths = pytest.importorskip("folder_paths")

from names import nodes  # noqa: E402
from pose_fakes import frames  # noqa: E402
from sam3_1_multiplex_fakes import sam3  # noqa: E402
from sampler_fakes import ANIMATE1, ANIMATE2, SCAIL2, node_module  # noqa: E402,F401
from test_nodes_wiring import fake_models, same  # noqa: E402,F401
from unused_outputs_fakes import NODE, graph, hidden, sampler_loop, unused  # noqa: E402

# node key -> its heavy outputs; every other node of the pack declares none
HEAVY = {
    "BCVWanAnimateLongVideoSampler": ("images",),
    "BCVWanAnimate2LongVideoSampler": ("images",),
    "BCVPoseDetection": ("pose_images",),
    "BCVFaceCrop": ("face_images",),
    "BCVWanAnimatePreprocess": ("pose_images", "face_images", "mask"),
    "BCVSCAIL2LongVideoSampler": ("images",),
    "BCVSCAIL2ColoredMask": ("pose_video_mask",),
    "BCVSCAIL2Preprocess": ("pose_video", "pose_video_mask", "mask"),
    "BCVSapiens2Pose": ("pose_images",),
    "BCVLoadVideo": ("images",),
}
WIDGETS = dict(body_stick_width=-1, hand_stick_width=-1, draw_head=True, draw_threshold=0.5)


@pytest.fixture
def input_folder(tmp_path, monkeypatch):
    (tmp_path / "input").mkdir()
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path / "input"))
    return tmp_path / "input"


def emptied(out, full):
    """`out` is the drop of `full`: a 0-frame tensor of its dtype and trailing shape, owning its
    (empty) storage."""
    return (isinstance(out, torch.Tensor) and out.shape == (0, *full.shape[1:]) and out.dtype == full.dtype
            and out._base is None)


def check_outputs(cls, out, full, unlinked):
    """Every heavy output of `unlinked` dropped, every other output what the full run returned."""
    assert len(out) == len(full) == len(cls.RETURN_NAMES)
    for name, a, b in zip(cls.RETURN_NAMES, out, full):
        if name in unlinked:
            assert emptied(a, b), name
        else:
            assert same(a, b), name


def calls(monkeypatch, name):
    """Records the calls of the `unused` seam `name` (args, kwargs) and calls through."""
    real = getattr(unused, name)
    seen = []

    def recorded(*args, **kwargs):
        seen.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(unused, name, recorded)
    return seen


# --- which outputs are heavy ---------------------------------------------------------------------

def test_the_heavy_outputs_and_their_hidden_inputs(input_folder):
    for key, cls in nodes.NODE_CLASS_MAPPINGS.items():
        heavy = getattr(cls, "HEAVY_OUTPUTS", ())
        assert heavy == HEAVY.get(key, ()), key
        if not heavy:
            continue
        assert set(heavy) <= set(cls.RETURN_NAMES), key
        assert all(cls.RETURN_TYPES[cls.RETURN_NAMES.index(name)] in ("IMAGE", "MASK") for name in heavy), key
        # a node that runs only when one of its outputs is linked needs a second output to drop one
        assert len(cls.RETURN_NAMES) > 1 or getattr(cls, "OUTPUT_NODE", False), key
        if not hasattr(cls, "ANIMATE_NODE"):  # the samplers' INPUT_TYPES read ComfyUI: below, under its stubs
            assert cls.INPUT_TYPES()["hidden"] == unused.LINK_INPUTS, key


def test_the_guards_pass_their_masks_through():
    from guard_fakes import clip

    masks, pose_data = clip()
    out = nodes.BCVWanAnimatePreprocessGuard().check(masks, pose_data, False)
    assert out[0] is masks and out[1] is pose_data
    assert nodes.BCVMaskGuard().check(masks, False)[0] is masks
    # the timelines are one plot image
    assert out[4].shape[0] == 1


# --- the on_prompt handler -------------------------------------------------------------------------

class FakeServer:
    """What the handler reads of ComfyUI's PromptServer: the handler list, and send_sync."""

    def __init__(self):
        self.on_prompt_handlers = []
        self.sent = []

    def add_on_prompt_handler(self, handler):
        self.on_prompt_handlers.append(handler)

    def send_sync(self, event, data, sid=None):
        self.sent.append((event, data, sid))

    def trigger(self, prompt, client_id="client"):
        json_data = {"prompt": prompt, "client_id": client_id}
        for handler in self.on_prompt_handlers:
            json_data = handler(json_data)
        return json_data["prompt"]


def a_prompt():
    """SCAIL-2 Preprocess "1" with pose_video_mask and reference_mask linked (reference_mask is not
    heavy), SCAIL-2 Colored Mask "2" with nothing linked, Mask Guard "3" (no heavy output)."""
    return {
        "1": {"class_type": "BCVSCAIL2Preprocess", "inputs": {"prompt": "person"}},
        "2": {"class_type": "BCVSCAIL2ColoredMask", "inputs": {"driving_mask": ["1", 3]}},
        "3": {"class_type": "BCVMaskGuard", "inputs": {"mask": ["1", 3]}},
        "4": {"class_type": "SomeSampler", "inputs": {"pose_video_mask": ["1", 1], "ref": ["1", 4], "other": ["2", 1]}},
    }


@pytest.fixture
def server():
    server = FakeServer()
    server.add_on_prompt_handler(unused.LinkStamp(nodes.NODE_CLASS_MAPPINGS, server))
    return server


def stamps(prompt):
    return {node_id: node["inputs"].get(unused.STAMP) for node_id, node in prompt.items()}


def test_the_handler_stamps_the_linked_heavy_outputs(server):
    # mask is linked to the guard and the colored mask: "mask,pose_video_mask"
    assert stamps(server.trigger(a_prompt())) == {"1": "mask,pose_video_mask", "2": "", "3": None, "4": None}
    assert server.sent == []


def test_the_handler_overwrites_a_stale_stamp(server):
    prompt = a_prompt()
    prompt["2"]["inputs"][unused.STAMP] = "pose_video_mask"
    assert stamps(server.trigger(prompt))["2"] == ""


def test_the_handler_leaves_other_payloads_alone(server):
    handler = server.on_prompt_handlers[0]
    for json_data in ({}, {"prompt": None}, {"prompt": {"1": "not a node"}}, None):
        assert handler(json_data) == json_data


def another_pack_handler(json_data):
    return json_data


def test_another_packs_handler_after_ours_turns_it_off_and_says_so(server, caplog):
    server.add_on_prompt_handler(another_pack_handler)
    prompt = a_prompt()
    prompt["2"]["inputs"][unused.STAMP] = ""  # a stamp left from an earlier run goes too
    caplog.set_level(logging.WARNING, logger="BCVideoNodes")
    assert stamps(server.trigger(prompt, client_id="abc")) == {"1": None, "2": None, "3": None, "4": None}
    message = "RAM saving of unused outputs is off for this run: test_unused_outputs changes the prompt after it."
    assert server.sent == [(unused.EVENT, {"message": message}, "abc")]
    assert message in caplog.text


def test_a_handler_before_ours_and_a_stamping_handler_after_it_keep_it_on(server):
    class BCNodesStamp:  # ComfyUI-BCNodes' handler, as marked there
        bc_link_stamp = True

        def __call__(self, json_data):
            return json_data

    server.on_prompt_handlers.insert(0, another_pack_handler)
    server.add_on_prompt_handler(BCNodesStamp())
    assert stamps(server.trigger(a_prompt()))["1"] == "mask,pose_video_mask"
    assert server.sent == []


def test_the_pack_is_named_by_its_folder(server, monkeypatch):
    # ComfyUI imports a custom node folder as a module named by its path, dots written as _x_
    name = "/comfy/custom_nodes/some_x_pack"
    module = types.ModuleType(name)
    module.__file__ = "/comfy/custom_nodes/some.pack/__init__.py"
    monkeypatch.setitem(sys.modules, name, module)
    handler = types.FunctionType(another_pack_handler.__code__, {}, "handler")
    handler.__module__ = name + ".hooks"
    server.add_on_prompt_handler(handler)
    server.trigger(a_prompt())
    assert "some.pack changes the prompt" in server.sent[0][1]["message"]
    del module.__file__
    server.trigger(a_prompt())
    assert "some.pack changes the prompt" in server.sent[1][1]["message"]


def test_registration_reads_the_server_from_sys_modules(monkeypatch, caplog):
    monkeypatch.delitem(sys.modules, "server", raising=False)
    caplog.set_level(logging.INFO, logger="BCVideoNodes")
    assert unused.register_link_stamp(nodes.NODE_CLASS_MAPPINGS) is None
    assert "no ComfyUI server" in caplog.text
    fake = FakeServer()
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=fake)))
    handler = unused.register_link_stamp(nodes.NODE_CLASS_MAPPINGS)
    assert fake.on_prompt_handlers == [handler] and handler.bc_link_stamp is True


# --- the node side ---------------------------------------------------------------------------------

def test_no_stamp_wants_everything_and_drops_nothing():
    cls = nodes.BCVSCAIL2ColoredMask
    assert unused.heavy_wanted(cls, graph(cls, stamp=False), NODE) is None
    assert unused.heavy_wanted(cls, None, None) is None
    outputs = (torch.rand(3, 4, 5, 3), torch.rand(1, 4, 5, 3))
    out = unused.drop_unlinked_heavy(cls, outputs, None, None)
    assert out[0] is outputs[0] and out[1] is outputs[1]


def test_the_stamp_says_what_is_wanted_and_a_missed_link_is_kept(caplog):
    cls = nodes.BCVSCAIL2Preprocess
    assert unused.heavy_wanted(cls, graph(cls, ["mask"]), NODE) == {"mask"}
    # the stamp lists nothing, the prompt links pose_video_mask: full, with a warning
    caplog.set_level(logging.WARNING, logger="BCVideoNodes")
    assert unused.heavy_wanted(cls, graph(cls, links=["pose_video_mask"]), NODE) == {"pose_video_mask"}
    assert "links pose_video_mask, which its link stamp does not list" in caplog.text


def test_a_dropped_output_is_a_new_empty_tensor_of_its_kind():
    cls = nodes.BCVSCAIL2Preprocess
    full = (torch.rand(3, 4, 5, 3), torch.rand(3, 4, 5, 3), torch.rand(1, 4, 5, 3), torch.rand(3, 4, 5).half(),
            torch.rand(1, 4, 5))
    out = unused.drop_unwanted(cls, full, {"pose_video_mask"})
    check_outputs(cls, out, full, {"pose_video", "mask"})
    assert out[1] is full[1] and out[2] is full[2]


# --- each node ---------------------------------------------------------------------------------------

def test_pose_detection_does_not_draw_unlinked_pose_images(fake_models, monkeypatch):
    cls, images = nodes.BCVPoseDetection, frames()
    full = cls().detect(images, **WIDGETS)
    drawn = calls(monkeypatch, "draw")
    out = cls().detect(images, **WIDGETS, **hidden(cls))
    assert drawn == []
    check_outputs(cls, out, full, {"pose_images"})
    check_outputs(cls, cls().detect(images, **WIDGETS, **hidden(cls, ["pose_images"])), full, set())
    assert len(drawn) == 1


@pytest.fixture
def fake_sapiens2(fake_models, monkeypatch):
    from sapiens2_fakes import FakeSapiens2, sapiens2

    monkeypatch.setattr(sapiens2, "load_pose_estimator", lambda name: FakeSapiens2())


def test_sapiens2_pose_does_not_draw_unlinked_pose_images(fake_sapiens2, monkeypatch):
    cls, images = nodes.BCVSapiens2Pose, frames()
    full = cls().detect(images, "0.4b bf16", **WIDGETS)
    drawn = calls(monkeypatch, "draw")
    check_outputs(cls, cls().detect(images, "0.4b bf16", **WIDGETS, **hidden(cls)), full, {"pose_images"})
    assert drawn == []


def test_face_crop_does_not_cut_unlinked_faces(fake_models, monkeypatch):
    cls, images = nodes.BCVFaceCrop, frames()
    pose_data = nodes.BCVPoseDetection().detect(images, **WIDGETS)[1]
    full = cls().crop(images, pose_data, 8)
    resized = []
    monkeypatch.setattr(cv2, "resize", lambda *args, **kwargs: resized.append(args) or cv2_resize(*args, **kwargs))
    check_outputs(cls, cls().crop(images, pose_data, 8, **hidden(cls)), full, {"face_images"})
    assert resized == []
    check_outputs(cls, cls().crop(images, pose_data, 8, **hidden(cls, ["face_images"])), full, set())
    assert len(resized) == len(images)


cv2_resize = cv2.resize


@pytest.mark.parametrize("linked", [(), ("mask",), ("pose_images", "face_images")])
def test_wan_animate_preprocess_computes_only_linked_heavy_outputs(fake_models, monkeypatch, caplog, linked):
    cls, images = nodes.BCVWanAnimatePreprocess, frames()
    widgets = dict(WIDGETS, face_padding=8, mode="box_keypoint", prompt=sam3.PROMPT)
    full = cls().process(images, **widgets)
    fake_models.calls.clear()
    drawn = calls(monkeypatch, "draw")
    caplog.set_level(logging.INFO, logger="BCVideoNodes")
    caplog.clear()
    out = cls().process(images, **widgets, **hidden(cls, linked))
    check_outputs(cls, out, full, set(cls.HEAVY_OUTPUTS) - set(linked))
    assert len(fake_models.calls) == ("mask" in linked)  # the SAM track runs only for a linked mask
    assert len(drawn) == ("pose_images" in linked)
    assert ("cropping the faces" in caplog.text) == ("face_images" in linked)


def test_scail2_colored_mask_does_not_render_an_unlinked_driving_mask(monkeypatch):
    cls = nodes.BCVSCAIL2ColoredMask
    driving, reference = (torch.rand(5, 16, 8) > 0.5).float(), (torch.rand(1, 16, 8) > 0.5).float()
    full = cls().render(driving, False, reference)
    rendered = calls(monkeypatch, "render_identity")
    check_outputs(cls, cls().render(driving, False, reference, **hidden(cls)), full, {"pose_video_mask"})
    assert [len(args[0]) for args, _ in rendered] == [0, 1]  # the driving mask's render covers no frame


@pytest.mark.parametrize("black_background", [False, True])
def test_scail2_preprocess_drops_what_is_not_linked(fake_models, monkeypatch, black_background):
    cls = nodes.BCVSCAIL2Preprocess
    images, reference = frames(), frames()[:1]
    full = cls().process(images, reference, False, "prompt", "person", black_background=black_background)
    fake_models.calls.clear()
    blacked = calls(monkeypatch, "driving_on_black")
    rendered = calls(monkeypatch, "render_identity")
    out = cls().process(images, reference, False, "prompt", "person", black_background=black_background,
                        **hidden(cls, ["pose_video_mask"]))
    check_outputs(cls, out, full, {"pose_video", "mask"})
    # the driving video is not blacked out, the masks are rendered, the SAM track runs (mask is
    # what the colored mask is cut by): mask is dropped at return
    assert blacked == [] and len(rendered) == 2 and len(fake_models.calls) == 2
    out = cls().process(images, reference, False, "prompt", "person", black_background=black_background,
                        **hidden(cls, ["pose_video"]))
    check_outputs(cls, out, full, {"pose_video_mask", "mask"})
    assert len(blacked) == black_background


def test_load_video_drops_unlinked_frames(input_folder):
    from video_input_fakes import grey_clip

    cls = nodes.NODE_CLASS_MAPPINGS["BCVLoadVideo"]
    grey_clip(input_folder / "clip.mkv", 9)
    args = ("clip.mkv", "SCAIL", "512p", "auto", "", 1, "")
    full = cls().load(*args)
    check_outputs(cls, cls().load(*args, **hidden(cls)), full, {"images"})
    check_outputs(cls, cls().load(*args, **hidden(cls, ["images"])), full, set())


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2, SCAIL2])
def test_the_samplers_drop_unlinked_frames(node_module, monkeypatch, node):
    cls = getattr(node_module, node)
    assert cls.INPUT_TYPES()["hidden"] == unused.LINK_INPUTS
    generated = (torch.rand(5, 16, 8, 3), 5, "plan")
    monkeypatch.setattr(sampler_loop, "generate", lambda *args: generated)
    widgets = [None] * 20  # model ... tail_padding: the chunk loop is a stand-in
    check_outputs(cls, cls().generate(*widgets, **hidden(cls)), generated, {"images"})
    check_outputs(cls, cls().generate(*widgets, **hidden(cls, ["images"])), generated, set())


@pytest.mark.parametrize("mode", ["box_keypoint", "prompt_pose"])
def test_scail2_preprocess_draws_no_pose_images(fake_models, monkeypatch, mode):
    # SCAIL-2 draws no pose: it reads Pose Detection's pose_data only, so the pose images are not drawn
    images, reference = frames(), frames()[:1]
    pose_data = nodes.BCVPoseDetection().detect(images, **WIDGETS)[1]  # drawn, as before
    (mask,) = nodes.BCVSAM3VideoTrack().track(images, mode, "person", 1, -1, pose_data=pose_data)
    (reference_mask,) = nodes.BCVSAM3VideoTrack().track(reference, "prompt", "person", 1, -1)
    chained = (images, *nodes.BCVSCAIL2ColoredMask().render(mask, False, reference_mask), mask, reference_mask)
    drawn = calls(monkeypatch, "draw")
    tracked, track = [], nodes.BCVSAM3VideoTrack.track
    monkeypatch.setattr(nodes.BCVSAM3VideoTrack, "track",
                        lambda self, *args, **kwargs: tracked.append(kwargs.get("pose_data")) or track(self, *args, **kwargs))
    out = nodes.BCVSCAIL2Preprocess().process(images, reference, False, mode, "person")
    assert drawn == []
    assert same(tracked[0], pose_data)  # the pose the track reads is Pose Detection's
    for name, a, b in zip(nodes.BCVSCAIL2Preprocess.RETURN_NAMES, out, chained):
        assert same(a, b), name
