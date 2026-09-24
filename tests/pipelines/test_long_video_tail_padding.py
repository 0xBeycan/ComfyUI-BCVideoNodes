"""The tail_padding widget of the three long-video samplers: how the loop extends the adapter's
HELD_VIDEOS past their last frame when a chunk needs frames beyond them. last_frame repeats the
last frame, ping_pong plays the video backwards from its end (libs/video.ping_pong, checked
against the official padding in tests/libs/test_video.py). Past total_frames the padded frames
are cut; the Animate character mask is never extended.

ComfyUI itself is stubbed (comfy.*, nodes); torch is real. Every core call is wrapped so the
driving videos it is handed are kept; with run()'s frame-index videos a video's frame value is
its source frame index. Skipped when torch is not installed.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (ANIMATE1, ANIMATE2, SCAIL2, Calls, IndexVAE, aligned, animate_aligned,  # noqa: E402,F401
                           driving_videos, node_module, run)

NODES = [ANIMATE1, ANIMATE2, SCAIL2]
OPTIONS = ["last_frame", "ping_pong"]
CORE_NODE = {ANIMATE1: "WanAnimateToVideo", ANIMATE2: "WanAnimate2ToVideo", SCAIL2: "WanSCAILToVideo"}
# the adapters' HELD_VIDEOS, as the core call names them (SCAIL-2 renames nothing it holds)
HELD = {ANIMATE1: ("pose_video", "face_video", "background_video"), ANIMATE2: ("pose_video",),
        SCAIL2: ("pose_video", "pose_video_mask")}
VIDEOS = ("pose_video", "face_video", "background_video", "character_mask", "pose_video_mask")


@pytest.fixture
def received(animate_aligned, monkeypatch):
    """`animate_aligned`, with every core node keeping the driving videos of each call: a list
    of {name: tensor}, one per chunk."""
    calls = []
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    for node in CORE_NODE.values():
        core = mappings[node]

        def execute(cls, _core=core, **kwargs):
            calls.append({name: kwargs[name] for name in VIDEOS if kwargs.get(name) is not None})
            return _core.EXECUTE_NORMALIZED(**kwargs)

        monkeypatch.setitem(mappings, node, type(core.__name__, (core,), {"EXECUTE_NORMALIZED": classmethod(execute)}))
    return animate_aligned, calls


def frames(video):
    return [int(v) for v in video.reshape(video.shape[0], -1)[:, 0].tolist()]


def inputs(node, pose_frames):
    """The driving videos besides the pose, all `pose_frames` long and frame-index valued."""
    if node == ANIMATE1:
        mask = torch.arange(pose_frames, dtype=torch.float32).view(-1, 1, 1).expand(-1, 64, 32).contiguous()
        return dict(driving_videos(pose_frames), character_mask=mask)
    return {}


# --- output frame i is driving frame i; the padding reaches the output only past the input -----

@pytest.mark.parametrize("pose_frames, total, last_chunk", [
    (240, 0, "fit"),    # the fit snap-up: 241 produced
    (240, 0, "full"),   # last_chunk full: 309 produced
    (100, 250, None),   # total_frames longer than the input, with the node's default last_chunk
])
@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("node", NODES)
def test_output_frame_i_is_driving_frame_i(received, node, option, pose_frames, total, last_chunk):
    module, calls = received
    policy = {} if last_chunk is None else dict(last_chunk=last_chunk)
    images, count, _ = run(module, pose_frames=pose_frames, total_frames=total, node=node, vae=IndexVAE(),
                           tail_padding=option, **policy, **inputs(node, pose_frames))
    total = total or pose_frames
    assert count == total and images.shape[0] == total
    pose = calls[-1]["pose_video"]  # the pose as extended for the last chunk
    assert frames(pose)[:pose_frames] == list(range(pose_frames))
    assert frames(images) == frames(pose)[:total]


# --- the padded chunk gets the chosen padding on every HELD_VIDEOS input -----------------------

# pose frames and the last frame the full plan samples: 81 + 81 + 81 + 81 at overlap 5 (1 for Animate 2)
FULL_PLAN = {ANIMATE1: (240, 309), ANIMATE2: (250, 321), SCAIL2: (240, 309)}


@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("node", NODES)
def test_the_padded_chunk_gets_the_padding_on_every_held_video(received, node, option):
    module, calls = received
    pose_frames, reach = FULL_PLAN[node]
    run(module, pose_frames=pose_frames, node=node, vae=IndexVAE(), last_chunk="full", tail_padding=option,
        **inputs(node, pose_frames))
    assert [c["length"] for c in Calls.animate] == [81] * 4
    extra = reach - pose_frames
    padding = {"last_frame": [pose_frames - 1] * extra,  # the last frame, repeated
               "ping_pong": list(range(pose_frames - 2, pose_frames - 2 - extra, -1))}[option]  # backwards from the end
    last = calls[-1]
    assert sorted(last) == sorted(HELD[node] + (("character_mask",) if node == ANIMATE1 else ()))
    for name in HELD[node]:
        assert frames(last[name]) == list(range(pose_frames)) + padding, name


@pytest.mark.parametrize("option", OPTIONS)
def test_the_character_mask_is_never_extended(received, option):
    module, calls = received
    driving = inputs(ANIMATE1, 240)
    run(module, pose_frames=240, node=ANIMATE1, vae=IndexVAE(), last_chunk="full", tail_padding=option, **driving)
    assert all(call["character_mask"] is driving["character_mask"] for call in calls)
    assert calls[-1]["background_video"].shape[0] == 309


@pytest.mark.parametrize("option, words", [("last_frame", "last frame held to 309 frames"),
                                           ("ping_pong", "ping_pong padded to 309 frames")])
def test_the_log_line_names_the_padding(node_module, caplog, option, words):
    caplog.set_level("INFO")
    run(node_module, pose_frames=240, node=SCAIL2, tail_padding=option)
    lines = [line for line in caplog.text.splitlines() if " to 309 frames: " in line]
    assert len(lines) == 1 and words + ": pose_video +69, pose_video_mask +69 (" in lines[0]


# --- an unknown value --------------------------------------------------------------------------

@pytest.mark.parametrize("node", NODES)
def test_an_unknown_tail_padding_is_an_error(node_module, node):
    with pytest.raises(ValueError, match=r"tail_padding must be one of last_frame, ping_pong; found 'mirror'"):
        run(node_module, pose_frames=240, node=node, tail_padding="mirror")
    assert Calls.animate == []
