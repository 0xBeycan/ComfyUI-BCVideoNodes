"""The tail_padding widget of the three long-video samplers: how the loop extends the adapter's
HELD_VIDEOS past their last frame when a chunk needs frames beyond them. last_frame repeats the
last frame, ping_pong plays the video backwards from its end (libs/video.ping_pong, checked
against the official padding in tests/libs/test_video.py). Past total_frames the padded frames
are cut; the Animate character mask is never extended. The videos are never extended as a whole:
a chunk inside all of them gets the videos themselves (in Animate's replacement mode every chunk
gets its window: test_long_video_replacement.py), a chunk that reads past the end of one gets the
window it reads of every video the core node seeks, with its offset moved into the window.

ComfyUI itself is stubbed (comfy.*, nodes); torch is real. Every core call is wrapped so the
driving videos and the anchor it is handed are kept; with run()'s frame-index videos a video's
frame value is its source frame index (read as the frame's largest value: the Animate character
mask holds it on the character's side of the frame only, where the background is painted black).
What a core call reads is what core's seek takes from what it was handed: `length` frames from the
offset it moved back (the fakes' `offset_in`). Skipped when torch is not installed.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (ANIMATE1, ANIMATE2, EXTENDED, SCAIL2, Calls, IndexVAE, aligned, animate_aligned,  # noqa: E402,F401
                           driving_videos, node_module, run)

NODES = [ANIMATE1, ANIMATE2, SCAIL2]
OPTIONS = ["last_frame", "ping_pong"]
POLICIES = ["fit", "full", "min29"]
CORE_NODE = {ANIMATE1: "WanAnimateToVideo", ANIMATE2: "WanAnimate2ToVideo", SCAIL2: "WanSCAILToVideo"}
# the adapters' HELD_VIDEOS, as the core call names them (SCAIL-2 renames nothing it holds)
HELD = {ANIMATE1: ("pose_video", "face_video", "background_video"), ANIMATE2: ("pose_video",),
        SCAIL2: ("pose_video", "pose_video_mask")}
VIDEOS = ("pose_video", "face_video", "background_video", "character_mask", "pose_video_mask")
ANCHOR = {ANIMATE1: "continue_motion", ANIMATE2: "continue_motion", SCAIL2: "previous_frames"}
OVERLAP = {ANIMATE1: 5, ANIMATE2: 1, SCAIL2: 5}  # the frames each chained chunk is moved back by, at run()'s defaults


@pytest.fixture
def received(animate_aligned, monkeypatch):
    """`animate_aligned`, with every core node keeping the driving videos and the anchor of each
    call: a list of {name: tensor}, one per chunk."""
    calls = []
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    for node in CORE_NODE.values():
        core = mappings[node]

        def execute(cls, _core=core, **kwargs):
            names = VIDEOS + ("continue_motion", "previous_frames")
            calls.append({name: kwargs[name] for name in names if kwargs.get(name) is not None})
            return _core.EXECUTE_NORMALIZED(**kwargs)

        monkeypatch.setitem(mappings, node, type(core.__name__, (core,), {"EXECUTE_NORMALIZED": classmethod(execute)}))
    return animate_aligned, calls


def frames(video):
    """Each frame's value: its largest."""
    return [int(v) for v in video.flatten(1).amax(1).tolist()]


def inputs(node, pose_frames):
    """The driving videos besides the pose, all `pose_frames` long and frame-index valued; the
    Animate character mask on the right half of the frame only."""
    if node == ANIMATE1:
        mask = torch.zeros(pose_frames, 64, 32)
        mask[:, :, 16:] = torch.arange(pose_frames, dtype=torch.float32).view(-1, 1, 1)
        return dict(driving_videos(pose_frames), character_mask=mask)
    return {}


def chunk_starts(node, lengths):
    """(the output frames made before each chunk, the driving frame it reads from): a chained
    chunk reads from the frames it is seeded with, OVERLAP back."""
    made, starts = 0, []
    for index, length in enumerate(lengths):
        starts.append((made, made - OVERLAP[node] if index else 0))
        made += length - (OVERLAP[node] if index else 0)
    return starts


# --- every chunk reads the driving frames of the extended videos, and is seeded from the output -

@pytest.mark.parametrize("pose_frames, total", [(240, 0), (150, 0), (161, 0), (100, 250)])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("option", OPTIONS)
@pytest.mark.parametrize("node", NODES)
def test_every_chunk_reads_the_extended_driving_frames(received, node, option, policy, pose_frames, total):
    module, calls = received
    images, count, _ = run(module, pose_frames=pose_frames, total_frames=total, node=node, vae=IndexVAE(),
                           last_chunk=policy, tail_padding=option, **inputs(node, pose_frames))
    total = total or pose_frames
    extended = EXTENDED[option](pose_frames, 1000)
    # the output: frame i shows extended driving frame i
    assert count == total and frames(images) == extended[:total]
    lengths = [c["length"] for c in Calls.animate]
    for (made, first), call, core in zip(chunk_starts(node, lengths), calls, Calls.animate):
        length, seek = core["length"], core["offset_in"]
        for name in HELD[node]:
            assert frames(call[name][seek:seek + length]) == extended[first:first + length], name
        if node == ANIMATE1:  # never extended: past its end the core node reads nothing
            assert frames(call["character_mask"][seek:seek + length]) == list(range(first, min(first + length, pose_frames)))
        # the seed: the output's last frames before the chunk, just the ones the core node keeps
        if made:
            assert frames(call[ANCHOR[node]]) == extended[max(0, made - OVERLAP[node]):made]
        else:
            assert ANCHOR[node] not in call


# --- the videos themselves while a chunk is inside them, only its window past their end --------

def test_a_chunk_inside_the_videos_gets_them_and_one_past_their_end_its_window(received):
    # animate mode (no background, no character mask)
    module, calls = received
    driving = {"face_video": driving_videos(240)["face_video"]}
    run(module, pose_frames=240, node=ANIMATE1, vae=IndexVAE(), last_chunk="full", **driving)
    # 81 + 81 + 81 + 81 at overlap 5: the chunks read from 0, 76, 152, 228; the last one past 239
    assert [c["length"] for c in Calls.animate] == [81] * 4
    for call in calls[:3]:
        assert call["face_video"] is driving["face_video"]
        assert call["pose_video"].shape[0] == 240
    last = calls[-1]
    # the frames it reads and no more: core moved back 5 frames on the chunks before
    assert Calls.animate[-1]["offset_in"] == 0
    for name in ("pose_video", "face_video"):
        assert frames(last[name]) == list(range(228, 240)) + [239] * 69, name


def test_the_first_chained_chunk_s_window_reaches_back_as_far_as_its_seed(received):
    # before core has moved an offset back, the window reaches back as far as the anchor it is
    # handed is long: the 5 frames it keeps; from then on as far as core moved back
    module, calls = received
    run(module, pose_frames=100, total_frames=250, node=SCAIL2, vae=IndexVAE())
    assert [c["length"] for c in Calls.animate] == [81] * 4
    assert [frames(call["previous_frames"]) for call in calls[1:]] == [[76, 77, 78, 79, 80], [99] * 5, [99] * 5]
    assert [call["pose_video"].shape[0] for call in calls] == [100, 81 + 5, 81, 81]
    assert [c["offset_in"] for c in Calls.animate] == [0, 0, 0, 0]
    assert [c["pose"] for c in Calls.animate] == [0.0, 76.0, 99.0, 99.0]


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
    assert sorted(last) == sorted(HELD[node] + ((ANCHOR[node], "character_mask") if node == ANIMATE1 else (ANCHOR[node],)))
    for name in HELD[node]:  # the last chunk's window: its 81 frames, from reach - 81
        assert frames(last[name]) == (list(range(pose_frames)) + padding)[reach - 81:], name


@pytest.mark.parametrize("option", OPTIONS)
def test_the_character_mask_is_never_extended(received, option):
    # replacement mode: every chunk gets its window of the mask, a cut of it
    module, calls = received
    driving = inputs(ANIMATE1, 240)
    run(module, pose_frames=240, node=ANIMATE1, vae=IndexVAE(), last_chunk="full", tail_padding=option, **driving)
    for call in calls:
        cut = frames(call["character_mask"])
        assert cut == list(range(cut[0], cut[0] + len(cut))) and cut[-1] <= 239
    assert frames(calls[-1]["character_mask"]) == list(range(228, 240))  # cut to the window, not extended
    assert calls[-1]["background_video"].shape[0] == 81


def test_a_mask_is_never_cut_to_a_single_frame(received):
    # core reads a one-frame mask as a still for the whole chunk: a window that would leave one
    # frame of a longer mask starts a frame earlier, so core still reads that frame alone
    module, calls = received
    videos = dict(background_video=driving_videos(89)["background_video"],
                  character_mask=torch.arange(89, dtype=torch.float32).view(-1, 1, 1).expand(-1, 64, 32).contiguous())
    run(module, pose_frames=89, total_frames=250, frames_per_chunk=49, node=ANIMATE1, vae=IndexVAE(), last_chunk="full",
        tail_padding="ping_pong", **videos)
    # 49 + 49 + 49 + ... at overlap 5: the third chunk reads from driving frame 88, the mask's last
    assert [c["pose"] for c in Calls.animate][:3] == [0.0, 44.0, 88.0]
    third = calls[2]
    assert frames(third["character_mask"]) == [87, 88]
    assert Calls.animate[2]["offset_in"] == 1
    assert frames(third["character_mask"][1:]) == [88]


def test_a_single_frame_mask_reaches_every_chunk_as_it_is(received):
    module, calls = received
    mask = torch.ones(1, 64, 32)
    run(module, pose_frames=150, total_frames=260, node=ANIMATE1, vae=IndexVAE(), last_chunk="full", character_mask=mask)
    assert all(call["character_mask"] is mask for call in calls)


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
