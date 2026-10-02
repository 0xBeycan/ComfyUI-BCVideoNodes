"""Replacement mode of the Wan Animate sampler (background_video and character_mask connected):
the sampler paints the background black wherever the character mask is above 0, one chunk's
window at a time, so Load Video's frames can be wired as background_video directly. On every chunk
the core node reads what today's chain gives it: WanAnimate Preprocess's bg_images (the frames
painted whole, written out below as `painted`) held past its end as tail_padding says. A
background painted already (bg_images) reads the same; float16 frames (Load Video at precision
fp16) read as the float32 run's. Animate mode is untouched: no window, nothing painted.

ComfyUI itself is stubbed (comfy.*, nodes); torch is real. Every core call is wrapped so the
videos it is handed are kept. The pose is run()'s frame-index video, so what core reads of it
names the source frame behind every frame it reads of the others, the held ones included.
Skipped when torch is not installed.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import ANIMATE1, Calls, node_module, run  # noqa: E402,F401
from video_input_fakes import levels  # noqa: E402

FRAMES = 240
VIDEOS = ("pose_video", "face_video", "background_video", "character_mask")
LONGEST = 81 + 5  # the first chained chunk's window: its length and the 5 seed frames the core node keeps


@pytest.fixture
def received(node_module, monkeypatch):
    """`node_module`, with WanAnimateToVideo keeping the videos of each call: a list of
    {name: tensor}, one per chunk."""
    calls = []
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    core = mappings["WanAnimateToVideo"]

    def execute(cls, **kwargs):
        calls.append({name: kwargs[name] for name in VIDEOS if kwargs.get(name) is not None})
        return core.EXECUTE_NORMALIZED(**kwargs)

    monkeypatch.setitem(mappings, "WanAnimateToVideo", type(core.__name__, (core,), {"EXECUTE_NORMALIZED": classmethod(execute)}))
    return node_module, calls


def scene(frames=FRAMES):
    """(frames, character mask): random 8-bit levels, as Load Video loads them, and a mask whose
    character moves across the frame: 1 inside, a soft column at its right edge (the level 128 /
    255, above 0: painted too), 0 elsewhere."""
    images = levels(frames, 64, 32, 3)
    mask = torch.zeros(frames, 64, 32)
    for f in range(frames):
        x = f % 20
        mask[f, 8:56, x:x + 10] = 1.0
        mask[f, 8:56, x + 10] = 128 / 255
    return images, mask


def painted(images, mask):
    """Today's chain, WanAnimate Preprocess's bg_images: every pixel the mask holds (above 0) black."""
    return torch.where(mask.unsqueeze(-1) > 0, torch.zeros((), dtype=images.dtype), images)


def reads(call, core, name):
    """What the core node reads of a video it was handed: `length` frames from the offset it moved
    back to."""
    return call[name][core["offset_in"]:core["offset_in"] + core["length"]]


def sources(call, core):
    """The source frame behind every frame the core node reads: the frame-index pose it reads."""
    return reads(call, core, "pose_video")[:, 0, 0, 0].long()


# pose frames, total_frames, last_chunk: inside the videos; the fit snap-up 3 frames past their
# end; every chunk full length, the last 69 frames past it; total_frames past it
RUNS = [(161, 0, "fit"), (150, 0, "fit"), (240, 0, "full"), (100, 250, "fit")]


@pytest.mark.parametrize("pose_frames, total, policy", RUNS)
@pytest.mark.parametrize("option", ["last_frame", "ping_pong"])
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
@pytest.mark.parametrize("wired", ["frames", "bg_images"])
def test_the_core_node_reads_todays_painted_background(received, wired, precision, option, pose_frames, total, policy):
    module, calls = received
    images, mask = scene(pose_frames)
    expected, exact_mask = painted(images, mask), mask
    if precision == "fp16":
        images, mask = images.half(), mask.half()
    background = images if wired == "frames" else painted(images, mask)
    run(module, pose_frames=pose_frames, total_frames=total, node=ANIMATE1, last_chunk=policy, tail_padding=option,
        background_video=background, character_mask=mask)
    assert len(calls) == len(Calls.animate) > 1
    held = False
    for call, core in zip(calls, Calls.animate):
        source = sources(call, core)
        held |= bool((source[1:] <= source[:-1]).any())  # last_frame repeats a frame, ping_pong turns
        got = reads(call, core, "background_video")
        assert got.dtype == torch.float32 and torch.equal(got, expected[source])
        cut = reads(call, core, "character_mask")
        assert torch.equal(cut, exact_mask[source[:len(cut)]])  # the mask itself: cut at its end, never held
    assert held == ((pose_frames, total, policy) != (161, 0, "fit"))  # 161: the plan ends on the last frame
    assert torch.equal(background, images if wired == "frames" else painted(images, mask))  # the inputs untouched


def test_a_single_frame_mask_paints_every_frame_with_it(received):
    module, calls = received
    images, mask = scene()
    still = mask[7:8].clone()
    run(module, pose_frames=FRAMES, node=ANIMATE1, last_chunk="full", background_video=images, character_mask=still)
    for call, core in zip(calls, Calls.animate):
        assert call["character_mask"] is still  # core repeats a still itself
        assert torch.equal(reads(call, core, "background_video"), painted(images, still.expand(FRAMES, -1, -1))[sources(call, core)])


def test_animate_mode_is_untouched(received):
    # the background or the mask alone: nothing painted, the videos themselves while a chunk is inside them
    module, calls = received
    images, mask = scene(161)  # 81 + 81 + 9: no chunk reads past the videos
    run(module, pose_frames=161, node=ANIMATE1, background_video=images)
    run(module, pose_frames=161, node=ANIMATE1, character_mask=mask)
    alone = len(calls) // 2
    assert alone > 1 and all(call["background_video"] is images for call in calls[:alone])
    assert all(call["character_mask"] is mask for call in calls[alone:])


def test_every_chunk_gets_its_window_painted_never_the_whole_background(received, monkeypatch, caplog):
    caplog.set_level("INFO")
    module, calls = received
    painted_frames, real = [], module.painted_black

    def recording(images, mask):
        painted_frames.append(int(images.shape[0]))
        return real(images, mask)

    monkeypatch.setattr(module, "painted_black", recording)
    images, mask = scene()
    run(module, pose_frames=FRAMES, node=ANIMATE1, last_chunk="full", background_video=images, character_mask=mask)
    assert len(painted_frames) == len(calls) == 4 and max(painted_frames) <= LONGEST < FRAMES
    assert all(call["background_video"].shape[0] <= LONGEST for call in calls)
    assert "background_video is painted black where character_mask is above 0, a chunk's window at a time." in caplog.text


def test_a_half_window_is_freed_once_requantized(received, monkeypatch):
    # float16 frames: the painted window is float16 and the core node gets its float32 copy; the
    # float16 one goes at once, not kept through the chunk
    import gc
    import weakref

    module, calls = received
    painted, real = [], module.painted_black

    def recording(images, mask):
        out = real(images, mask)
        painted.append(weakref.ref(out))
        return out

    monkeypatch.setattr(module, "painted_black", recording)
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    core, alive = mappings["WanAnimateToVideo"], []

    def execute(cls, **kwargs):
        gc.collect()
        alive.append(painted[-1]() is not None)
        return core.EXECUTE_NORMALIZED(**kwargs)

    monkeypatch.setitem(mappings, "WanAnimateToVideo", type(core.__name__, (core,), {"EXECUTE_NORMALIZED": classmethod(execute)}))
    images, mask = scene()
    run(module, pose_frames=FRAMES, node=ANIMATE1, last_chunk="full", background_video=images.half(), character_mask=mask.half())
    assert len(alive) == 4 and not any(alive)


def test_a_mask_of_another_size_is_an_error(node_module):
    images, mask = scene()
    with pytest.raises(ValueError) as error:
        run(node_module, pose_frames=FRAMES, node=ANIMATE1, background_video=images, character_mask=mask[:, :32, :16])
    assert str(error.value) == ("character_mask is 16x32 but background_video is 32x64: the background is painted black "
                                "where the mask marks the character, so connect the two from the same video.")
    assert Calls.animate == []  # it stops before anything is sampled
