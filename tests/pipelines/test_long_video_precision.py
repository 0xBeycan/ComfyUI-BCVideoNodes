"""Half-precision inputs of the three long-video samplers (Load Video at precision fp16): core
resizes and scales what it gets in its dtype, so it gets every video it seeks one chunk's window
at a time, requantized to float32 (libs/video.requantized), and every image (the reference,
SCAIL-2's reference mask, a one-frame character mask) whole and requantized. The run is the float32
run: the core node reads the same frames, the output is the same. A video is never widened as a
whole: the largest window is the first chained chunk's, its length plus the anchor it is seeded
with, and the clips themselves stay as they are.

ComfyUI itself is stubbed (comfy.*, nodes); torch is real. Every core call is wrapped so what it is
handed is kept. The videos are frame-index valued (whole numbers, exact in float16 and through the
requantization), so a frame's value is its source frame index. Skipped when torch is not installed.
"""

import logging
import sys

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (ANIMATE1, ANIMATE2, SCAIL2, Calls, IndexVAE, aligned, animate_aligned,  # noqa: E402,F401
                           driving_videos, node_module, reference_mask, run)
from video_input_fakes import levels  # noqa: E402

NODES = [ANIMATE1, ANIMATE2, SCAIL2]
CORE_NODE = {ANIMATE1: "WanAnimateToVideo", ANIMATE2: "WanAnimate2ToVideo", SCAIL2: "WanSCAILToVideo"}
VIDEOS = ("pose_video", "face_video", "background_video", "character_mask", "pose_video_mask")
IMAGES = ("reference_image", "reference_image_mask")
FRAMES = 240  # 81 + 81 + 81 + 81 at overlap 5: four chunks (Animate 2, at overlap 1: three)
LONGEST = 81 + 81  # the first chained chunk's window: its length and the anchor before it


@pytest.fixture
def received(animate_aligned, monkeypatch):
    """`animate_aligned`, with every core node keeping the videos and the images of each call: a
    list of {name: tensor}, one per chunk."""
    calls = []
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    for node in CORE_NODE.values():
        core = mappings[node]

        def execute(cls, _core=core, **kwargs):
            calls.append({name: kwargs[name] for name in VIDEOS + IMAGES if kwargs.get(name) is not None})
            return _core.EXECUTE_NORMALIZED(**kwargs)

        monkeypatch.setitem(mappings, node, type(core.__name__, (core,), {"EXECUTE_NORMALIZED": classmethod(execute)}))
    return animate_aligned, calls


def inputs(node, frames=FRAMES, mask_frames=None):
    """The run's videos and images, the videos frame-index valued; `mask_frames` 1: Animate's
    character mask a single frame."""
    index = torch.arange(frames, dtype=torch.float32)
    values = {"pose_video": index.view(-1, 1, 1, 1).expand(-1, 64, 32, 3).contiguous(),
              "reference_image": levels(1, 64, 32, 3)}
    if node == ANIMATE1:
        mask = index[:mask_frames or frames].view(-1, 1, 1).expand(-1, 64, 32).contiguous()
        values.update(driving_videos(frames), character_mask=mask.clamp(max=1.0) if mask_frames == 1 else mask)
    if node == SCAIL2:
        values.update(pose_video_mask=values["pose_video"].clone(), reference_image_mask=reference_mask())
    return values


def halved(values):
    return {name: value.half() for name, value in values.items()}


def scalars(record):
    """A core call's record without its tensors and objects, and without what a window changes by
    design: the offset it seeked from and the length of the pose video it was handed."""
    return {key: value for key, value in record.items()
            if key not in ("offset_in", "pose_in") and (value is None or isinstance(value, (bool, int, float, str)))}


def reads(call, core):
    """What the core node reads of each video it was handed: `length` frames from the offset it moved
    back to (a single-frame mask: as it is)."""
    seek, length = core["offset_in"], core["length"]
    return {name: video if video.shape[0] == 1 else video[seek:seek + length]
            for name, video in call.items() if name in VIDEOS}


@pytest.mark.parametrize("node", NODES)
def test_the_core_node_reads_the_float32_runs_frames_from_float32_windows(received, node):
    module, calls = received
    exact = inputs(node)
    images, count, plan = run(module, pose_frames=FRAMES, node=node, vae=IndexVAE(), **exact)
    expected, expected_calls, first = list(calls), list(Calls.animate), len(calls)
    Calls.animate.clear()
    half = halved(exact)
    half_images, half_count, half_plan = run(module, pose_frames=FRAMES, node=node, vae=IndexVAE(), **half)
    got = calls[first:]
    assert (half_count, half_plan) == (count, plan) and torch.equal(half_images, images)
    assert [scalars(r) for r in Calls.animate] == [scalars(r) for r in expected_calls] and len(got) == len(expected) > 2
    for call, core, reference, reference_core in zip(got, Calls.animate, expected, expected_calls):
        assert all(value.dtype == torch.float32 for value in call.values())
        assert all(video.shape[0] <= LONGEST for name, video in call.items() if name in VIDEOS)
        for name, frames in reads(call, core).items():
            assert torch.equal(frames, reads(reference, reference_core)[name]), name
        for name in IMAGES:
            if name in call:
                assert torch.equal(call[name], reference[name]), name
    assert all(torch.equal(half[name], exact[name].half()) and half[name].dtype == torch.float16 for name in half)


@pytest.mark.parametrize("node", NODES)
def test_no_video_is_widened_as_a_whole(received, node, monkeypatch):
    module, calls = received
    widened, real = [], module.requantized

    def recording(frames):
        widened.append(tuple(frames.shape))
        return real(frames)

    monkeypatch.setattr(module, "requantized", recording)
    run(module, pose_frames=FRAMES, node=node, vae=IndexVAE(), **halved(inputs(node)))
    assert widened and max(shape[0] for shape in widened) <= LONGEST < FRAMES
    assert [shape[0] for shape in widened].count(1) >= 1  # the reference image, whole


def test_a_one_frame_character_mask_reaches_every_chunk_whole_and_float32(received):
    module, calls = received
    run(module, pose_frames=FRAMES, node=ANIMATE1, vae=IndexVAE(), **halved(inputs(ANIMATE1, mask_frames=1)))
    masks = [call["character_mask"] for call in calls]
    assert len(masks) == 4 and all(m.dtype == torch.float32 and m.shape[0] == 1 for m in masks)
    assert all(m is masks[0] for m in masks)  # requantized once, before the loop


@pytest.mark.parametrize("node", NODES)
def test_the_log_names_the_half_videos(node_module, caplog, node):
    caplog.set_level(logging.INFO)
    run(node_module, pose_frames=FRAMES, node=node, **halved(inputs(node)))
    half = {ANIMATE1: "pose_video, face_video, background_video, character_mask", ANIMATE2: "pose_video",
            SCAIL2: "pose_video, pose_video_mask"}[node]
    assert f"{half} half precision: every chunk gets its window requantized to float32." in caplog.text


def test_a_float32_run_hands_the_core_node_the_videos_themselves(received):
    module, calls = received
    exact = inputs(ANIMATE1)
    run(module, pose_frames=FRAMES, node=ANIMATE1, vae=IndexVAE(), last_chunk="full", **exact)
    for call in calls[:3]:  # the chunks inside the videos (the last one reads past them: its window)
        assert all(call[name] is exact[name] for name in ("pose_video", "face_video", "background_video", "character_mask"))


def test_the_colored_masks_mode_is_read_from_the_requantized_first_frame(node_module):
    for replacement_mode in (False, True):
        mask = reference_mask(replacement_mode)
        assert node_module.mask_convention(mask.half()) == node_module.mask_convention(mask)
