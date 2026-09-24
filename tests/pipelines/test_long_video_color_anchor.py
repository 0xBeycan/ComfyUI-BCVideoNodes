"""The color_anchor_strength widget of the three long-video samplers: every chunk after the first
gets the Lab transform that maps its regenerated overlap frames onto the frames it was seeded with,
before it is trimmed and carried, so the chain stays on the first chunk's colours; in replacement
mode only on the character. At 0 the loop is the loop without the widget.

ComfyUI itself is stubbed (comfy.*, nodes); torch is real. CastVAE stands in for chained chunks
that drift in colour: every frame of chunk k decodes as one seeded pattern with the k-th colour
cast. Skipped when torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import ANIMATE1, ANIMATE2, SCAIL2, FakeVAE, driving_videos, node_module, reference_mask, run  # noqa: E402,F401
from test_long_video_scail2 import driving_mask  # noqa: E402

NODES = [ANIMATE1, ANIMATE2, SCAIL2]
# frames each chunk adds to run()'s 200-frame output at 81 frames per chunk: 81, then 81 - overlap
CHUNKS = {ANIMATE1: [81, 76, 43], ANIMATE2: [81, 80, 39], SCAIL2: [81, 76, 43]}


class CastVAE(FakeVAE):
    """Decodes every frame of the k-th chunk as one seeded pattern (of run()'s 64 x 32 size unless
    given) under the k-th colour cast (a per-channel gain and an offset); no output clamp."""

    CASTS = [((1.0, 1.0, 1.0), 0.0), ((0.85, 1.0, 1.15), 0.04), ((1.15, 0.9, 0.95), -0.03), ((0.9, 1.1, 1.0), 0.02)]

    def __init__(self, height=64, width=32):
        self.decoded = 0
        self.pattern = 0.25 + 0.5 * torch.rand(height, width, 3, generator=torch.Generator().manual_seed(0))

    def cast(self, k):
        gain, offset = self.CASTS[k]
        return self.pattern * torch.tensor(gain) + offset

    def decode(self, latent):
        frames = (latent.shape[2] - 1) * 4 + 1
        image = self.cast(self.decoded)
        self.decoded += 1
        return image.expand(1, frames, *image.shape).clone()

    process_output = staticmethod(lambda image: image)


def chunk_frames(node):
    """The first output frame of each chunk."""
    return [sum(CHUNKS[node][:k]) for k in range(len(CHUNKS[node]))]


def lab_statistics(module, frame, region=None):
    """A frame's per-channel Lab mean and std, written out: every pixel, or the pixels where region is 1."""
    pixels = module.srgb_to_lab(frame).reshape(-1, 3)
    if region is not None:
        pixels = pixels[region.reshape(-1) > 0.5]
    mean = pixels.mean(dim=0)
    return torch.cat((mean, ((pixels - mean) ** 2).mean(dim=0).sqrt())).tolist()


def character():
    """A [64, 32] character region in the middle of run()'s frame: the driving_mask person."""
    region = torch.zeros(64, 32)
    region[16:48, 8:24] = 1.0
    return region


# --- off: the loop without the widget ----------------------------------------------------------

@pytest.mark.parametrize("node", NODES)
def test_strength_0_is_the_loop_without_the_widget(node_module, monkeypatch, caplog, node):
    caplog.set_level("INFO")
    without, _, plan = run(node_module, pose_frames=200, node=node, vae=CastVAE())

    def never(*args):
        raise AssertionError("color anchor code path entered at strength 0")

    monkeypatch.setattr(node_module, "lab_transfer", never)
    images, _, plan_off = run(node_module, pose_frames=200, node=node, vae=CastVAE(), color_anchor_strength=0.0)
    assert torch.equal(images, without) and plan_off == plan
    assert "color anchor" not in caplog.text


# --- on: the chain is anchored to the first chunk ----------------------------------------------

@pytest.mark.parametrize("node", NODES)
def test_every_chunk_keeps_the_first_chunk_s_colours(node_module, caplog, node):
    caplog.set_level("INFO")
    drifting, _, _ = run(node_module, pose_frames=200, node=node, vae=CastVAE())
    images, count, _ = run(node_module, pose_frames=200, node=node, vae=CastVAE(), color_anchor_strength=1.0)
    assert count == 200
    first = chunk_frames(node)
    assert torch.equal(images[:first[1]], drifting[:first[1]])  # the first chunk is the reference, untouched
    reference = lab_statistics(node_module, images[0])
    for start in first[1:]:
        assert lab_statistics(node_module, drifting[start]) != pytest.approx(reference, abs=1.0)  # the drift
        # the third chunk too, so the carried frames were the corrected ones
        assert lab_statistics(node_module, images[start]) == pytest.approx(reference, abs=0.02)
    lines = [line for line in caplog.text.splitlines() if "color anchor" in line]
    assert len(lines) == len(first) - 1
    assert all("color anchor 1.00 on the whole frame: mean dL " in line and "std ratio L " in line for line in lines)


def test_strength_blends_the_transform_in(node_module):
    drifting, _, _ = run(node_module, pose_frames=160, vae=CastVAE())
    full, _, _ = run(node_module, pose_frames=160, vae=CastVAE(), color_anchor_strength=1.0)
    half, _, _ = run(node_module, pose_frames=160, vae=CastVAE(), color_anchor_strength=0.5)
    # the second chunk is carried from the uncorrected first one under both strengths
    assert torch.allclose(half[81:], drifting[81:] + 0.5 * (full[81:] - drifting[81:]), atol=1e-5)


# --- the region: the character only in replacement mode ----------------------------------------

def test_animate_replacement_mode_anchors_the_character_only(node_module, caplog):
    caplog.set_level("INFO")
    region = character()
    videos = dict(background_video=driving_videos(200)["background_video"], character_mask=region.expand(200, -1, -1).clone())
    drifting, _, _ = run(node_module, pose_frames=200, node=ANIMATE1, vae=CastVAE(), **videos)
    images, _, _ = run(node_module, pose_frames=200, node=ANIMATE1, vae=CastVAE(), color_anchor_strength=1.0, **videos)
    reference = lab_statistics(node_module, images[0], region)
    for start in chunk_frames(ANIMATE1)[1:]:
        # the background is the source's again every chunk: left as decoded
        assert torch.equal(images[start][region == 0], drifting[start][region == 0])
        assert lab_statistics(node_module, images[start], region) == pytest.approx(reference, abs=0.02)
    assert "color anchor 1.00 on the character region" in caplog.text


def test_the_soft_edge_stays_inside_the_character(node_module):
    # a 96 x 192 frame, over 50 pixels on the shorter side, so the region gets a soft edge
    region = torch.zeros(192, 96)
    region[48:144, 24:72] = 1.0
    videos = dict(width=96, height=192, background_video=torch.zeros(200, 192, 96, 3), character_mask=region.expand(200, -1, -1).clone())
    drifting, _, _ = run(node_module, pose_frames=200, node=ANIMATE1, vae=CastVAE(192, 96), **videos)
    images, _, _ = run(node_module, pose_frames=200, node=ANIMATE1, vae=CastVAE(192, 96), color_anchor_strength=1.0, **videos)
    for start in chunk_frames(ANIMATE1)[1:]:
        # no halo: the background around the character, which core re-feeds, is left as decoded
        assert torch.equal(images[start][region == 0], drifting[start][region == 0])
        changed = (images[start] != drifting[start]).any(dim=-1)
        assert changed[60:132, 36:60].all() and changed[48, 24:72].any()  # the character, up to its edge


def replacement_inputs(node, spans):
    """Replacement-mode inputs over 200 driving frames whose person (the character() box) is on the
    driving frames in ``spans`` only."""
    person = torch.zeros(200, dtype=torch.bool)
    for first, end in spans:
        person[first:end] = True
    if node == ANIMATE1:
        character_mask = torch.zeros(200, 64, 32)
        character_mask[person] = character()
        return dict(background_video=driving_videos(200)["background_video"], character_mask=character_mask)
    pose_video_mask = torch.ones(200, 64, 32, 3)
    pose_video_mask[person] = driving_mask(True, 1)[0]
    return dict(replacement_mode=True, pose_video_mask=pose_video_mask, reference_image_mask=reference_mask(True))


@pytest.mark.parametrize("node", [ANIMATE1, SCAIL2])
@pytest.mark.parametrize("spans, anchored", [(((76, 81), (152, 157)), True), (((81, 86), (157, 162)), False)])
def test_the_region_is_read_at_the_driving_frames_of_the_overlap(node_module, caplog, node, spans, anchored):
    # core seeks each chained chunk at its offset less the 5 seed frames (81 - 5, 157 - 5), so the
    # overlap frames of chunks 2 and 3 show driving frames 76-80 and 152-156: a person on only those
    # frames is measured, a person on only the 5 frames after them is not there to measure
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, node=node, vae=CastVAE(), color_anchor_strength=1.0, **replacement_inputs(node, spans))
    assert caplog.text.count("color anchor 1.00 on the character region") == (2 if anchored else 0)
    assert caplog.text.count("color anchor skipped: the character region is empty") == (0 if anchored else 2)


def test_animate_without_a_background_anchors_the_whole_frame(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, node=ANIMATE1, vae=CastVAE(), color_anchor_strength=1.0,
        character_mask=character().expand(200, -1, -1).clone())
    assert "on the whole frame" in caplog.text and "character region" not in caplog.text


def test_scail2_replacement_mode_anchors_the_character_only(node_module, caplog):
    caplog.set_level("INFO")
    inputs = dict(node=SCAIL2, replacement_mode=True, pose_video_mask=driving_mask(True, 200),
                  reference_image_mask=reference_mask(True))
    region = character()
    drifting, _, _ = run(node_module, pose_frames=200, vae=CastVAE(), **inputs)
    images, _, _ = run(node_module, pose_frames=200, vae=CastVAE(), color_anchor_strength=1.0, **inputs)
    reference = lab_statistics(node_module, images[0], region)
    for start in chunk_frames(SCAIL2)[1:]:
        assert torch.equal(images[start][region == 0], drifting[start][region == 0])
        assert lab_statistics(node_module, images[start], region) == pytest.approx(reference, abs=0.02)
    assert "color anchor 1.00 on the character region" in caplog.text


def test_scail2_animation_mode_anchors_the_whole_frame(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, node=SCAIL2, vae=CastVAE(), color_anchor_strength=1.0,
        pose_video_mask=driving_mask(False, 200))
    assert "on the whole frame" in caplog.text and "character region" not in caplog.text


def test_an_empty_character_region_skips_the_chunk(node_module, caplog):
    caplog.set_level("INFO")
    white = torch.ones(200, 64, 32, 3)  # replacement mode, no person on any frame
    drifting, _, _ = run(node_module, pose_frames=200, node=SCAIL2, vae=CastVAE(), replacement_mode=True,
                         pose_video_mask=white, reference_image_mask=reference_mask(True))
    images, _, _ = run(node_module, pose_frames=200, node=SCAIL2, vae=CastVAE(), replacement_mode=True, color_anchor_strength=1.0,
                       pose_video_mask=white, reference_image_mask=reference_mask(True))
    assert torch.equal(images, drifting)
    assert caplog.text.count("color anchor skipped: the character region is empty in the 5 overlap frames") == 2


# --- the regions the adapters name -------------------------------------------------------------

def test_the_animate_region_is_the_character_mask_window(node_module):
    mask = torch.arange(10, dtype=torch.float32).view(-1, 1, 1).expand(-1, 8, 4) / 10  # frame i is i / 10
    inputs = dict(character_mask=mask, background_video=torch.zeros(10, 8, 4, 3))
    region = node_module.WanAnimateAdapter("node", "fit").anchor_region(3, 9, 16, 8, inputs)
    assert region.shape == (9, 16, 8)
    # driving frames 3 .. 9, then past the mask's end the whole frame (core leaves those rows unknown)
    assert region[:, 0, 0].tolist() == pytest.approx([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0])
    assert node_module.WanAnimateAdapter("node", "fit").anchor_region(10, 9, 16, 8, inputs) is None  # the mask ended
    assert node_module.WanAnimateAdapter("node", "fit").anchor_region(3, 9, 16, 8, dict(character_mask=mask)) is None


def test_the_scail2_region_is_the_driving_mask_s_character(node_module):
    mask = driving_mask(True, 10)
    mask[6:] = 1.0  # the person leaves after frame 5
    region = node_module.SCAIL2Adapter("node", "full").anchor_region(4, 4, 64, 32, dict(replacement_mode=True, pose_video_mask=mask))
    assert torch.equal(region[0], character()) and torch.equal(region[1], character())
    assert region[2:].sum() == 0
    assert node_module.SCAIL2Adapter("node", "full").anchor_region(4, 4, 64, 32, dict(replacement_mode=False, pose_video_mask=mask)) is None


def test_wan_animate2_has_no_replacement_mode(node_module):
    assert node_module.AnimateAdapter("node", "fit").anchor_region(0, 81, 64, 32, {}) is None


# --- after_chunk -------------------------------------------------------------------------------

@pytest.mark.parametrize("node", NODES)
def test_after_chunk_is_called_once_per_chunk(node_module, monkeypatch, node):
    seen = []
    adapter = node_module.registry.get("animate", getattr(node_module, node).ANIMATE_NODE).implementation
    monkeypatch.setattr(adapter, "after_chunk", lambda self, index: seen.append(index))
    run(node_module, pose_frames=200, node=node)
    assert seen == [0, 1, 2]
