"""The Wan SCAIL-2 Long Video Sampler's chunk loop against the fake WanSCAILToVideo.

ComfyUI itself is stubbed (comfy.*, nodes); torch is real. For the end-to-end alignment a VAE
that carries each pixel frame's value through its latent (IndexVAE) and a SamplerCustom that
"generates" what the pose conditioning says wherever the noise mask lets it (PoseFollowingSampler)
stand in for the model: with run()'s frame-index pose video, output frame i must then show
driving frame i. Skipped when torch is not installed.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (LATENT_DOWN, SCAIL2, Calls, FakeCLIPVisionEncode, FakeNodeOutput,  # noqa: E402,F401
                           FakeSamplerCustom, FakeVAE, node_module, reference_mask, run)


class IndexVAE(FakeVAE):
    """Frame-count faithful like FakeVAE, and value faithful: pixel frame 0 lives in channel 0 of
    latent 0, pixel frame 4k - 3 + j in channel j of latent k. A frame's value is read from its
    first pixel, and decoded back as a constant frame; no output clamp."""

    def encode(self, pixels):
        frames = pixels.shape[0]
        latent = torch.zeros(1, 16, ((frames - 1) // 4) + 1, pixels.shape[1] // LATENT_DOWN, pixels.shape[2] // LATENT_DOWN)
        values = pixels[:, 0, 0, 0]
        latent[0, 0, 0] = values[0]
        for f in range(1, frames):
            latent[0, (f - 1) % 4, (f - 1) // 4 + 1] = values[f]
        return latent

    def decode(self, latent):
        latents = latent.shape[2]
        values = torch.stack([latent[0, 0, 0, 0, 0]] + [latent[0, j, k, 0, 0] for k in range(1, latents) for j in range(4)])
        h, w = latent.shape[3] * LATENT_DOWN, latent.shape[4] * LATENT_DOWN
        return values.view(1, -1, 1, 1, 1).expand(1, len(values), h, w, 3).clone()

    process_output = staticmethod(lambda image: image)


class PoseFollowingSampler(FakeSamplerCustom):
    """The latent frames the noise mask leaves open become the pose conditioning's latents; the
    known ones (the previous frames) stay."""

    @classmethod
    def EXECUTE_NORMALIZED(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        FakeSamplerCustom.EXECUTE_NORMALIZED(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)
        samples = latent_image["samples"]
        generated = positive[0][1]["pose_video_latent"][..., :1, :1].expand_as(samples)
        known = latent_image.get("noise_mask")
        out = dict(latent_image, samples=generated if known is None else torch.where(known > 0, generated, samples))
        return FakeNodeOutput(out, dict(out))


class RecordingCLIPVisionEncode(FakeCLIPVisionEncode):
    images = []

    @classmethod
    def EXECUTE_NORMALIZED(cls, clip_vision, image, crop):
        cls.images.append((image.clone(), crop))
        return FakeCLIPVisionEncode.EXECUTE_NORMALIZED(clip_vision, image, crop)


@pytest.fixture
def aligned(node_module, monkeypatch):
    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "SamplerCustom", PoseFollowingSampler)
    return node_module


@pytest.fixture
def clip_run(node_module, monkeypatch):
    """(the sampler Names, the images CLIPVisionEncode was handed with their crop)."""
    RecordingCLIPVisionEncode.images = []
    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "CLIPVisionEncode", RecordingCLIPVisionEncode)
    return node_module, RecordingCLIPVisionEncode.images


def chunks_for(total):
    """81, then 76 new frames per chunk."""
    count = 1
    while 81 + 76 * (count - 1) < total:
        count += 1
    return count


# --- the regression test: output frame i is driving frame i ------------------------------------

@pytest.mark.parametrize("total", [81, 157, 240, 357, 449, 450])
def test_output_frame_i_shows_driving_frame_i(aligned, total):
    images, count, plan = run(aligned, pose_frames=total, node=SCAIL2, vae=IndexVAE())
    assert count == total and images.shape[0] == total
    # continuous: no repeated frame at a seam, no rewind, nothing skipped
    assert images[:, 0, 0, 0].tolist() == [float(i) for i in range(total)]
    chunks = chunks_for(total)
    assert plan == "{} -> {} produced -> {} frames (pose {}, overlap 5)".format(
        " + ".join(["81"] * chunks), 81 + 76 * (chunks - 1), total, total)


def test_core_gets_the_full_pose_and_the_chained_offsets(aligned):
    run(aligned, pose_frames=357, node=SCAIL2, vae=IndexVAE())
    # every chunk runs the full 81 frames, the last one too, on a pose held up to its end
    assert [c["length"] for c in Calls.animate] == [81] * 5
    assert [c["pose_in"] for c in Calls.animate] == [385] * 5
    assert [c["pose_frames"] for c in Calls.animate] == [81] * 5
    # the loop passes the returned offset (0, 81, 157, ...); core keeps 5 previous frames and
    # moves back by them, so the pose is read from 0, 76, 152, ...
    assert [c["previous"] for c in Calls.animate] == [None, 5, 5, 5, 5]
    assert [c["offset_in"] + (c["previous"] or 0) for c in Calls.animate] == [0, 81, 157, 233, 309]
    assert [c["offset_in"] for c in Calls.animate] == [0, 76, 152, 228, 304]
    assert [c["pose"] for c in Calls.animate] == [0.0, 76.0, 152.0, 228.0, 304.0]
    # the previous frames the model is seeded with are the driving frames the pose window starts on
    assert all(c["previous_first"] == c["pose"] for c in Calls.animate[1:])
    assert [s["seed"] for s in Calls.sampler] == [7, 8, 9, 10, 11]


def test_fixed_seed_mode(node_module):
    run(node_module, pose_frames=200, node=SCAIL2, seed=3, seed_mode="fixed")
    assert [s["seed"] for s in Calls.sampler] == [3, 3, 3]


def test_the_last_chunk_is_full_length_and_the_output_is_cut(node_module, caplog):
    caplog.set_level("INFO")
    images, count, plan = run(node_module, pose_frames=500, node=SCAIL2, total_frames=100)
    assert count == 100 and images.shape[0] == 100
    assert [c["length"] for c in Calls.animate] == [81, 81]
    assert plan == "81 + 81 -> 157 produced -> 100 frames (pose 500, overlap 5)"
    assert "every chunk runs the full 81 frames" in caplog.text


def test_the_pose_mask_is_held_like_the_pose(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=100, node=SCAIL2, total_frames=250)
    assert [c["mask_frames"] for c in Calls.animate] == [81] * len(Calls.animate)
    assert [c["mask"] for c in Calls.animate] == [c["pose"] for c in Calls.animate]
    assert Calls.animate[-1]["mask"] == 99.0  # the last mask frame, held
    held = [line for line in caplog.text.splitlines() if "held" in line]
    assert len(held) == 1 and "pose_video +209" in held[0] and "pose_video_mask +209" in held[0]


def test_a_pose_mask_of_another_length_is_an_error(node_module):
    with pytest.raises(ValueError, match="pose_video_mask has 80 frames but pose_video has 100: both come from the same driving video"):
        run(node_module, pose_frames=100, node=SCAIL2, pose_video_mask=torch.zeros(80, 64, 32, 3))
    assert Calls.animate == []


@pytest.mark.parametrize("width, height", [(48, 64), (32, 80)])
def test_width_and_height_must_be_divisible_by_32(node_module, width, height):
    with pytest.raises(ValueError, match="width and height must be divisible by 32 for SCAIL-2"):
        run(node_module, pose_frames=81, node=SCAIL2, width=width, height=height)
    assert Calls.animate == []


@pytest.mark.parametrize("replacement_mode, rendered", [(True, "animation"), (False, "replacement")])
def test_masks_rendered_for_the_other_mode_are_an_error(node_module, replacement_mode, rendered):
    with pytest.raises(ValueError, match="rendered for {} mode".format(rendered)):
        run(node_module, pose_frames=81, node=SCAIL2, replacement_mode=replacement_mode,
            reference_image_mask=reference_mask(not replacement_mode))
    assert Calls.animate == []


def test_a_mask_without_a_clear_background_is_logged(node_module, caplog):
    caplog.set_level("WARNING")
    grey = torch.full((1, 64, 32, 3), 0.5)
    run(node_module, pose_frames=81, node=SCAIL2, reference_image_mask=grey)
    assert "no clear white or black background" in caplog.text
    assert len(Calls.animate) == 1


def test_inputs_reach_core_under_its_names(node_module):
    run(node_module, pose_frames=81, node=SCAIL2, pose_strength=0.5, pose_start_percent=0.1, pose_end_percent=0.9)
    call = Calls.animate[0]
    assert (call["pose_strength"], call["pose_start"], call["pose_end"]) == (0.5, 0.1, 0.9)
    assert call["replacement_mode"] is False and call["previous_frame_count"] == 5
    assert call["width"] == 32 and call["height"] == 64
    positive = Calls.sampler[0]["positive"][0][1]
    assert positive["ref_mask_flag"] is True


def test_pose_start_after_end_is_an_error(node_module):
    with pytest.raises(ValueError, match="pose_start_percent"):
        run(node_module, pose_frames=81, node=SCAIL2, pose_start_percent=0.8, pose_end_percent=0.2)
    assert Calls.animate == []


def test_previous_frame_count_is_snapped_to_the_grid(node_module, caplog):
    caplog.set_level("INFO")
    images, count, plan = run(node_module, pose_frames=200, node=SCAIL2, previous_frame_count=7)
    assert count == 200 and plan.endswith("overlap 5)")
    assert all(c["previous_frame_count"] == 5 for c in Calls.animate)
    assert "previous_frame_count 7 is not on the 4k+1 grid; using 5" in caplog.text


def test_a_larger_overlap_keeps_the_frames_aligned(aligned):
    images, count, plan = run(aligned, pose_frames=300, node=SCAIL2, vae=IndexVAE(), previous_frame_count=9)
    assert images[:, 0, 0, 0].tolist() == [float(i) for i in range(300)]
    assert all(c["previous"] == 9 for c in Calls.animate[1:])


@pytest.mark.parametrize("frames_per_chunk, warned", [(81, False), (65, False), (49, True), (85, True)])
def test_chunks_off_the_trained_lengths_are_logged(node_module, caplog, frames_per_chunk, warned):
    caplog.set_level("WARNING")
    run(node_module, pose_frames=81, node=SCAIL2, frames_per_chunk=frames_per_chunk)
    assert ("the segment lengths SCAIL-2 was trained on" in caplog.text) == warned


def test_old_core_is_rejected(node_module, monkeypatch):
    class OldSCAIL(sys.modules["nodes"].NODE_CLASS_MAPPINGS["WanSCAILToVideo"]):
        RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")

    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "WanSCAILToVideo", OldSCAIL)
    with pytest.raises(RuntimeError, match="Update ComfyUI.*WanSCAILToVideo"):
        run(node_module, pose_frames=81, node=SCAIL2)


# --- CLIP vision -------------------------------------------------------------------------------

def test_clip_is_encoded_once_per_run_from_the_plain_reference_in_animation_mode(clip_run):
    module, images = clip_run
    reference = torch.rand(1, 64, 32, 3, generator=torch.Generator().manual_seed(0))
    run(module, pose_frames=240, node=SCAIL2, reference_image=reference)
    assert len(images) == 1
    image, crop = images[0]
    assert crop == "none" and torch.equal(image, reference)
    assert all(c["clip"] == ("clip", "cv", float(reference[0, 0, 0, 0]), "none") for c in Calls.animate)


def test_clip_gets_the_character_on_black_in_replacement_mode(clip_run):
    module, images = clip_run
    reference = torch.rand(1, 64, 32, 3, generator=torch.Generator().manual_seed(0))
    run(module, pose_frames=240, node=SCAIL2, reference_image=reference, replacement_mode=True)
    assert len(images) == 1
    image, crop = images[0]
    expected = torch.zeros_like(reference)
    expected[:, 16:48, 8:24] = reference[:, 16:48, 8:24]  # the blue person of reference_mask()
    assert crop == "none" and torch.equal(image, expected)
