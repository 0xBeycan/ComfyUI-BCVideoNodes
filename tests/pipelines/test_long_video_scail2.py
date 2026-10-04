"""The SCAIL-2 Long Video Sampler's chunk loop against the fake WanSCAILToVideo.

ComfyUI itself is stubbed (comfy.*, nodes); torch is real. For the end-to-end alignment the
`aligned` fixture and IndexVAE (sampler_fakes) stand in for the model: with run()'s frame-index
pose video, output frame i must then show driving frame i. Skipped when torch is not installed.
"""

import re
import sys

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (CLIP_MEAN, CLIP_STD, SCAIL2, Calls, FakeCLIPVision, FakeModel, IndexVAE, aligned,  # noqa: E402,F401
                           fake_common_upscale, node_module, official_clip_pixels, reference_mask, run)


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


def test_core_gets_the_pose_and_the_chained_offsets(aligned):
    run(aligned, pose_frames=357, node=SCAIL2, vae=IndexVAE())
    # every chunk runs the full 81 frames, the last one too, on a pose held up to its end
    assert [c["length"] for c in Calls.animate] == [81] * 5
    assert [c["pose_frames"] for c in Calls.animate] == [81] * 5
    # the loop passes the returned offset (0, 81, 157, ...); core keeps 5 previous frames and
    # moves back by them, so the pose is read from 0, 76, 152, ...
    assert [c["previous"] for c in Calls.animate] == [None, 5, 5, 5, 5]
    assert [c["offset_in"] + (c["previous"] or 0) for c in Calls.animate[:4]] == [0, 81, 157, 233]
    assert [c["pose"] for c in Calls.animate] == [0.0, 76.0, 152.0, 228.0, 304.0]
    # the pose itself while a chunk is inside it; the last chunk runs past frame 356 and gets
    # just the window it reads, held past the end, with the offset into it
    assert [c["pose_in"] for c in Calls.animate] == [357] * 4 + [81]
    assert [c["offset_in"] for c in Calls.animate] == [0, 76, 152, 228, 0]
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


def test_a_pose_mask_of_the_pose_size_passes(node_module):
    run(node_module, pose_frames=81, node=SCAIL2, pose_video_mask=torch.zeros(81, 64, 32, 3))
    assert len(Calls.animate) == 1


# run()'s pose video is 64 high and 32 wide: another height, another width, half the size
@pytest.mark.parametrize("height, width, size", [(48, 32, "32x48"), (64, 64, "64x64"), (32, 16, "16x32")])
def test_a_pose_mask_of_another_size_is_an_error(node_module, height, width, size):
    message = ("pose_video_mask is {} but pose_video is 32x64: both come from the same driving video; re-run the "
               "preprocess (SCAIL-2 Preprocess), or resize the mask to the pose video's size.".format(size))
    with pytest.raises(ValueError, match=re.escape(message)):
        run(node_module, pose_frames=81, node=SCAIL2, pose_video_mask=torch.zeros(81, height, width, 3))
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


def driving_mask(replacement_mode, frames=81):
    """A colored driving mask of run()'s 64 x 32 size: a blue person in the middle, on black
    (animation mode) or white (replacement mode)."""
    mask = torch.full((frames, 64, 32, 3), 1.0 if replacement_mode else 0.0)
    mask[:, 16:48, 8:24] = torch.tensor([0.0, 0.0, 1.0])
    return mask


@pytest.mark.parametrize("replacement_mode, rendered, background", [(True, "animation", "black"),
                                                                    (False, "replacement", "white")])
def test_a_driving_mask_rendered_for_the_other_mode_is_an_error(node_module, replacement_mode, rendered, background):
    expected = ("pose_video_mask was rendered for {} mode ({} background) but replacement_mode is {}: set "
                "replacement_mode to {}".format(rendered, background, replacement_mode, not replacement_mode))
    with pytest.raises(ValueError, match=expected.replace("(", r"\(").replace(")", r"\)")):
        run(node_module, pose_frames=81, node=SCAIL2, replacement_mode=replacement_mode,
            pose_video_mask=driving_mask(not replacement_mode))
    assert Calls.animate == []


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_a_driving_mask_rendered_for_the_mode_runs_without_a_log_line(node_module, caplog, replacement_mode):
    caplog.set_level("WARNING")
    run(node_module, pose_frames=81, node=SCAIL2, replacement_mode=replacement_mode,
        pose_video_mask=driving_mask(replacement_mode))
    assert "pose_video_mask" not in caplog.text
    assert len(Calls.animate) == 1


def test_a_driving_mask_without_a_clear_background_is_logged(node_module, caplog):
    caplog.set_level("WARNING")
    run(node_module, pose_frames=81, node=SCAIL2, pose_video_mask=torch.full((81, 64, 32, 3), 0.5))
    assert "pose_video_mask has no clear white or black background; cannot check that it was rendered for animation mode" \
        in caplog.text
    assert "reference_image_mask has no clear" not in caplog.text
    assert len(Calls.animate) == 1


def test_the_reference_mask_is_checked_before_the_driving_mask(node_module):
    with pytest.raises(ValueError, match="^reference_image_mask was rendered for animation mode"):
        run(node_module, pose_frames=81, node=SCAIL2, replacement_mode=True, reference_image_mask=reference_mask(False),
            pose_video_mask=driving_mask(False))


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

def core_vae_reference(reference, width=32, height=64):
    """The reference core's WanSCAILToVideo VAE-encodes: center-cropped to the generation aspect
    and resized bicubic to run()'s 32 x 64 (comfy/utils.py common_upscale)."""
    return fake_common_upscale(reference.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)


def clip_run(module, **overrides):
    """Runs three chunks with a reference of another aspect than the generation size; returns
    (the reference, the FakeCLIPVision the run encoded with)."""
    reference = torch.rand(1, 96, 40, 3, generator=torch.Generator().manual_seed(0))
    clip_vision = FakeCLIPVision()
    run(module, pose_frames=240, node=SCAIL2, reference_image=reference, clip_vision=clip_vision, **overrides)
    return reference, clip_vision


def test_clip_gets_the_vae_reference_preprocessed_as_official_once_per_run(node_module):
    reference, clip_vision = clip_run(node_module)
    assert len(clip_vision.calls) == 1
    pixels, layer = clip_vision.calls[0]
    expected = official_clip_pixels(core_vae_reference(reference))
    assert layer == -2  # the penultimate block: official use_31_block of the 32-block ViT-H
    assert pixels.shape == (1, 3, 224, 224) and pixels.dtype == torch.float32
    assert torch.allclose(pixels, expected, atol=1e-5, rtol=0)
    # core's CLIPVisionEncode (crop none) antialiases and rounds to 8 bit: not what official feeds
    core = torch.nn.functional.interpolate(core_vae_reference(reference).movedim(-1, 1), size=(224, 224), mode="bicubic", antialias=True)
    core = (torch.clip(255.0 * core, 0, 255).round() / 255.0 - CLIP_MEAN) / CLIP_STD
    assert not torch.allclose(pixels, core, atol=1e-3, rtol=0)
    # every chunk gets the outputs of that one encode, as CLIPVisionEncode returns them
    outputs = [c["clip"] for c in Calls.animate]
    assert len(outputs) == 4 and all(o is outputs[0] for o in outputs)
    assert outputs[0].penultimate_hidden_states is pixels and outputs[0].last_hidden_state is pixels
    assert outputs[0].image_sizes == [pixels.shape[1:]]


def test_clip_gets_the_character_on_black_in_replacement_mode(node_module):
    reference, clip_vision = clip_run(node_module, replacement_mode=True)
    expected = torch.zeros(1, 64, 32, 3)
    expected[:, 16:48, 8:24] = core_vae_reference(reference)[:, 16:48, 8:24]  # the blue person of reference_mask()
    assert torch.allclose(clip_vision.calls[0][0], official_clip_pixels(expected), atol=1e-5, rtol=0)


def test_clip_of_a_half_reference_is_read_at_its_8_bit_levels(node_module):
    reference = torch.randint(0, 256, (1, 96, 40, 3), generator=torch.Generator().manual_seed(0)) / 255.0
    clip_vision = FakeCLIPVision()
    run(node_module, pose_frames=81, node=SCAIL2, reference_image=reference.half(), clip_vision=clip_vision)
    assert torch.allclose(clip_vision.calls[0][0], official_clip_pixels(core_vae_reference(reference)), atol=1e-5, rtol=0)


def test_a_clip_vision_model_other_than_vit_h_is_an_error(node_module):
    clip_vision = FakeCLIPVision()
    clip_vision.model_type = "siglip_vision_model"
    with pytest.raises(ValueError, match="clip_vision is a siglip_vision_model model; this sampler needs the CLIP ViT-H"):
        run(node_module, pose_frames=81, node=SCAIL2, clip_vision=clip_vision)


# --- the pose RoPE -----------------------------------------------------------------------------

def test_official_pose_rope_is_an_object_patch_on_the_clone(node_module):
    model = FakeModel()
    run(node_module, pose_frames=81, node=SCAIL2, model=model)
    patched = Calls.sampler[0]["model"]
    assert list(patched.object_patches) == ["diffusion_model.rope_encode"]
    # it wraps the model's own rope_encode: a call without pose tokens is core's call
    rope_encode = patched.object_patches["diffusion_model.rope_encode"]
    assert rope_encode(3, 8, 4, device="cpu", dtype=torch.bfloat16) == ("core rope_encode", (3, 8, 4), {"device": "cpu", "dtype": torch.bfloat16})
    assert model.object_patches == {}  # the model the node was handed keeps core's rope_encode


# --- multi-reference: the primary and an extra reference (SCAIL-2 Preprocess's face close-up) ----

def two_references(replacement_mode):
    """(reference_image [2, 96, 40, 3], reference_image_mask [2, 64, 32, 3]): the primary's mask
    on the mode's background, the extra's blue on black in both modes."""
    reference = torch.rand(2, 96, 40, 3, generator=torch.Generator().manual_seed(1))
    mask = reference_mask(replacement_mode, frames=2)
    mask[1] = 0.0
    mask[1, 8:40, 4:28] = torch.tensor([0.0, 0.0, 1.0])
    return reference, mask


@pytest.mark.parametrize("replacement_mode", [False, True])
def test_two_references_reach_core_whole_and_clip_reads_the_primary(node_module, replacement_mode):
    reference, mask = two_references(replacement_mode)
    clip_vision = FakeCLIPVision()
    run(node_module, pose_frames=240, node=SCAIL2, reference_image=reference, reference_image_mask=mask,
        clip_vision=clip_vision, replacement_mode=replacement_mode)
    # every chunk's core call gets both references and both masks, as linked
    assert len(Calls.animate) == 4
    assert all(c["reference"] is reference and c["reference_mask"] is mask for c in Calls.animate)
    # CLIP is encoded once, from the primary alone (in replacement mode cut by the primary's mask)
    primary = core_vae_reference(reference[:1])
    if replacement_mode:
        cut = torch.zeros_like(primary)
        cut[:, 16:48, 8:24] = primary[:, 16:48, 8:24]
        primary = cut
    [(pixels, _)] = clip_vision.calls
    assert pixels.shape == (1, 3, 224, 224)
    assert torch.allclose(pixels, official_clip_pixels(primary), atol=1e-5, rtol=0)


@pytest.mark.parametrize("images, masks", [(2, 1), (1, 2)])
def test_references_and_masks_of_another_count_are_an_error(node_module, images, masks):
    with pytest.raises(ValueError, match=r"reference_image has {} frame\(s\) but reference_image_mask {}: .* Link the "
                                         r"sampler's reference_image from SCAIL-2 Preprocess's reference_images".format(images, masks)):
        run(node_module, pose_frames=81, node=SCAIL2, reference_image=torch.zeros(images, 64, 32, 3),
            reference_image_mask=reference_mask(frames=masks))
    assert Calls.animate == []
