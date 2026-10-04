"""Runs the nodes' chunk loop against fake core nodes.

ComfyUI itself is stubbed (comfy.*, nodes, latent_preview); torch is real so
the tensor plumbing (trim, cat, crop, pad) is exercised for real. Skipped when
torch is not installed.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (ANIMATE1, ANIMATE2, CLIP_MEAN, CLIP_STD, LATENT_DOWN, Calls, FakeCLIPVision,  # noqa: E402,F401
                           FakeNodeOutput, FakeProgressBar, FakeWanAnimate2ToVideo, core_concat_mask, fake_common_upscale,
                           node_module, official_clip_pixels, reference_concat_mask, run)


# --- shared behaviour, both nodes ---

@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_fixed_seed_mode(node_module, node):
    run(node_module, pose_frames=200, node=node, seed=3, seed_mode="fixed")
    assert [s["seed"] for s in Calls.sampler] == [3, 3, 3]


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_sigmas_override_skips_scheduler(node_module, caplog, node):
    caplog.set_level("INFO")
    override = torch.linspace(1.0, 0.0, 4)
    run(node_module, pose_frames=81, node=node, sigmas_override=override)
    assert Calls.sampler[0]["sigmas"] is override
    assert "sigmas_override" in caplog.text


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_sampler_uses_shift_patched_model(node_module, node):
    run(node_module, pose_frames=81, node=node, shift=8.0)
    assert Calls.sampler[0]["model"].shift == 8.0


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_empty_schedule_is_an_error(node_module, node):
    with pytest.raises(ValueError):
        run(node_module, pose_frames=81, node=node, denoise=0.0)


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_old_core_is_rejected(node_module, monkeypatch, node):
    animate_node = getattr(node_module, node).ANIMATE_NODE

    class OldAnimate(sys.modules["nodes"].NODE_CLASS_MAPPINGS[animate_node]):
        RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")

    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, animate_node, OldAnimate)
    with pytest.raises(RuntimeError, match="Update ComfyUI.*" + animate_node):
        run(node_module, pose_frames=81, node=node)


def test_a_missing_core_node_is_an_error(node_module):
    with pytest.raises(RuntimeError):
        node_module._call_node("NotACoreNode", given=1)


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_log_prefix_names_the_node(node_module, caplog, node):
    caplog.set_level("INFO")
    run(node_module, pose_frames=81, node=node)
    assert "[{}] chunk plan:".format(node) in caplog.text


# --- Wan Animate 2 node ---

def test_animate2_exact_length_from_pose(node_module):
    images, count, plan = run(node_module, pose_frames=360)
    assert images.shape[0] == 360
    assert count == 360
    assert plan == "81 + 81 + 81 + 81 + 41 -> 361 produced -> 360 frames (pose 360, overlap 1)"
    assert [c["length"] for c in Calls.animate] == [81, 81, 81, 81, 41]
    # first chunk has no anchor, every later chunk is seeded with the previous batch (core keeps 1 frame)
    assert Calls.animate[0]["continue"] is None
    assert all(c["continue"] == 1 for c in Calls.animate[1:])
    # offset chain: returned offset feeds the next call, adjusted by the overlap inside the node,
    # and the pose is read from exactly there (the last chunk from its window: it runs past frame 359)
    assert [c["offset_in"] for c in Calls.animate] == [0, 80, 160, 240, 0]
    assert [c["pose"] for c in Calls.animate] == [0.0, 80.0, 160.0, 240.0, 320.0]
    assert [s["seed"] for s in Calls.sampler] == [7, 8, 9, 10, 11]
    assert FakeProgressBar.instances[0].total == 5
    assert FakeProgressBar.instances[0].current == 5


def test_animate2_single_chunk_when_it_fits(node_module):
    images, count, plan = run(node_module, pose_frames=81)
    assert count == 81 and images.shape[0] == 81
    assert len(Calls.animate) == 1


def test_animate2_total_frames_widget_crops_pose(node_module):
    images, count, _ = run(node_module, pose_frames=500, total_frames=100)
    assert count == 100
    assert [c["length"] for c in Calls.animate] == [81, 21]


def test_animate2_total_beyond_pose_holds_last_frame(node_module, caplog):
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=100, total_frames=250, frames_per_chunk=49)
    assert count == 250
    assert "held" in caplog.text
    assert plan.endswith("(pose 100, overlap 1)")


def test_animate2_pass_through_inputs_reach_core(node_module):
    run(node_module, pose_frames=81, pose_strength=0.5, positive_pose=[["motion", {}]])
    assert Calls.animate[0]["pose_strength"] == 0.5
    assert Calls.animate[0]["positive_pose"] == [["motion", {}]]


def core_vae_frame(image, width=32, height=64):
    """The first frame of ``image`` as core's WanAnimateToVideo and WanAnimate2ToVideo resize the
    reference and the pose video for the VAE: center-cropped to the generation aspect and resized
    area to run()'s 32 x 64 (comfy_extras/nodes_wan.py, comfy/utils.py common_upscale)."""
    return fake_common_upscale(image[:1].movedim(-1, 1), width, height, "area", "center").movedim(1, -1)


def other_aspect(frames, seed=0):
    """``frames`` random frames of another aspect than run()'s 32 x 64."""
    return torch.rand(frames, 96, 40, 3, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_clip_vision_encodes_the_vae_reference_as_official_once_per_run(node_module, caplog, node):
    caplog.set_level("INFO")
    reference, clip_vision = other_aspect(1), FakeCLIPVision()
    run(node_module, pose_frames=200, node=node, reference_image=reference, clip_vision=clip_vision, clip_vision_output="clip")
    pixels, layer = clip_vision.calls[0]  # prepare encodes the reference before any chunk (Wan Animate 2: then the pose per chunk)
    assert len(clip_vision.calls) == (1 if node == ANIMATE1 else 4)
    assert layer == -2  # the penultimate block: official use_31_block of the 32-block ViT-H
    assert pixels.shape == (1, 3, 224, 224) and pixels.dtype == torch.float32
    assert torch.allclose(pixels, official_clip_pixels(core_vae_frame(reference)), atol=1e-5, rtol=0)
    # core's CLIPVisionEncode antialiases and rounds to 8 bit: not what official feeds
    core = torch.nn.functional.interpolate(core_vae_frame(reference).movedim(-1, 1), size=(224, 224), mode="bicubic", antialias=True)
    core = (torch.clip(255.0 * core, 0, 255).round() / 255.0 - CLIP_MEAN) / CLIP_STD
    assert not torch.allclose(pixels, core, atol=1e-3, rtol=0)
    # every chunk gets the outputs of that one encode in place of the connected clip_vision_output
    outputs = [c["clip"] for c in Calls.animate]
    assert len(outputs) == 3 and all(o is outputs[0] for o in outputs)
    assert outputs[0].penultimate_hidden_states is pixels
    assert caplog.text.count("clip_vision_output is ignored") == 1


@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2])
def test_without_clip_vision_the_connected_clip_vision_output_reaches_core(node_module, node):
    run(node_module, pose_frames=200, node=node, reference_image=other_aspect(1), clip_vision_output="clip")
    assert [c["clip"] for c in Calls.animate] == ["clip"] * 3


def test_animate2_pose_clip_reencoded_per_chunk_as_official_when_clip_vision_connected(node_module, caplog):
    caplog.set_level("INFO")
    pose, clip_vision = other_aspect(200, seed=1), FakeCLIPVision()
    run(node_module, pose_frames=200, pose_video=pose, clip_vision_output_pose="static", clip_vision=clip_vision)
    # the first frame of each chunk's pose window: 0, then 80k (offset moved back by the 1 seed frame),
    # as the VAE gets it, preprocessed as official
    pose_calls = clip_vision.calls[1:]
    assert len(pose_calls) == 3
    for (pixels, layer), first in zip(pose_calls, (0, 80, 160)):
        assert layer == -2
        assert torch.allclose(pixels, official_clip_pixels(core_vae_frame(pose[first:first + 1])), atol=1e-5, rtol=0)
    assert [c["clip_pose"].penultimate_hidden_states for c in Calls.animate] == [pixels for pixels, _ in pose_calls]
    assert "re-encoded per chunk" in caplog.text

    Calls.animate, Calls.sampler = [], []
    run(node_module, pose_frames=200, clip_vision_output_pose="static")
    assert [c["clip_pose"] for c in Calls.animate] == ["static", "static", "static"]


def test_animate2_attn_log_scale_installs_the_attention_override(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=81)
    assert "optimized_attention_override" not in Calls.sampler[0]["model"].model_options["transformer_options"]

    Calls.animate, Calls.sampler = [], []
    run(node_module, pose_frames=81, attn_log_scale=-1.3, shift=5.0)
    model = Calls.sampler[0]["model"]
    assert callable(model.model_options["transformer_options"]["optimized_attention_override"])
    assert model.shift == 5.0  # the clone keeps ModelSamplingSD3's patch
    assert "attn_log_scale -1.30" in caplog.text


def test_animate2_pose_percent_validation(node_module):
    with pytest.raises(ValueError):
        run(node_module, pose_frames=81, pose_start_percent=0.8, pose_end_percent=0.2)
    assert Calls.animate == []


def test_animate2_overlap_follows_the_core_constant(node_module, monkeypatch):
    # a core that keeps 5 motion frames: the loop must still land exactly
    monkeypatch.setattr(FakeWanAnimate2ToVideo, "CONTINUE_MOTION_FRAMES", 5)
    images, count, plan = run(node_module, pose_frames=200, frames_per_chunk=49)
    assert count == 200
    assert plan.endswith("overlap 5)")
    assert all(c["continue"] == 5 for c in Calls.animate[1:])


class _KeepsFive(FakeWanAnimate2ToVideo):
    CONTINUE_MOTION_FRAMES = 5


class KeepsFiveFrames(FakeWanAnimate2ToVideo):
    """Would keep 5 continue_motion frames while its constant says 1."""

    CONTINUE_MOTION_FRAMES = 1

    @classmethod
    def EXECUTE_NORMALIZED(cls, **kwargs):
        return _KeepsFive.EXECUTE_NORMALIZED(**kwargs)


def test_animate2_is_seeded_with_the_frames_its_constant_names(node_module, monkeypatch, caplog):
    # the loop hands the core node only the anchor frames its constant names: it keeps those
    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "WanAnimate2ToVideo", KeepsFiveFrames)
    caplog.set_level("INFO")
    images, count, plan = run(node_module, pose_frames=200, frames_per_chunk=49)
    assert count == 200 and plan.endswith("overlap 1)")
    assert all(c["continue"] == 1 for c in Calls.animate[1:]) and "planner assumed" not in caplog.text


class TrimsFive(FakeWanAnimate2ToVideo):
    """Trims 5 decoded frames back off a chained chunk, whatever it was seeded with."""

    @classmethod
    def EXECUTE_NORMALIZED(cls, **kwargs):
        positive, negative, latent, trim_latent, trim_image, offset = super().EXECUTE_NORMALIZED(**kwargs).args
        return FakeNodeOutput(positive, negative, latent, trim_latent, 5 if trim_image else 0, offset)


def test_animate2_overlap_self_correction_is_logged(node_module, monkeypatch, caplog):
    # a core that trims more than the frames it keeps decode to: the loop takes the trim it found
    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "WanAnimate2ToVideo", TrimsFive)
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, frames_per_chunk=49)
    assert "planner assumed 1; using 5" in caplog.text and " ran: " in caplog.text


def test_animate2_chunk_and_step_logging(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, frames_per_chunk=81, seed=7)
    # 81 + 81 + 41 -> 81 + 80 + 40 = 201 produced, cropped to 200
    assert "chunk 1/3 (frames 1-81/200): length 81, pose offset 0, seed 7" in caplog.text
    assert "chunk 2/3 (frames 82-161/200): length 81, pose offset 81, seed 8" in caplog.text
    assert "chunk 3/3 (frames 162-200/200): length 41, pose offset 161, seed 9" in caplog.text
    assert "chunk 2/3 (frames 82-161/200) done: 80 new frames, 161/200 total" in caplog.text


# --- Wan Animate (1) node ---

def test_animate1_exact_length_from_pose(node_module):
    images, count, plan = run(node_module, pose_frames=360, node=ANIMATE1, frames_per_chunk=77)
    assert images.shape[0] == 360
    assert count == 360
    assert plan == "77 + 77 + 77 + 77 + 73 -> 361 produced -> 360 frames (pose 360, overlap 5)"
    assert [c["length"] for c in Calls.animate] == [77, 77, 77, 77, 73]
    assert Calls.animate[0]["continue"] is None
    assert all(c["continue"] == 5 for c in Calls.animate[1:])
    assert all(c["max_frames"] == 5 for c in Calls.animate)
    # offset chain: each chunk starts 5 frames before the previous one ended (the last one in
    # its window: it runs past frame 359)
    assert [c["offset_in"] for c in Calls.animate] == [0, 72, 144, 216, 0]
    # and reads the pose from exactly there
    assert [c["pose"] for c in Calls.animate] == [0.0, 72.0, 144.0, 216.0, 288.0]
    assert [s["seed"] for s in Calls.sampler] == [7, 8, 9, 10, 11]
    assert FakeProgressBar.instances[0].total == 5
    assert FakeProgressBar.instances[0].current == 5


def test_animate1_single_chunk_when_it_fits(node_module):
    images, count, plan = run(node_module, pose_frames=77, node=ANIMATE1, frames_per_chunk=77)
    assert count == 77 and images.shape[0] == 77
    assert len(Calls.animate) == 1


def test_animate1_larger_overlap_widget(node_module):
    images, count, plan = run(node_module, pose_frames=200, node=ANIMATE1, frames_per_chunk=77, continue_motion_max_frames=9)
    assert count == 200
    assert plan == "77 + 77 + 65 -> 201 produced -> 200 frames (pose 200, overlap 9)"
    assert [c["pose"] for c in Calls.animate] == [0.0, 68.0, 136.0]
    assert all(c["continue"] == 9 for c in Calls.animate[1:])


def test_animate1_overlap_above_half_the_chunk_keeps_the_full_seed(node_module, caplog):
    # a middle chunk adds 17 - 13 = 4 new frames; the next chunk must still be seeded with 13,
    # or core moves the offset back by 4, trims only 1 and the pose falls behind the output
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=60, node=ANIMATE1, frames_per_chunk=17, continue_motion_max_frames=13)
    assert count == 60
    assert all(c["continue"] == 13 for c in Calls.animate[1:])
    assert [c["pose"] for c in Calls.animate] == [min(4.0 * k, 59.0) for k in range(len(Calls.animate))]
    assert "planner assumed" not in caplog.text


def test_animate1_snaps_continue_motion_max_frames_to_grid(node_module, caplog):
    caplog.set_level("INFO")
    images, count, plan = run(node_module, pose_frames=200, node=ANIMATE1, frames_per_chunk=77, continue_motion_max_frames=7)
    assert count == 200
    assert plan.endswith("overlap 5)")
    assert all(c["max_frames"] == 5 for c in Calls.animate)
    assert all(c["continue"] == 5 for c in Calls.animate[1:])
    assert "continue_motion_max_frames 7 is not on the 4k+1 grid; using 5" in caplog.text


def test_animate1_overlap_must_be_smaller_than_chunk(node_module):
    with pytest.raises(ValueError, match="must exceed the overlap"):
        run(node_module, pose_frames=200, node=ANIMATE1, frames_per_chunk=77, continue_motion_max_frames=77)
    assert Calls.animate == []


def _reads(call, key):
    """What core reads of a video it was handed: `length` frames from the offset it moved back to."""
    return call[key][call["offset_in"]:call["offset_in"] + call["length"]]


def test_animate1_optional_videos_reach_core(node_module):
    face = torch.rand(81, 16, 16, 3)
    background = torch.rand(81, 64, 32, 3)
    mask = torch.zeros(1, 64, 32)  # a still: core repeats it; nothing of the background is painted
    run(node_module, pose_frames=81, node=ANIMATE1, frames_per_chunk=77,
        clip_vision_output="clip", face_video=face, background_video=background, character_mask=mask)
    assert len(Calls.animate) == 2
    for call in Calls.animate:
        first = int(call["pose"])  # the driving frame the chunk reads from
        assert call["clip"] == "clip"
        assert torch.equal(_reads(call, "face"), face[first:first + call["length"]])
        assert torch.equal(_reads(call, "background"), background[first:first + call["length"]])
        assert call["mask"] is mask


def test_animate1_total_beyond_pose_keeps_pose_on_every_chunk(node_module, caplog):
    # WanAnimateToVideo silently drops a pose video the offset has run past;
    # the up-front pad means every chunk still sees one
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=100, node=ANIMATE1, total_frames=250, frames_per_chunk=49)
    assert count == 250
    assert "held" in caplog.text
    assert plan.endswith("(pose 100, overlap 5)")
    assert all(c["pose"] is not None for c in Calls.animate)
    assert Calls.animate[-1]["pose"] == 99.0


def _videos(frames):
    """face / background / multi-frame mask whose content is the frame index, so a held frame is
    visible; the mask on the right half of the frame only, so the left half of the background is
    never painted."""
    index = torch.arange(frames, dtype=torch.float32)
    mask = torch.zeros(frames, 64, 32)
    mask[:, :, 16:] = index.view(-1, 1, 1)
    return dict(face_video=index.view(-1, 1, 1, 1).expand(-1, 8, 8, 3).contiguous(),
                background_video=index.view(-1, 1, 1, 1).expand(-1, 64, 32, 3).contiguous(),
                character_mask=mask)


def _painted(videos):
    """The background black wherever the mask is above 0: WanAnimate Preprocess's bg_images."""
    return torch.where(videos["character_mask"].unsqueeze(-1) > 0, torch.zeros(()), videos["background_video"])


@pytest.mark.parametrize("frames, produced", [(150, 153), (152, 153)])
def test_animate1_overshooting_last_chunk_holds_every_video(node_module, caplog, frames, produced):
    # off the 4k+1 grid the last chunk is snapped up and runs past the source: pose, face and
    # background must reach that far, or core zero-pads the face and greys the background. The
    # mask is not held: past its end core leaves its rows unknown
    caplog.set_level("INFO")
    images, count, plan = run(node_module, pose_frames=frames, node=ANIMATE1, frames_per_chunk=81, **_videos(frames))
    assert count == frames
    assert plan.startswith("81 + 77 -> {} produced".format(produced))
    last = Calls.animate[-1]
    first = int(last["pose"])  # the driving frame the last chunk reads from
    assert first + last["length"] == produced
    videos = _videos(frames)
    for key, source in (("face", videos["face_video"]), ("background", _painted(videos))):
        held = torch.cat((source, source[-1:].expand(produced - frames, *source.shape[1:])))  # the last frame, held
        assert torch.equal(_reads(last, key), held[first:produced])
    assert torch.equal(_reads(last, "mask"), videos["character_mask"][first:])  # cut at its end, never held
    padded = [line for line in caplog.text.splitlines() if "held" in line]
    assert len(padded) == 1 and "character_mask" not in padded[0]
    for name in ("pose_video", "face_video", "background_video"):
        assert "{} +{}".format(name, produced - frames) in padded[0]


def test_animate1_total_beyond_inputs_holds_every_video_but_the_mask(node_module, caplog):
    # total_frames past the input length: once the offset passes a video's end core drops it
    # (face driving and, in replacement mode, the scene), so they are held like the pose; the
    # mask is not, past its end the character may be anywhere
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=100, node=ANIMATE1, total_frames=250, frames_per_chunk=49, **_videos(100))
    assert count == 250
    first = 0  # the driving frame each chunk reads from: the next one starts 5 frames before this one ends
    for call in Calls.animate:
        seek, length = call["offset_in"], call["length"]
        for key in ("face", "background"):
            read = call[key][seek:seek + length]
            assert read[:, 0, 0, 0].tolist() == [float(min(i, 99)) for i in range(first, first + length)]
        assert call["mask"][seek:seek + length][:, 0, -1].tolist() == [float(i) for i in range(first, min(first + length, 100))]
        first += length - 5
    assert "total_frames (250) exceeds" in caplog.text


def test_animate1_single_frame_mask_is_not_padded(node_module):
    mask = torch.ones(1, 64, 32)
    run(node_module, pose_frames=150, node=ANIMATE1, character_mask=mask)
    assert all(call["mask"] is mask for call in Calls.animate)


def test_animate1_grid_inputs_reach_core_unchanged(node_module, caplog):
    # 4k+1 past the seed: the plan ends exactly on the last frame, nothing is padded or logged
    caplog.set_level("INFO")
    videos = _videos(161)
    images, count, plan = run(node_module, pose_frames=161, node=ANIMATE1, frames_per_chunk=81, **videos)
    assert plan.startswith("81 + 81 + 9 -> 161 produced")
    for call in Calls.animate:
        first, length = int(call["pose"]), call["length"]
        assert torch.equal(_reads(call, "face"), videos["face_video"][first:first + length])
        assert torch.equal(_reads(call, "background"), _painted(videos)[first:first + length])
        assert torch.equal(_reads(call, "mask"), videos["character_mask"][first:first + length])
    assert "held" not in caplog.text


def test_animate2_overshooting_last_chunk_is_logged(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=150)
    padded = [line for line in caplog.text.splitlines() if "held" in line]
    assert len(padded) == 1 and "pose_video +3" in padded[0] and "153" in padded[0]


def test_animate1_mask_repair_reaches_sampler_only_with_character_mask(node_module):
    torch.manual_seed(0)
    frames = 200
    # longer than the video so every chunk, including the short last one, is fully covered
    character_mask = (torch.rand(frames + 80, 64 // LATENT_DOWN, 32 // LATENT_DOWN) > 0.5).float()
    background = torch.zeros(frames + 80, 64 // LATENT_DOWN, 32 // LATENT_DOWN, 3)  # the mask's size: it is painted under it
    run(node_module, pose_frames=frames, node=ANIMATE1, frames_per_chunk=77, character_mask=character_mask, background_video=background)
    assert [c["length"] for c in Calls.animate] == [77, 77, 57]
    for call, sampled in zip(Calls.animate, Calls.sampler):
        mask = sampled["positive"][0][1]["concat_mask"]
        offset, length = int(call["pose"]), call["length"]  # the driving frame the chunk reads from
        seed = 0 if call["continue"] is None else call["continue"]
        assert torch.equal(mask, reference_concat_mask((length - 1) // 4 + 1, 8, 4, seed, character_mask[offset:offset + length]))
        assert sampled["negative"][0][1]["concat_mask"] is mask  # shared tensor, repaired once

    Calls.animate, Calls.sampler = [], []
    run(node_module, pose_frames=frames, node=ANIMATE1, frames_per_chunk=77)
    for call, sampled in zip(Calls.animate, Calls.sampler):
        seed = 0 if call["continue"] is None else call["continue"]
        seed_latents = 0 if seed == 0 else ((seed - 1) // 4) + 1
        assert torch.equal(sampled["positive"][0][1]["concat_mask"], core_concat_mask((call["length"] - 1) // 4 + 1, 8, 4, seed_latents, seed, None))

    # the mask marks where the character goes in each background frame: a mask video of
    # another length than the background is an error, before anything is sampled
    Calls.animate, Calls.sampler = [], []
    with pytest.raises(ValueError, match="character_mask has 100 frames but background_video has 200"):
        run(node_module, pose_frames=frames, node=ANIMATE1, frames_per_chunk=77, character_mask=character_mask[:100],
            background_video=background[:frames])
    assert Calls.animate == []  # it stops before anything is sampled


def test_animate1_chunk_logging(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, node=ANIMATE1, frames_per_chunk=77, seed=7)
    # 77 + 77 + 57 -> 77 + 72 + 52 = 201 produced, cropped to 200
    assert "chunk 1/3 (frames 1-77/200): length 77, pose offset 0, seed 7" in caplog.text
    assert "chunk 2/3 (frames 78-149/200): length 77, pose offset 77, seed 8" in caplog.text
    assert "chunk 3/3 (frames 150-200/200): length 57, pose offset 149, seed 9" in caplog.text
    assert "chunk 2/3 (frames 78-149/200) done: 72 new frames, 149/200 total" in caplog.text


def test_wan_beta_is_used_without_basic_scheduler(node_module, caplog):
    pytest.importorskip("scipy")
    caplog.set_level("INFO")
    run(node_module, pose_frames=81, scheduler="wan_beta", steps=4)
    sigmas = Calls.sampler[0]["sigmas"].tolist()
    assert [round(v, 4) for v in sigmas] == [1.0, 0.7313, 0.2931, 0.0244, 0.0]
    assert "sigmas (wan_beta): 1.0000, 0.7313, 0.2931, 0.0244, 0.0000" in caplog.text


def test_step_logger_labels_the_live_tqdm_bar(node_module):
    tqdm = pytest.importorskip("tqdm")
    import io

    class Inner:
        def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            # k-diffusion creates the bar inside the loop and drives the callback from it
            with tqdm.tqdm(total=len(sigmas) - 1, file=io.StringIO()) as bar:
                for i in range(len(sigmas) - 1):
                    callback(i, "x0", "x", len(sigmas) - 1)
                    bar.update(1)
                return bar.desc

    logger = node_module._StepLogger(Inner(), "[Node]")
    logger.label = "chunk 2/3 (frames 82-161/200)"
    assert logger.sample("wrap", [3, 2, 1, 0], {}, None, "noise") == "chunk 2/3 (frames 82-161/200): "


def test_step_logger_wraps_callback_and_delegates(node_module, caplog):
    caplog.set_level("INFO")
    seen = []

    class Inner:
        extra_options = {"eta": 1.0}

        def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
            for i in range(len(sigmas) - 1):
                callback(i, "x0", "x", len(sigmas) - 1)
            return "samples"

    logger = node_module._StepLogger(Inner(), "[Node]")
    logger.label = "chunk 2/3 (frames 82-161/200)"
    assert logger.extra_options == {"eta": 1.0}
    result = logger.sample("wrap", [3, 2, 1, 0], {}, lambda *a: seen.append(a), "noise")
    assert result == "samples"
    assert seen == [(0, "x0", "x", 3), (1, "x0", "x", 3), (2, "x0", "x", 3)]
    assert "[Node] chunk 2/3 (frames 82-161/200) step 1/3" in caplog.text
    assert "[Node] chunk 2/3 (frames 82-161/200) step 3/3" in caplog.text
