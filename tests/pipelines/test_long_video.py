"""Runs the nodes' chunk loop against fake core nodes.

ComfyUI itself is stubbed (comfy.*, nodes, latent_preview); torch is real so
the tensor plumbing (trim, cat, crop, pad) is exercised for real. Skipped when
torch is not installed.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (ANIMATE1, ANIMATE2, LATENT_DOWN, Calls, FakeProgressBar,  # noqa: E402,F401
                           FakeWanAnimate2ToVideo, core_concat_mask, node_module, reference_concat_mask, run)


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
    # offset chain: returned offset feeds the next call, adjusted by the overlap inside the node
    assert [c["offset_in"] for c in Calls.animate] == [0, 80, 160, 240, 320]
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


def test_animate2_pose_clip_reencoded_per_chunk_when_clip_vision_connected(node_module, caplog):
    caplog.set_level("INFO")
    run(node_module, pose_frames=200, clip_vision_output_pose="static", clip_vision="cv")
    # the first frame of each chunk's pose window: 0, then 80k (offset moved back by the 1 seed frame)
    assert [c["clip_pose"] for c in Calls.animate] == [("clip", "cv", 0.0, "none"), ("clip", "cv", 80.0, "none"), ("clip", "cv", 160.0, "none")]
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
    # offset chain: each chunk starts 5 frames before the previous one ended
    assert [c["offset_in"] for c in Calls.animate] == [0, 72, 144, 216, 288]
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
    assert [c["offset_in"] for c in Calls.animate] == [0, 68, 136]
    assert all(c["continue"] == 9 for c in Calls.animate[1:])


def test_animate1_overlap_above_half_the_chunk_keeps_the_full_seed(node_module, caplog):
    # a middle chunk adds 17 - 13 = 4 new frames; the next chunk must still be seeded with 13,
    # or core moves the offset back by 4, trims only 1 and the pose falls behind the output
    caplog.set_level("WARNING")
    images, count, plan = run(node_module, pose_frames=60, node=ANIMATE1, frames_per_chunk=17, continue_motion_max_frames=13)
    assert count == 60
    assert all(c["continue"] == 13 for c in Calls.animate[1:])
    assert [c["offset_in"] for c in Calls.animate] == [4 * k for k in range(len(Calls.animate))]
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


def test_animate1_optional_videos_reach_core(node_module):
    face = torch.zeros(81, 512, 512, 3)
    background = torch.zeros(81, 64, 32, 3)
    mask = torch.zeros(1, 64, 32)
    run(node_module, pose_frames=81, node=ANIMATE1, frames_per_chunk=77,
        clip_vision_output="clip", face_video=face, background_video=background, character_mask=mask)
    for call in Calls.animate:
        assert call["clip"] == "clip"
        assert call["face"] is face
        assert call["background"] is background
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
    """face / background / multi-frame mask whose content is the frame index, so a held frame is visible."""
    index = torch.arange(frames, dtype=torch.float32)
    return dict(face_video=index.view(-1, 1, 1, 1).expand(-1, 8, 8, 3).contiguous(),
                background_video=index.view(-1, 1, 1, 1).expand(-1, 64, 32, 3).contiguous(),
                character_mask=index.view(-1, 1, 1).expand(-1, 64, 32).contiguous())


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
    assert last["offset_in"] + last["length"] == produced
    for key in ("face", "background"):
        video = last[key]
        assert video.shape[0] == produced
        assert torch.equal(video[:frames], _videos(frames)[{"face": "face_video", "background": "background_video"}[key]])
        assert (video[frames:] == frames - 1).all()  # the last frame, held
    assert last["mask"].shape[0] == frames
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
    for call in Calls.animate:
        for key in ("face", "background"):
            assert call[key].shape[0] > call["offset_in"] + call["length"] - 1
            assert (call[key][100:] == 99).all()
        assert call["mask"].shape[0] == 100
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
        assert call["face"] is videos["face_video"]
        assert call["background"] is videos["background_video"]
        assert call["mask"] is videos["character_mask"]
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
    run(node_module, pose_frames=frames, node=ANIMATE1, frames_per_chunk=77, character_mask=character_mask, background_video=torch.zeros(frames + 80, 64, 32, 3))
    assert [c["length"] for c in Calls.animate] == [77, 77, 57]
    for call, sampled in zip(Calls.animate, Calls.sampler):
        mask = sampled["positive"][0][1]["concat_mask"]
        offset, length = call["offset_in"], call["length"]
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
            background_video=torch.zeros(frames, 64, 32, 3))
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
