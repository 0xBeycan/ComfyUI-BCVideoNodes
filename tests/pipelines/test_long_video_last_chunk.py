"""The last_chunk widget of the three long-video samplers: fit shortens the last chunk to the
frames still needed, full runs it at frames_per_chunk with the driving inputs held, min29 is fit
with a last chunk of at least 29 frames; either way the output is exactly total_frames.

ComfyUI itself is stubbed (comfy.*, nodes); torch is real. For the end-to-end alignment the
`aligned` fixture and IndexVAE (sampler_fakes) stand in for the model; the Animate fakes are
wrapped so they hand the pose window they read to the sampler as SCAIL-2's does. With run()'s
frame-index pose video, output frame i must then show driving frame i. Skipped when torch is not
installed.
"""

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (ANIMATE1, ANIMATE2, SCAIL2, Calls, IndexVAE, aligned, animate_aligned,  # noqa: E402,F401
                           driving_videos, node_module, run)

NODES = [ANIMATE1, ANIMATE2, SCAIL2]
POLICIES = ["fit", "full", "min29"]
OVERLAP = {ANIMATE1: 5, ANIMATE2: 1, SCAIL2: 5}


def test_the_defaults(node_module):
    assert [getattr(node_module, node).DEFAULT_LAST_CHUNK for node in NODES] == ["fit", "fit", "full"]


# --- output frame i is driving frame i, under every policy ------------------------------------

@pytest.mark.parametrize("total", [81, 157, 240, 301, 450])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("node", NODES)
def test_output_frame_i_shows_driving_frame_i(animate_aligned, node, policy, total):
    images, count, plan = run(animate_aligned, pose_frames=total, node=node, vae=IndexVAE(), last_chunk=policy)
    assert count == total and images.shape[0] == total
    assert images[:, 0, 0, 0].tolist() == [float(i) for i in range(total)]
    lengths = [c["length"] for c in Calls.animate]
    assert plan.startswith(" + ".join(str(length) for length in lengths) + " -> ")
    assert lengths[:-1] == [81] * (len(lengths) - 1)
    if policy == "full":
        assert lengths[-1] == 81
    else:  # the need left before the last chunk, plus its overlap, snapped up to 4k+1 (min29: at least 29)
        produced = 81 + (81 - OVERLAP[node]) * (len(lengths) - 2) if len(lengths) > 1 else 0
        need = total - produced + (OVERLAP[node] if produced else 0)
        fit = min(81, -(-(need - 1) // 4) * 4 + 1)
        assert lengths[-1] == (fit if policy == "fit" else max(fit, 29))


# --- the plans ---------------------------------------------------------------------------------

@pytest.mark.parametrize("node, total, policy, plan", [
    (ANIMATE1, 240, "fit", "81 + 81 + 81 + 13 -> 241 produced -> 240 frames (pose 240, overlap 5)"),
    (ANIMATE1, 240, "full", "81 + 81 + 81 + 81 -> 309 produced -> 240 frames (pose 240, overlap 5)"),
    (ANIMATE2, 250, "fit", "81 + 81 + 81 + 13 -> 253 produced -> 250 frames (pose 250, overlap 1)"),
    (ANIMATE2, 250, "full", "81 + 81 + 81 + 81 -> 321 produced -> 250 frames (pose 250, overlap 1)"),
    (SCAIL2, 240, "fit", "81 + 81 + 81 + 13 -> 241 produced -> 240 frames (pose 240, overlap 5)"),
    (SCAIL2, 240, "full", "81 + 81 + 81 + 81 -> 309 produced -> 240 frames (pose 240, overlap 5)"),
    (ANIMATE1, 240, "min29", "81 + 81 + 81 + 29 -> 257 produced -> 240 frames (pose 240, overlap 5)"),
    (ANIMATE2, 888, "min29", " + ".join(["81"] * 11) + " + 29 -> 909 produced -> 888 frames (pose 888, overlap 1)"),
    (SCAIL2, 240, "min29", "81 + 81 + 81 + 29 -> 257 produced -> 240 frames (pose 240, overlap 5)"),
])
def test_the_plan_follows_last_chunk(node_module, caplog, node, total, policy, plan):
    caplog.set_level("INFO")
    images, count, found = run(node_module, pose_frames=total, node=node, last_chunk=policy)
    assert found == plan and count == total
    assert [c["length"] for c in Calls.animate] == [int(length) for length in plan.split(" -> ")[0].split(" + ")]
    held = [line for line in caplog.text.splitlines() if "held" in line]
    reason = {"fit": "the last chunk is snapped up to 4k+1", "full": "the last chunk runs the full frames_per_chunk",
              "min29": "the last chunk is snapped up to 4k+1, at least 29 frames"}[policy]
    produced = int(plan.split(" -> ")[1].split()[0])
    assert len(held) == 1 and "{} and runs {} frames past total_frames".format(reason, produced - total) in held[0]


@pytest.mark.parametrize("policy", POLICIES)
def test_the_scail2_full_length_line_is_logged_only_with_full(node_module, caplog, policy):
    caplog.set_level("INFO")
    run(node_module, pose_frames=240, node=SCAIL2, last_chunk=policy)
    assert ("every chunk runs the full 81 frames" in caplog.text) == (policy == "full")


# --- full: the last chunk runs frames_per_chunk on driving inputs held to its end --------------

def test_full_holds_pose_face_and_background_for_animate(animate_aligned):
    images, count, plan = run(animate_aligned, pose_frames=240, node=ANIMATE1, vae=IndexVAE(), last_chunk="full",
                              **driving_videos(240))
    assert count == 240 and images[:, 0, 0, 0].tolist() == [float(i) for i in range(240)]
    last = Calls.animate[-1]
    assert last["length"] == 81 and last["offset_in"] + last["length"] == 309
    assert last["pose_in"] == 309
    for key in ("face", "background"):
        assert last[key].shape[0] == 309
        assert (last[key][240:] == 239).all()  # the last frame, held


def test_full_holds_the_pose_for_animate2(animate_aligned):
    run(animate_aligned, pose_frames=250, node=ANIMATE2, vae=IndexVAE(), last_chunk="full")
    last = Calls.animate[-1]
    assert last["length"] == 81 and last["offset_in"] + last["length"] == 321
    assert last["pose_in"] == 321


def test_full_holds_pose_and_pose_mask_for_scail2(node_module):
    run(node_module, pose_frames=240, node=SCAIL2, last_chunk="full")
    last = Calls.animate[-1]
    assert last["length"] == 81 and last["pose_in"] == 309
    assert last["pose_frames"] == last["mask_frames"] == 81  # a full window of both, held past frame 239
    assert last["offset_in"] == 228 and last["pose"] == last["mask"] == 228.0


def test_fit_runs_a_short_last_chunk_for_scail2(node_module):
    run(node_module, pose_frames=240, node=SCAIL2, last_chunk="fit")
    last = Calls.animate[-1]
    assert last["length"] == 13 and last["pose_in"] == 241
    assert last["pose_frames"] == last["mask_frames"] == 13


# --- an unknown value --------------------------------------------------------------------------

@pytest.mark.parametrize("node", NODES)
def test_an_unknown_last_chunk_is_an_error(node_module, node):
    with pytest.raises(ValueError, match=r"last_chunk must be one of fit, full, min29; found 'fitted'"):
        run(node_module, pose_frames=240, node=node, last_chunk="fitted")
    assert Calls.animate == []
