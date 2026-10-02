"""What the chunk loop keeps alive and what it decodes: the output is one buffer of total_frames
filled chunk by chunk, every chunk's decoded frames and its conditioning are released before the
next core call, a chained chunk is seeded with the output's own frames (no copy of them), and a
last chunk that runs past total_frames decodes only the latent frames total_frames needs (the Wan
VAE decodes causally: the frames before them come out the same).

ComfyUI itself is stubbed (comfy.*, nodes); torch is real. Weak references to what the core node,
the sampler and the decoder hand back show what is still alive when the next core call starts.
Skipped when torch is not installed.
"""

import sys
import weakref

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import (ANIMATE1, ANIMATE2, SCAIL2, Calls, FakeNodeOutput, FakeSamplerCustom, FakeVAEDecode,  # noqa: E402,F401
                           IndexVAE, aligned, animate_aligned, node_module, run)

NODES = [ANIMATE1, ANIMATE2, SCAIL2]
CORE_NODE = {ANIMATE1: "WanAnimateToVideo", ANIMATE2: "WanAnimate2ToVideo", SCAIL2: "WanSCAILToVideo"}
ANCHOR = {ANIMATE1: "continue_motion", ANIMATE2: "continue_motion", SCAIL2: "previous_frames"}
# pose frames whose last_chunk full plan is four chunks of 81, the last one past the pose's end
FOUR_CHUNKS = {ANIMATE1: 240, ANIMATE2: 250, SCAIL2: 240}
SEED = {ANIMATE1: 5, ANIMATE2: 1, SCAIL2: 5}  # the anchor frames each core node keeps, at run()'s defaults


def own_buffer(tensor):
    """True when `tensor` is the whole of its storage: no larger buffer behind a view."""
    return tensor.is_contiguous() and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


# --- the output --------------------------------------------------------------------------------

@pytest.mark.parametrize("node", NODES)
def test_the_output_is_one_buffer_of_total_frames(animate_aligned, node):
    # full runs the last chunk past total: the frames past total are never part of the output's memory
    pose_frames = FOUR_CHUNKS[node]
    images, count, _ = run(animate_aligned, pose_frames=pose_frames, node=node, vae=IndexVAE(), last_chunk="full")
    assert count == pose_frames and images.shape[0] == pose_frames and own_buffer(images)
    assert images[:, 0, 0, 0].tolist() == [float(i) for i in range(pose_frames)]


@pytest.mark.parametrize("pose_frames, total, decoded", [(500, 50, 53), (81, 0, 81)])
@pytest.mark.parametrize("node", NODES)
def test_a_single_chunk_s_frames_are_the_output(animate_aligned, node, pose_frames, total, decoded):
    # one chunk: its decoded frames are the output, not copied; a last chunk that runs past
    # total_frames decodes only up to the 4k+1 frame grid past it (here 53 frames for 50)
    images, count, _ = run(animate_aligned, pose_frames=pose_frames, total_frames=total, node=node, vae=IndexVAE(), last_chunk="full")
    total = total or pose_frames
    assert count == total and images.shape[0] == total and images.is_contiguous()
    assert images.untyped_storage().nbytes() == decoded * images[0].numel() * images.element_size()
    assert images[:, 0, 0, 0].tolist() == [float(i) for i in range(total)]


# --- what is alive when the next core call starts ----------------------------------------------

@pytest.fixture
def alive(node_module, monkeypatch):
    """(the sampler Names, check) with weak references to every chunk's conditioning, sampled
    latent and decoded frames; check() asserts, at the start of each core call, that those of
    the chunks before are gone. The sampler keeps no reference of its own."""
    refs = []
    checked = []
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS

    def check():
        checked.append([name for name, ref in refs if ref() is not None])

    for node in CORE_NODE.values():
        core = mappings[node]

        def execute(cls, _core=core, **kwargs):
            check()
            output = _core.EXECUTE_NORMALIZED(**kwargs)
            positive, negative, *rest = output.args
            payload = torch.zeros(4)  # stands in for the conditioning's tensors
            refs.append(("conditioning", weakref.ref(payload)))
            positive = [[c[0], {**c[1], "payload": payload}] for c in positive]
            negative = [[c[0], {**c[1], "payload": payload}] for c in negative]
            return FakeNodeOutput(positive, negative, *rest)

        monkeypatch.setitem(mappings, node, type(core.__name__, (core,), {"EXECUTE_NORMALIZED": classmethod(execute)}))

    class Sampler(FakeSamplerCustom):
        @classmethod
        def EXECUTE_NORMALIZED(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
            samples = latent_image["samples"].clone()
            refs.append(("latent", weakref.ref(samples)))
            return FakeNodeOutput(dict(latent_image, samples=samples), None)

    class Decode(FakeVAEDecode):
        def decode(self, vae, samples):
            images = super().decode(vae, samples)[0]
            refs.append(("decoded", weakref.ref(images)))
            return (images,)

    monkeypatch.setitem(mappings, "SamplerCustom", Sampler)
    monkeypatch.setitem(mappings, "VAEDecode", Decode)
    return node_module, checked


@pytest.mark.parametrize("node", NODES)
def test_a_chunk_s_conditioning_latent_and_decode_are_gone_before_the_next_core_call(alive, node):
    module, checked = alive
    run(module, pose_frames=FOUR_CHUNKS[node], node=node, last_chunk="full", **(dict(clip_vision="cv") if node == ANIMATE2 else {}))
    assert len(checked) == 4 and checked == [[]] * 4


@pytest.mark.parametrize("node", NODES)
def test_what_the_core_call_leaves_in_a_reference_cycle_is_collected_before_sampling(node_module, monkeypatch, node):
    # core's WanAnimateToVideo leaves the VAE encoder's features in a reference cycle, which only a
    # collection frees; with Python's automatic collector off, only the loop's own collection can
    import gc

    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    core = mappings[CORE_NODE[node]]
    features, at_sampling = [], []

    def execute(cls, **kwargs):
        cycle = {"features": torch.zeros(4)}
        cycle["self"] = cycle
        features.append(weakref.ref(cycle["features"]))
        return core.EXECUTE_NORMALIZED(**kwargs)

    class Sampler(FakeSamplerCustom):
        @classmethod
        def EXECUTE_NORMALIZED(cls, **kwargs):
            at_sampling.append([ref() is None for ref in features])
            return super().EXECUTE_NORMALIZED(**kwargs)

    monkeypatch.setitem(mappings, CORE_NODE[node], type(core.__name__, (core,), {"EXECUTE_NORMALIZED": classmethod(execute)}))
    monkeypatch.setitem(mappings, "SamplerCustom", Sampler)
    enabled = gc.isenabled()
    gc.disable()
    try:
        run(node_module, pose_frames=FOUR_CHUNKS[node], node=node, last_chunk="full",
            **(dict(clip_vision="cv") if node == ANIMATE2 else {}))
    finally:
        if enabled:
            gc.enable()
    assert at_sampling == [[True] * chunk for chunk in range(1, 5)]


# --- the seed ----------------------------------------------------------------------------------

@pytest.mark.parametrize("node", NODES)
def test_a_chained_chunk_is_seeded_with_the_output_s_own_frames(animate_aligned, monkeypatch, node):
    seeds = []
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    core = mappings[CORE_NODE[node]]

    def execute(cls, **kwargs):
        seeds.append(kwargs.get(ANCHOR[node]))
        return core.EXECUTE_NORMALIZED(**kwargs)

    monkeypatch.setitem(mappings, CORE_NODE[node], type(core.__name__, (core,), {"EXECUTE_NORMALIZED": classmethod(execute)}))
    images, _, _ = run(animate_aligned, pose_frames=FOUR_CHUNKS[node], node=node, vae=IndexVAE(), last_chunk="full")
    assert seeds[0] is None and len(seeds) == 4
    for seed in seeds[1:]:
        assert seed.untyped_storage().data_ptr() == images.untyped_storage().data_ptr()  # a view, not a copy
        last = int(seed[-1, 0, 0, 0])
        # the output's last frames before the chunk, just the ones the core node keeps
        assert torch.equal(seed, images[last + 1 - SEED[node]:last + 1])


# --- the last chunk decodes only what total_frames needs ----------------------------------------

@pytest.mark.parametrize("node, pose_frames, total, policy, latents, plan", [
    # 233 frames made before the last chunk, 7 more needed after its 5 seed frames: 13 frames, 4 latent frames
    (SCAIL2, 240, 0, "full", [21, 21, 21, 4], "81 + 81 + 81 + 81 -> 309 produced -> 240 frames (pose 240, overlap 5)"),
    (SCAIL2, 240, 0, "min29", [21, 21, 21, 4], "81 + 81 + 81 + 29 -> 257 produced -> 240 frames (pose 240, overlap 5)"),
    (SCAIL2, 240, 0, "fit", [21, 21, 21, 4], "81 + 81 + 81 + 13 -> 241 produced -> 240 frames (pose 240, overlap 5)"),
    (ANIMATE1, 240, 0, "full", [21, 21, 21, 4], "81 + 81 + 81 + 81 -> 309 produced -> 240 frames (pose 240, overlap 5)"),
    # 241 made, 9 more after 1 seed frame: 10 frames, on the 4k+1 grid 13, 4 latent frames
    (ANIMATE2, 250, 0, "full", [21, 21, 21, 4], "81 + 81 + 81 + 81 -> 321 produced -> 250 frames (pose 250, overlap 1)"),
    # one chunk for a short total_frames: 50 frames, on the grid 53, 14 latent frames
    (SCAIL2, 500, 50, "full", [14], "81 -> 81 produced -> 50 frames (pose 500, overlap 5)"),
])
def test_the_last_chunk_decodes_only_what_total_frames_needs(animate_aligned, monkeypatch, node, pose_frames, total, policy, latents, plan):
    decoded = []

    class Decode(FakeVAEDecode):
        def decode(self, vae, samples):
            decoded.append(samples["samples"].shape[2])
            return super().decode(vae, samples)

    monkeypatch.setitem(sys.modules["nodes"].NODE_CLASS_MAPPINGS, "VAEDecode", Decode)
    images, count, found = run(animate_aligned, pose_frames=pose_frames, total_frames=total, node=node, vae=IndexVAE(), last_chunk=policy)
    total = total or pose_frames
    assert decoded == latents
    assert found == plan and count == total
    assert images[:, 0, 0, 0].tolist() == [float(i) for i in range(total)]
