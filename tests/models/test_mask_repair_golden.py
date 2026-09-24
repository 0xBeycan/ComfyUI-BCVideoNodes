"""G10: Wan Animate's concat-mask repair, bit exact. `_replacement_mask_rows` over mask lengths,
offsets, window lengths, seed frames and latent sizes (a window the mask does not reach gives
None), and `_fix_replacement_mask` on two distinct concat masks and on one mask both
conditionings share, which the `seen` set writes only once.

The masks are seeded, 16x8 pixels, so the (8, 8) latent grid takes core's center crop. The resize
is sampler_fakes' stand-in for core's common_upscale. Recorded in the ComfyUI venv on CPU.
ComfyUI itself is stubbed; skipped when torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from golden import check, digest  # noqa: E402
from sampler_fakes import node_module  # noqa: E402,F401

MASK_FRAMES = (1, 8, 30)
OFFSETS = (0, 3, 29, 40)
LENGTHS = (5, 9)
SEED_FRAMES = (0, 1, 5)
LATENT_SIZES = ((4, 2), (8, 8))


def seeded(*shape, seed=0):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


def test_replacement_mask_rows_golden(node_module):
    for frames in MASK_FRAMES:
        mask = seeded(frames, 16, 8, seed=frames)
        for offset in OFFSETS:
            for length in LENGTHS:
                for seed_frames in SEED_FRAMES:
                    for lat_h, lat_w in LATENT_SIZES:
                        rows = node_module._replacement_mask_rows(mask, offset, length, seed_frames, lat_h, lat_w)
                        key = "rows frames={} offset={} length={} seed_frames={} latent={}x{}".format(
                            frames, offset, length, seed_frames, lat_h, lat_w)
                        check(__file__, key, digest(rows))


def conditioning(label, concat_mask):
    """Core's conditioning list with entries the repair must skip around the one it writes."""
    return [[label + " no options"], [label + " not a dict", "options"], [label + " no mask", {}],
            [label, {"concat_mask": concat_mask, "other": 1}]]


def test_fix_replacement_mask_golden(node_module):
    lat_h, lat_w, length = 4, 2, 9
    latents = (length - 1) // 4 + 1
    rows_a = node_module._replacement_mask_rows(seeded(length, 16, 8, seed=1), 0, length, 1, lat_h, lat_w)
    rows_b = node_module._replacement_mask_rows(seeded(length, 16, 8, seed=2), 0, length, 5, lat_h, lat_w)

    # two distinct tensors: each is written
    positive_mask, negative_mask = seeded(1, 4, 1 + latents, lat_h, lat_w, seed=3), seeded(1, 4, 1 + latents, lat_h, lat_w, seed=4)
    seen = set()
    node_module._fix_replacement_mask(conditioning("positive", positive_mask), rows_a, seen)
    node_module._fix_replacement_mask(conditioning("negative", negative_mask), rows_a, seen)
    check(__file__, "fix distinct", [digest(positive_mask), digest(negative_mask), len(seen)])

    # one tensor both hold: the first write wins, the second call finds it in seen
    shared = seeded(1, 4, 1 + latents, lat_h, lat_w, seed=5)
    seen = set()
    node_module._fix_replacement_mask(conditioning("positive", shared), rows_a, seen)
    node_module._fix_replacement_mask(conditioning("negative", shared), rows_b, seen)
    check(__file__, "fix shared", [digest(shared), len(seen)])
