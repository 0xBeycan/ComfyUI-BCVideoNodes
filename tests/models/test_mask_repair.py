"""Wan Animate's concat-mask repair against the reference implementation and core's
construction, on seeded masks. ComfyUI itself is stubbed; skipped when torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import core_concat_mask, node_module, reference_concat_mask  # noqa: E402,F401


@pytest.mark.parametrize("seed_frames", [0, 1, 5])
def test_animate1_mask_repair_matches_reference_implementation(node_module, seed_frames):
    torch.manual_seed(0)
    latent_length, lat_h, lat_w = 21, 6, 4
    frames = (latent_length - 1) * 4 + 1
    character_mask = (torch.rand(frames, lat_h, lat_w) > 0.5).float()  # changes every frame, so a 3-row shift is visible
    seed_latents = 0 if seed_frames == 0 else ((seed_frames - 1) // 4) + 1
    core = core_concat_mask(latent_length, lat_h, lat_w, seed_latents, seed_frames, character_mask)
    reference = reference_concat_mask(latent_length, lat_h, lat_w, seed_frames, character_mask)
    if seed_frames > 0:
        assert not torch.equal(core, reference)  # the core bug this repairs
        # the last seed latent has 3 of its 4 rows overwritten by the character mask
        assert core[:, 0, seed_latents].sum() == 0 and core[:, 1:, seed_latents].sum() > 0
        assert reference[:, :, 1:1 + seed_latents].sum() == 0
    rows = node_module._replacement_mask_rows(character_mask, 0, frames, seed_frames, lat_h, lat_w)
    assert rows.shape == (frames + 3, lat_h, lat_w)
    cond = [["c", {"concat_mask": core}], ["c2", {"concat_mask": core}]]
    node_module._fix_replacement_mask(cond, rows, set())
    assert torch.equal(core, reference)


def test_animate1_mask_rows_center_crop_like_core(node_module):
    # a square mask onto a 1:2 latent grid: core crops the sides off before resizing (as it does
    # the pose and background videos), so a stripe in the cropped-off column must not survive
    mask = torch.zeros(5, 8, 8)
    mask[:, :, 2] = 1.0
    rows = node_module._replacement_mask_rows(mask, 0, 5, 0, 4, 2)
    expected = torch.nn.functional.interpolate(mask[:, None, :, 2:6], size=(4, 2), mode="nearest-exact")[:, 0]
    assert torch.equal(rows[3:], expected)


def test_animate1_mask_beyond_offset_is_left_to_core(node_module):
    assert node_module._replacement_mask_rows(torch.zeros(10, 4, 4), 10, 9, 1, 4, 4) is None
    single = node_module._replacement_mask_rows(torch.ones(1, 4, 4), 500, 9, 1, 4, 4)  # one frame is repeated whatever the offset
    assert single.shape == (12, 4, 4) and single[:4].sum() == 0 and single[4:].sum() == 8 * 16
