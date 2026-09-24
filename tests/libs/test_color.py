"""The colour anchor maths of libs/color.py: sRGB <-> CIE Lab against the published Lab values of
the sRGB primaries, and the Lab mean / std transform against its formula. Runs without ComfyUI or
any model.
"""
import pytest

torch = pytest.importorskip("torch")

from bcvideonodes.libs.color import (RATIO_LIMITS, apply_transfer, feather, lab_to_srgb,  # noqa: E402
                                     lab_transfer, srgb_to_lab)


def frames(seed, count=3, low=0.25, high=0.75):
    """Seeded frames [count, 16, 12, 3] inside the sRGB gamut, away from its edges."""
    return low + (high - low) * torch.rand(count, 16, 12, 3, generator=torch.Generator().manual_seed(seed))


def lab_statistics(images, weight=None):
    """Per-channel Lab mean and std, written out: every pixel, or the pixels where weight is 1."""
    pixels = srgb_to_lab(images).reshape(-1, 3)
    if weight is not None:
        pixels = pixels[weight.reshape(-1) > 0.5]
    return pixels.mean(dim=0), ((pixels - pixels.mean(dim=0)) ** 2).mean(dim=0).sqrt()


# --- sRGB <-> Lab ------------------------------------------------------------------------------

@pytest.mark.parametrize("rgb, lab", [
    ((1.0, 1.0, 1.0), (100.0, 0.0, 0.0)),
    ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ((1.0, 0.0, 0.0), (53.24, 80.09, 67.20)),
    ((0.0, 1.0, 0.0), (87.73, -86.18, 83.18)),
    ((0.0, 0.0, 1.0), (32.30, 79.19, -107.86)),
    ((0.5, 0.5, 0.5), (53.39, 0.0, 0.0)),
])
def test_lab_of_the_srgb_primaries(rgb, lab):
    assert srgb_to_lab(torch.tensor(rgb)).tolist() == pytest.approx(lab, abs=0.01)


def test_the_lab_round_trip():
    images = torch.rand(4, 16, 12, 3, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(lab_to_srgb(srgb_to_lab(images)), images, atol=1e-5)


def test_a_lab_colour_outside_the_gamut_lands_on_its_edge():
    rgb = lab_to_srgb(torch.tensor([[50.0, 200.0, 0.0], [120.0, 0.0, 0.0]]))
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
    assert rgb[1].tolist() == pytest.approx([1.0, 1.0, 1.0])


# --- the transform -----------------------------------------------------------------------------

def test_strength_1_maps_the_overlap_statistics_onto_the_carried_ones():
    carried = frames(1)
    overlap = (carried * torch.tensor([0.9, 1.0, 1.1]) + 0.03).clamp(0, 1)  # the same frames with a colour cast
    transfer = lab_transfer(overlap, carried)
    corrected = apply_transfer(overlap, transfer, 1.0)
    for got, expected in zip(lab_statistics(corrected), lab_statistics(carried)):
        assert got.tolist() == pytest.approx(expected.tolist(), abs=1e-3)
    # the formula: (lab - source mean) * source-to-target std ratio + target mean
    source_mean, source_std = lab_statistics(overlap)
    target_mean, target_std = lab_statistics(carried)
    expected = lab_to_srgb((srgb_to_lab(overlap) - source_mean) * (target_std / source_std) + target_mean)
    assert torch.allclose(corrected, expected, atol=1e-5)


def test_strength_blends_between_the_frames_and_the_transform():
    carried = frames(2)
    overlap = (carried * 0.8 + 0.1).clamp(0, 1)
    transfer = lab_transfer(overlap, carried)
    full = apply_transfer(overlap, transfer, 1.0)
    assert torch.allclose(apply_transfer(overlap, transfer, 0.5), overlap + 0.5 * (full - overlap), atol=1e-6)
    assert torch.equal(apply_transfer(overlap, transfer, 0.0), overlap)


def test_the_std_ratio_is_clamped():
    carried = frames(3)
    flat = torch.full_like(carried, 0.5)  # no spread at all: an unclamped ratio would be infinite
    assert lab_transfer(flat, carried).ratio.tolist() == [RATIO_LIMITS[1]] * 3
    assert lab_transfer(carried, flat).ratio.tolist() == [RATIO_LIMITS[0]] * 3


def test_the_region_limits_both_the_estimate_and_the_correction():
    carried = frames(4)
    overlap = (carried * torch.tensor([1.1, 0.95, 0.9])).clamp(0, 1)
    region = torch.zeros(carried.shape[:3])
    region[:, 4:12, 3:9] = 1.0
    # outside the region the two differ wildly (a background the source re-feeds): it must not count
    overlap[region == 0] = torch.tensor([0.9, 0.1, 0.1])
    carried_background = carried.clone()
    carried_background[region == 0] = torch.tensor([0.1, 0.1, 0.9])

    transfer = lab_transfer(overlap, carried_background, region)
    inside = lab_transfer(overlap[:, 4:12, 3:9], carried[:, 4:12, 3:9])
    for got, expected in zip(transfer, inside):
        assert got.tolist() == pytest.approx(expected.tolist(), abs=1e-4)

    corrected = apply_transfer(overlap, transfer, 1.0, region)
    assert torch.equal(corrected[region == 0], overlap[region == 0])
    for got, expected in zip(lab_statistics(corrected, region), lab_statistics(carried, region)):
        assert got.tolist() == pytest.approx(expected.tolist(), abs=1e-3)


def test_an_empty_region_gives_no_transform():
    assert lab_transfer(frames(5), frames(6), torch.zeros(3, 16, 12)) is None


def test_the_frames_are_converted_in_batches_with_the_same_result():
    carried = frames(7, count=19)
    overlap = (carried + 0.05).clamp(0, 1)
    transfer = lab_transfer(overlap, carried)
    whole = lab_to_srgb((srgb_to_lab(overlap) - transfer.source_mean) * transfer.ratio + transfer.target_mean)
    assert torch.allclose(apply_transfer(overlap, transfer, 1.0), whole, atol=1e-6)


# --- the feathered region ----------------------------------------------------------------------

def test_feather_softens_only_the_edge_inside_the_region():
    region = torch.zeros(2, 400, 200)
    region[:, 100:300, 50:150] = 1.0
    soft = feather(region)  # an edge about 2% of 200 = 4 pixels wide
    assert soft.shape == region.shape and soft.min() >= 0.0 and soft.max() <= 1.0
    assert (soft[:, 110:290, 60:140] == 1.0).all()
    assert (soft[region == 0.0] == 0.0).all()  # the edge never reaches past the region
    edge = soft[0, 200, 50:60]
    assert 0.0 < edge[0] < 1.0 and edge[-1] == 1.0
    assert (edge[1:] >= edge[:-1]).all()  # rising into the region
    soft_region = 0.5 * region  # a soft region: the weights stay under it
    assert (feather(soft_region) <= soft_region).all()
    small = region[:, 90:130, 30:70]  # under 50 pixels on the shorter side: no soft edge
    assert torch.equal(feather(small), small)
