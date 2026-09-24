"""The colour anchor of the long-video samplers (their color_anchor_strength widget): sRGB <-> CIE
Lab (D65), and the per-channel Lab mean / std transform (Reinhard et al. 2001, "Color Transfer
between Images") that maps a chunk's regenerated overlap frames onto the frames it was seeded with.
Frames are IMAGE batches [frames, height, width, 3] in 0..1, regions [frames, height, width]
weights in 0..1."""

from typing import NamedTuple

# linear sRGB -> CIE XYZ (IEC 61966-2-1) and the D65 white point
RGB_TO_XYZ = ((0.4124564, 0.3575761, 0.1804375),
              (0.2126729, 0.7151522, 0.0721750),
              (0.0193339, 0.1191920, 0.9503041))
D65 = (0.95047, 1.0, 1.08883)
DELTA = 6.0 / 29.0  # where CIE Lab's cube root turns linear
RATIO_LIMITS = (0.5, 2.0)  # the std ratio a transform may scale by, so a flat frame cannot blow a chunk up
FEATHER = 0.02  # the width of a region's soft edge, as a share of the frame's shorter side
BATCH = 8  # frames converted at once, so a full-resolution chunk is never in Lab as a whole


class LabTransfer(NamedTuple):
    """Maps a Lab colour ``lab`` to ``(lab - source_mean) * ratio + target_mean``: the source's
    per-channel mean and std onto the target's. Each field is a [3] tensor (L, a, b)."""

    source_mean: object
    target_mean: object
    ratio: object  # target std / source std, clamped to RATIO_LIMITS


def srgb_to_lab(rgb):
    """sRGB [..., 3] in 0..1 as CIE Lab [..., 3] (D65, float32): L in 0..100, a and b about -128..127."""
    import torch

    rgb = rgb.float()
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)
    xyz = linear @ torch.tensor(RGB_TO_XYZ, device=rgb.device).T / torch.tensor(D65, device=rgb.device)
    f = torch.where(xyz > DELTA ** 3, xyz.clamp(min=DELTA ** 3) ** (1.0 / 3.0), xyz / (3 * DELTA ** 2) + 4.0 / 29.0)
    fx, fy, fz = f.unbind(-1)
    return torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)), dim=-1)


def lab_to_srgb(lab):
    """CIE Lab [..., 3] (D65) as sRGB [..., 3] (float32), clipped to 0..1: a colour outside the sRGB
    gamut lands on its edge."""
    import torch

    lab = lab.float()
    L, a, b = lab.unbind(-1)
    fy = (L + 16.0) / 116.0
    f = torch.stack((fy + a / 500.0, fy, fy - b / 200.0), dim=-1)
    xyz = torch.where(f > DELTA, f ** 3, 3 * DELTA ** 2 * (f - 4.0 / 29.0)) * torch.tensor(D65, device=lab.device)
    xyz_to_rgb = torch.linalg.inv(torch.tensor(RGB_TO_XYZ, dtype=torch.float64)).to(torch.float32).to(lab.device)
    linear = (xyz @ xyz_to_rgb.T).clamp(0.0, 1.0)
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1.0 / 2.4) - 0.055)


def feather(region):
    """``region`` [frames, H, W] with a soft edge about FEATHER of the frame's shorter side wide,
    inside the region: area-averaged down by that factor, bilinearly back up, and never above the
    region itself, so a pixel outside it (a background core re-feeds) gets no weight."""
    import torch
    import torch.nn.functional as F

    region = region.float()
    height, width = region.shape[-2:]
    step = max(1, round(min(height, width) * FEATHER))
    coarse = F.interpolate(region[:, None], size=(-(-height // step), -(-width // step)), mode="area")
    return torch.minimum(F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)[:, 0], region)


def _statistics(lab, weight):
    """Per-channel mean and std of Lab pixels [..., 3], each weighted by ``weight`` [...] (None: 1)."""
    import torch

    pixels = lab.reshape(-1, 3)
    weight = torch.ones_like(pixels[:, :1]) if weight is None else weight.reshape(-1, 1).to(pixels)
    total = weight.sum()
    mean = (pixels * weight).sum(dim=0) / total
    return mean, (((pixels - mean) ** 2 * weight).sum(dim=0) / total).sqrt()


def lab_transfer(source, target, weight=None):
    """The LabTransfer that maps the Lab mean and std of the frames ``source`` onto those of the
    frames ``target``: the same content, so one ``weight`` [frames, H, W] (None: every pixel)
    weights the pixels of both. None when ``weight`` is 0 everywhere: nothing to estimate from."""
    if weight is not None and float(weight.sum()) <= 0.0:
        return None
    source_mean, source_std = _statistics(srgb_to_lab(source), weight)
    target_mean, target_std = _statistics(srgb_to_lab(target), weight)
    ratio = (target_std / source_std.clamp(min=1e-6)).clamp(*RATIO_LIMITS)
    return LabTransfer(source_mean, target_mean, ratio)


def apply_transfer(images, transfer, strength, weight=None):
    """``images`` [frames, H, W, 3] moved toward the colours ``transfer`` maps them to by
    ``strength`` (0..1): ``x + strength * weight * (T(x) - x)``, clipped to 0..1, with ``weight``
    [frames, H, W] (None: 1) limiting it to a region. Returns a new tensor of the images' dtype."""
    import torch

    out = torch.empty_like(images)
    source_mean, target_mean, ratio = (value.to(images.device) for value in transfer)
    for start in range(0, images.shape[0], BATCH):
        x = images[start:start + BATCH].float()
        mapped = lab_to_srgb((srgb_to_lab(x) - source_mean) * ratio + target_mean)
        scale = strength if weight is None else strength * weight[start:start + BATCH, ..., None].to(x)
        out[start:start + BATCH] = (x + scale * (mapped - x)).clamp(0.0, 1.0).to(images.dtype)
    return out
