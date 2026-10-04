"""Official SCAIL-2's RoPE for the half-resolution pose tokens, as a replacement of core's
SCAILWanModel.rope_encode that the adapter installs as an object patch on its model clone."""

import torch

POSE_DOWN = 2  # the pose tokens' grid is half the video tokens' grid in height and width


def official_pose_rope(rope_encode):
    """``rope_encode`` (core's SCAILWanModel.rope_encode of the model being patched) with the
    pose tokens' rotary values as official SCAIL-2 builds them (zai-org/SCAIL-2
    wan/modules/model_scail2.py rope_apply_pose, wan-scail2 branch): the rotary values of the
    full-resolution grid (the video tokens' height and width positions, the width shifted by 120),
    each 2 x 2 block of them averaged as complex numbers, which puts the phase at the block's
    midpoint and scales the magnitude by cos(omega / 2) per frequency. Core takes one unit-magnitude
    rotation at the midpoint instead, and builds the positions in the inference dtype, so in bf16
    the pose width positions from 128 on round to whole numbers.

    The full grid is core's own pose path handed a full-resolution pose shape (scale 1, shift 0;
    its temporal start and width shift per mode stay core's), with the positions in float32. The
    other tokens' rotary values are core's call without the pose, untouched."""

    def official(t, h, w, *args, pose_latents=None, dtype=None, **kwargs):
        main = rope_encode(t, h, w, *args, dtype=dtype, **kwargs)
        if pose_latents is None:
            return main
        frames, pose_h, pose_w = (int(n) for n in pose_latents.shape[-3:])
        # the video tokens' grid (patches of 2), twice the pose tokens', as official asserts
        grid_h, grid_w = (h + 1) // 2, (w + 1) // 2
        if grid_h != POSE_DOWN * ((pose_h + 1) // 2) or grid_w != POSE_DOWN * ((pose_w + 1) // 2):
            raise RuntimeError("SCAIL-2 pose tokens {}x{} are not half the video tokens {}x{}: the pose video must be "
                               "encoded at half the generation size, and the generation size divisible by 32.".format(
                                   (pose_w + 1) // 2, (pose_h + 1) // 2, grid_w, grid_h))
        full = torch.empty(frames, h, w, device="meta")
        full = rope_encode(t, h, w, *args, pose_latents=full, dtype=torch.float32, **kwargs)
        pose = full[:, -frames * grid_h * grid_w:]
        rest = pose.shape[3:]
        pose = pose.reshape(frames, grid_h // POSE_DOWN, POSE_DOWN, grid_w // POSE_DOWN, POSE_DOWN, *rest)
        # a rotation matrix [[cos, -sin], [sin, cos]] is the complex number cos + i sin, so the mean
        # of the matrices is the mean of the complex values (official: avg_pool2d of real and imag)
        pose = pose.mean(dim=(2, 4)).reshape(1, -1, 1, *rest)
        return torch.cat((main, pose.to(main.dtype)), dim=1)

    return official

