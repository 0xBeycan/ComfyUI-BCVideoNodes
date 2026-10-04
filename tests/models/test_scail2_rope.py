"""Official SCAIL-2's pose RoPE (zai-org/SCAIL-2 wan/modules/model_scail2.py rope_params,
rope_apply_pose and the shifts of WanModel.forward, wan-scail2 branch) against the replacement of
core's SCAILWanModel.rope_encode the SCAIL-2 adapter installs: the pose tokens' rotary values are
official's, written out in the test in float64; the other tokens' are core's, bit for bit; and
the object patch leaves the model as core built it once its clone is unloaded. Runs on core's own
model class (tiny, on the meta device: rope_encode reads no weight) where ComfyUI is importable
(with the ComfyUI root on PYTHONPATH).
"""

import pytest

torch = pytest.importorskip("torch")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
wan = pytest.importorskip("comfy.ldm.wan.model")

import comfy.model_patcher  # noqa: E402
import comfy.ops  # noqa: E402

from scail2_fakes import scail2  # noqa: E402

HEAD_DIM = 128  # the 14B model's dim / num_heads, which sets the three rotary axes' sizes
FRAMES, H, W = 3, 8, 24  # latent video frames and size: 4 x 12 video tokens, the pose's 2 x 6; widths 120..131 cross 128
MODES = {"animation": True, "replacement": False, "no flag": None}  # core's ref_mask_flag per mode


@pytest.fixture(scope="module")
def model():
    return wan.SCAIL2WanModel(dim=HEAD_DIM, ffn_dim=32, freq_dim=16, text_dim=8, num_heads=1, num_layers=1,
                              operations=comfy.ops.disable_weight_init, device="meta")


def rope_params(max_seq_len, dim, theta=10000):
    """model_scail2.py rope_params: the complex rotary values of positions 0 .. max_seq_len - 1."""
    freqs = torch.outer(torch.arange(max_seq_len), 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return torch.polar(torch.ones_like(freqs), freqs)


def official_pose(ref_mask_flag):
    """rope_apply_pose's rotary values for the pose tokens, [tokens, HEAD_DIM / 2] complex: the
    full grid's values (T from the video's shift, H from 0, W from 120) averaged over 2 x 2
    blocks, real and imaginary parts apart (avg_pool2d)."""
    d, c = HEAD_DIM, HEAD_DIM // 2
    freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))], dim=1)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    f, h, w = FRAMES, H // 2, W // 2
    shift_f = 0 if ref_mask_flag is False else 1  # base_video_shift: 0 in replacement mode, else 1 (one reference)
    full = torch.cat([freqs[0][shift_f:shift_f + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                      freqs[1][0:h].view(1, h, 1, -1).expand(f, h, w, -1),
                      freqs[2][120:120 + w].view(1, 1, w, -1).expand(f, h, w, -1)], dim=-1)
    pool = lambda part: torch.nn.functional.avg_pool2d(part.permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)
    return torch.complex(pool(full.real), pool(full.imag)).reshape(-1, c)


def as_complex(freqs):
    """Core's rotary values [1, tokens, 1, HEAD_DIM / 2, 2, 2], each the matrix [[cos, -sin],
    [sin, cos]], as complex numbers cos + i sin."""
    return torch.complex(freqs[0, :, 0, :, 0, 0].double(), freqs[0, :, 0, :, 1, 0].double())


def encode(rope_encode, ref_mask_flag, dtype):
    """``rope_encode`` called as core's SCAILWanModel._forward calls it, one reference frame."""
    return rope_encode(FRAMES + 1, H, W, device="cpu", dtype=dtype, transformer_options={},
                       pose_latents=torch.zeros(1, 16, FRAMES, H // 2, W // 2), reference_latent=torch.zeros(1, 16, 1, H, W),
                       ref_mask_flag=ref_mask_flag)


VIDEO_TOKENS = (FRAMES + 1) * (H // 2) * (W // 2)  # the reference's and the video's


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", MODES)
def test_the_pose_tokens_get_official_rotary_values(model, mode, dtype):
    out = encode(scail2.official_pose_rope(model.rope_encode), MODES[mode], dtype)
    pose = as_complex(out[:, VIDEO_TOKENS:])
    assert out.shape[1] == VIDEO_TOKENS + FRAMES * (H // 4) * (W // 4)
    assert torch.allclose(pose, official_pose(MODES[mode]), atol=1e-6, rtol=0)
    assert torch.equal(out[..., 0, 1], -out[..., 1, 0]) and torch.equal(out[..., 0, 0], out[..., 1, 1])  # rotation matrices


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", MODES)
def test_the_other_tokens_keep_core_s_rotary_values(model, mode, dtype):
    out = encode(scail2.official_pose_rope(model.rope_encode), MODES[mode], dtype)
    core = encode(model.rope_encode, MODES[mode], dtype)
    assert out.dtype == core.dtype and torch.equal(out[:, :VIDEO_TOKENS], core[:, :VIDEO_TOKENS])


def test_core_s_pose_rope_is_what_the_patch_replaces(model):
    """Core's own pose values: one unit-magnitude rotation at the block midpoint, and in bf16 the
    width positions from 128 on rounded; official's magnitude is cos(omega / 2) per frequency."""
    official = official_pose(True)
    fp32 = as_complex(encode(model.rope_encode, True, torch.float32)[:, VIDEO_TOKENS:])
    bf16 = as_complex(encode(model.rope_encode, True, torch.bfloat16)[:, VIDEO_TOKENS:])
    assert torch.allclose(fp32.abs(), torch.ones_like(fp32.abs()), atol=1e-6)
    assert torch.allclose(fp32 / fp32.abs(), official / official.abs(), atol=1e-5, rtol=0)  # the same phase
    assert official.abs().min() < 0.88  # the highest height / width frequency: cos(1 / 2) = 0.8776
    assert not torch.allclose(bf16, fp32, atol=1e-3, rtol=0)


def test_pose_tokens_not_half_the_video_tokens_are_an_error(model):
    rope_encode = scail2.official_pose_rope(model.rope_encode)
    with pytest.raises(RuntimeError, match="SCAIL-2 pose tokens 6x4 are not half the video tokens 12x4"):
        rope_encode(FRAMES + 1, H, W, device="cpu", dtype=torch.float32, transformer_options={},
                    pose_latents=torch.zeros(1, 16, FRAMES, H, W // 2), reference_latent=torch.zeros(1, 16, 1, H, W), ref_mask_flag=True)


class Holder(torch.nn.Module):
    """The BaseModel a ModelPatcher holds, as far as an object patch reaches: its diffusion_model."""

    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion_model = diffusion_model


def test_the_object_patch_holds_only_while_its_clone_is_patched(model):
    base = comfy.model_patcher.ModelPatcher(Holder(model), load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
    clone = base.clone()
    clone.add_object_patch("diffusion_model.rope_encode", scail2.official_pose_rope(clone.get_model_object("diffusion_model.rope_encode")))
    core = encode(model.rope_encode, True, torch.bfloat16)
    assert base.object_patches == {}

    clone.patch_model(load_weights=False)
    patched = encode(model.rope_encode, True, torch.bfloat16)
    assert torch.allclose(as_complex(patched[:, VIDEO_TOKENS:]), official_pose(True), atol=1e-6, rtol=0)
    clone.unpatch_model(unpatch_weights=False)

    assert torch.equal(encode(model.rope_encode, True, torch.bfloat16), core)  # the model as core built it
    assert wan.SCAIL2WanModel.rope_encode is wan.SCAILWanModel.rope_encode  # the class never touched
