"""Wan Animate 2's seed-frame attention bias on stub attention calls. ComfyUI itself is stubbed;
skipped when torch is not installed.
"""

import sys
import types

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import node_module  # noqa: E402,F401


def test_seed_frame_attention_bias_targets_generation_attention_only(node_module, monkeypatch):
    seen = []
    attention = types.ModuleType("comfy.ldm.modules.attention")

    def attention_pytorch(q, k, v, heads, mask=None, **kwargs):
        seen.append(mask.clone())
        return "biased"

    attention.attention_pytorch = attention_pytorch
    ldm = types.ModuleType("comfy.ldm"); modules = types.ModuleType("comfy.ldm.modules")
    for name, module in {"comfy.ldm": ldm, "comfy.ldm.modules": modules, "comfy.ldm.modules.attention": attention}.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["comfy"].ldm = ldm; ldm.modules = modules; modules.attention = attention

    override = node_module._seed_frame_attention_bias(-1.3)
    frames, gh, gw = 4, 3, 2
    hw, tokens = gh * gw, 4 * gh * gw
    options = {"block_type": "double", "grid_sizes": (frames, gh, gw)}
    func = lambda q, k, v, **kw: "plain"
    q = torch.zeros(1, hw, 8, dtype=torch.float16)
    # per-frame generation call: q = one frame, k = all gen tokens + that frame's pose tokens
    assert override(func, q, torch.zeros(1, tokens + hw, 8), torch.zeros(1, tokens + hw, 8), heads=2, transformer_options=options) == "biased"
    mask = seen[-1]
    assert mask.shape == (1, 1, 1, tokens + hw) and mask.dtype == torch.float16
    assert torch.allclose(mask[..., hw:2 * hw].float(), torch.full((hw,), -1.3), atol=1e-3)
    assert (mask[..., :hw] == 0).all() and (mask[..., 2 * hw:] == 0).all()
    # frame 0's call has no pose tail
    assert override(func, q, torch.zeros(1, tokens, 8), torch.zeros(1, tokens, 8), heads=2, transformer_options=options) == "biased"
    # whole-clip self-attention when the pose branch is windowed out
    assert override(func, torch.zeros(1, tokens, 8), torch.zeros(1, tokens, 8), torch.zeros(1, tokens, 8), heads=2, transformer_options=options) == "biased"
    # cross-attention (769 text + image tokens), the pose branch ((f-1)*hw), and blocks without grid info pass through
    assert override(func, torch.zeros(1, tokens, 8), torch.zeros(1, 769, 8), torch.zeros(1, 769, 8), heads=2, transformer_options=options) == "plain"
    assert override(func, torch.zeros(1, 3 * hw, 8), torch.zeros(1, 3 * hw, 8), torch.zeros(1, 3 * hw, 8), heads=2, transformer_options=options) == "plain"
    assert override(func, q, torch.zeros(1, tokens, 8), torch.zeros(1, tokens, 8), heads=2, transformer_options={}) == "plain"
    assert len(seen) == 3
