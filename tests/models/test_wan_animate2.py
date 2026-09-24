"""Wan Animate 2's seed-frame attention bias on stub attention calls, chained after an override
installed before it, and the adapter that installs it and logs its count per chunk. ComfyUI
itself is stubbed; skipped when torch is not installed.
"""

import inspect
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import ANIMATE2, FakeModel, node_module  # noqa: E402,F401

FRAMES, GH, GW = 4, 3, 2
HW, TOKENS = GH * GW, FRAMES * GH * GW
TEXT = 769  # cross-attention keys: text + image tokens
HEADS = 2
OPTIONS = {"block_type": "double", "grid_sizes": (FRAMES, GH, GW)}
SLOT = "optimized_attention_override"


def sdpa(q, k, v, heads, mask=None):
    """Multi-head scaled dot-product attention over (batch, tokens, heads * dim), in float32."""
    b, _, dim = q.shape
    split = lambda t: t.float().view(b, -1, heads, dim // heads).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(split(q), split(k), split(v),
                                                           attn_mask=None if mask is None else mask.float())
    return out.transpose(1, 2).reshape(b, -1, dim)


def seed_frame_bias(lk):
    """The bias written out: -1.3 on the keys of latent frame 1, 0 on every other key."""
    bias = torch.zeros(1, 1, 1, lk)
    bias[..., HW:2 * HW] = -1.3
    return bias


def qkv(lq, lk):
    return torch.randn(1, lq, 8), torch.randn(1, lk, 8), torch.randn(1, lk, 8)


def core(q, k, v, heads, **kwargs):
    """Core's attention function, the `func` an override is handed."""
    return sdpa(q, k, v, heads)


def recording(result):
    """An override that records each call it gets and returns `result`."""
    def override(func, q, k, v, **kwargs):
        override.calls.append((func, q, k, v, kwargs))
        return result
    override.calls = []
    return override


def prepared(node_module, log_scale):
    """The Wan Animate 2 adapter after prepare, with attn_log_scale `log_scale`."""
    adapter = node_module.WanAnimate2Adapter(ANIMATE2, "fit")
    inputs = dict(pose_start_percent=0.0, pose_end_percent=1.0, attn_log_scale=log_scale)
    adapter.prepare(type("Core", (), {}), inputs, None, 64, 64, 81)
    return adapter


@pytest.fixture
def masks(node_module, monkeypatch):
    """comfy.ldm.modules.attention stubbed: attention_pytorch runs sdpa and records the mask of every call."""
    torch.manual_seed(0)
    seen = []
    attention = types.ModuleType("comfy.ldm.modules.attention")

    def attention_pytorch(q, k, v, heads, mask=None, **kwargs):
        seen.append(mask.clone())
        return sdpa(q, k, v, heads, mask)

    attention.attention_pytorch = attention_pytorch
    ldm = types.ModuleType("comfy.ldm"); modules = types.ModuleType("comfy.ldm.modules")
    for name, module in {"comfy.ldm": ldm, "comfy.ldm.modules": modules, "comfy.ldm.modules.attention": attention}.items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["comfy"].ldm = ldm; ldm.modules = modules; modules.attention = attention
    return seen


def test_seed_frame_attention_bias_targets_generation_attention_only(node_module, masks):
    override = node_module._seed_frame_attention_bias(-1.3)
    func = lambda q, k, v, **kw: "plain"
    q = torch.zeros(1, HW, 8, dtype=torch.float16)
    # per-frame generation call: q = one frame, k = all gen tokens + that frame's pose tokens
    override(func, q, torch.zeros(1, TOKENS + HW, 8), torch.zeros(1, TOKENS + HW, 8), heads=HEADS, transformer_options=OPTIONS)
    mask = masks[-1]
    assert mask.shape == (1, 1, 1, TOKENS + HW) and mask.dtype == torch.float16
    assert torch.allclose(mask[..., HW:2 * HW].float(), torch.full((HW,), -1.3), atol=1e-3)
    assert (mask[..., :HW] == 0).all() and (mask[..., 2 * HW:] == 0).all()
    # frame 0's call has no pose tail
    override(func, q, torch.zeros(1, TOKENS, 8), torch.zeros(1, TOKENS, 8), heads=HEADS, transformer_options=OPTIONS)
    # whole-clip self-attention when the pose branch is windowed out
    override(func, torch.zeros(1, TOKENS, 8), torch.zeros(1, TOKENS, 8), torch.zeros(1, TOKENS, 8), heads=HEADS, transformer_options=OPTIONS)
    assert len(masks) == 3
    # cross-attention, the pose branch ((f-1)*hw), and blocks without grid info pass through
    assert override(func, torch.zeros(1, TOKENS, 8), torch.zeros(1, TEXT, 8), torch.zeros(1, TEXT, 8), heads=HEADS, transformer_options=OPTIONS) == "plain"
    assert override(func, torch.zeros(1, 3 * HW, 8), torch.zeros(1, 3 * HW, 8), torch.zeros(1, 3 * HW, 8), heads=HEADS, transformer_options=OPTIONS) == "plain"
    assert override(func, q, torch.zeros(1, TOKENS, 8), torch.zeros(1, TOKENS, 8), heads=HEADS, transformer_options={}) == "plain"
    assert len(masks) == 3


def test_calls_without_the_bias_go_to_the_override_installed_before(node_module, masks):
    previous = recording("previous")
    override = node_module._seed_frame_attention_bias(-1.3, previous)
    # cross-attention, the pose branch, a block without grid info
    for (lq, lk), options in (((TOKENS, TEXT), OPTIONS), ((3 * HW, 3 * HW), OPTIONS), ((HW, TOKENS), {})):
        q, k, v = qkv(lq, lk)
        assert override(core, q, k, v, heads=HEADS, transformer_options=options) == "previous"
        func, *tensors, kwargs = previous.calls[-1]
        assert func is core and all(a is b for a, b in zip(tensors, (q, k, v)))
        assert kwargs == {"heads": HEADS, "transformer_options": options}
    assert len(previous.calls) == 3 and masks == []


def test_calls_with_the_bias_stay_on_sdpa_past_the_override_installed_before(node_module, masks):
    previous = recording("previous")
    override = node_module._seed_frame_attention_bias(-1.3, previous)
    # per frame with and without the pose tail, the whole clip
    for lq, lk in ((HW, TOKENS + HW), (HW, TOKENS), (TOKENS, TOKENS)):
        q, k, v = qkv(lq, lk)
        assert torch.equal(override(core, q, k, v, heads=HEADS, transformer_options=OPTIONS), sdpa(q, k, v, HEADS, seed_frame_bias(lk)))
        assert torch.equal(masks[-1], seed_frame_bias(lk))
    assert previous.calls == [] and len(masks) == 3


def test_without_an_override_installed_before_the_calls_run_as_they_did(node_module, masks):
    override = node_module._seed_frame_attention_bias(-1.3)
    # a call with the bias: SDPA with the seed-frame bias
    q, k, v = qkv(HW, TOKENS + HW)
    assert torch.equal(override(core, q, k, v, heads=HEADS, transformer_options=OPTIONS), sdpa(q, k, v, HEADS, seed_frame_bias(TOKENS + HW)))
    # a call without it: core's attention function, untouched
    q, k, v = qkv(TOKENS, TEXT)
    assert torch.equal(override(core, q, k, v, heads=HEADS, transformer_options=OPTIONS), sdpa(q, k, v, HEADS))
    assert len(masks) == 1


def test_patch_model_chains_the_override_installed_before(node_module, masks):
    previous = recording("previous")
    model = FakeModel()
    model.model_options["transformer_options"][SLOT] = previous
    installed = prepared(node_module, -1.3).patch_model(model, {}).model_options["transformer_options"][SLOT]
    assert installed is not previous and inspect.getclosurevars(installed).nonlocals["inner"] is previous
    assert model.model_options["transformer_options"][SLOT] is previous  # the input model keeps its own
    q, k, v = qkv(TOKENS, TEXT)
    assert installed(core, q, k, v, heads=HEADS, transformer_options=OPTIONS) == "previous"
    assert all(a is b for a, b in zip(previous.calls[-1], (core, q, k, v)))
    installed(core, *qkv(HW, TOKENS), heads=HEADS, transformer_options=OPTIONS)
    assert len(previous.calls) == 1 and len(masks) == 1


def test_after_chunk_logs_the_biased_calls_and_resets_the_count(node_module, masks, caplog):
    caplog.set_level("INFO")
    adapter = prepared(node_module, -1.3)
    installed = adapter.patch_model(FakeModel(), {}).model_options["transformer_options"][SLOT]
    for lq, lk in ((HW, TOKENS + HW), (HW, TOKENS), (TOKENS, TOKENS), (TOKENS, TEXT)):
        installed(core, *qkv(lq, lk), heads=HEADS, transformer_options=OPTIONS)
    adapter.after_chunk(0)
    installed(core, *qkv(HW, TOKENS), heads=HEADS, transformer_options=OPTIONS)
    adapter.after_chunk(1)
    lines = [r.getMessage() for r in caplog.records if "attention calls in chunk" in r.getMessage()]
    assert lines == ["[{}] attn_log_scale -1.30 applied to 3 attention calls in chunk 1.".format(ANIMATE2),
                     "[{}] attn_log_scale -1.30 applied to 1 attention calls in chunk 2.".format(ANIMATE2)]
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_after_chunk_warns_when_the_bias_matched_no_call(node_module, masks, caplog):
    caplog.set_level("INFO")
    adapter = prepared(node_module, -1.3)
    installed = adapter.patch_model(FakeModel(), {}).model_options["transformer_options"][SLOT]
    installed(core, *qkv(TOKENS, TEXT), heads=HEADS, transformer_options=OPTIONS)
    adapter.after_chunk(0)
    assert "applied to 0 attention calls in chunk 1." in caplog.text
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "attn_log_scale -1.30 matched no attention call in chunk 1" in warnings[0]


def test_attn_log_scale_0_installs_nothing_and_never_warns(node_module, masks, caplog):
    caplog.set_level("INFO")
    previous = recording("previous")
    model = FakeModel()
    model.model_options["transformer_options"][SLOT] = previous
    adapter = prepared(node_module, 0.0)
    assert adapter.patch_model(model, {}) is model and model.model_options["transformer_options"][SLOT] is previous
    adapter.after_chunk(0)
    assert "attn_log_scale 0.00 applied to 0 attention calls in chunk 1." in caplog.text
    assert not [r for r in caplog.records if r.levelname == "WARNING"]
