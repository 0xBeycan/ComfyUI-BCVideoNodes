"""Sapiens2 pose (models/sapiens2/, scripts/convert_sapiens2.py): the native module against
transformers' Sapiens2ForPoseEstimation on a tiny random config, the converter's key remap round trip
and config checks, Sapiens2's UDP crop and decode against the formula written out, the 70 -> 133 table by
name against the COCO-WholeBody names, the wrapper's choice of operations for a plain and an int8
ConvRot file, the model file round trip, and the registry and loader entries.

Needs ComfyUI (comfy.ops builds the layers) and is skipped where it is not importable; the transformers
parity and remap tests also need transformers with Sapiens2 (>= 5.16):

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/models/test_sapiens2.py
"""
import importlib.util
import json
import re

import numpy as np
import pytest

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("comfy.ops")

from sapiens2_fakes import TINY, sapiens2  # noqa: E402

CPU = torch.device("cpu")


def _have(module):
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:
        return False


needs_transformers = pytest.mark.skipif(not _have("transformers.models.sapiens2"),
                                        reason="transformers with Sapiens2 (>= 5.16) is not installed")

# the transformers config of the same tiny module as TINY
HF_TINY = dict(hidden_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads_per_layer=[4, 2, 2, 4],
               intermediate_size=128, num_register_tokens=2, image_size=[64, 48], patch_size=16, num_labels=5,
               head_config={"upsample_out_channels": [32, 16], "upsample_kernel_sizes": [4, 4],
                            "conv_out_channels": [16], "conv_kernel_sizes": [1]})


def _hf_model(seed=0):
    from transformers import Sapiens2Config, Sapiens2ForPoseEstimation
    torch.manual_seed(seed)
    model = Sapiens2ForPoseEstimation(Sapiens2Config(**HF_TINY)).eval()
    with torch.no_grad():  # random everything, so a wrongly mapped norm, scale or bias shows
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * 0.2 + (1.0 if p.ndim == 1 and p.shape[0] in (16, 64) else 0.0))
    return model


def _tiny_state(seed=0):
    """A random state dict of the TINY module, in fp32."""
    with torch.device("meta"):
        names = sapiens2.Sapiens2PoseNet(TINY).state_dict()
    g = torch.Generator().manual_seed(seed)
    # the RMSNorm weights around 1, so a wrongly mapped norm shows
    norms = re.compile(r"(^|\.)(ln1|ln2|ln_final|q_norm|k_norm)\.weight$")
    return {k: torch.randn(t.shape, generator=g) * 0.2 + (1.0 if norms.search(k) else 0.0) for k, t in names.items()}


# --- the module against transformers --------------------------------------------------------

@needs_transformers
def test_the_module_matches_transformers():
    hf = _hf_model()
    config = sapiens2.convert.config_from_hf(hf.config.to_dict())
    state = sapiens2.convert.state_from_hf(hf.state_dict())
    net, compute, quantized = sapiens2.build(state, config, CPU, dtype=torch.float32)
    assert (compute, quantized) == (torch.float32, False)
    x = torch.randn(2, 3, 64, 48)
    with torch.no_grad():
        expected = hf(pixel_values=x).heatmaps
        found = net(x)
    assert found.shape == expected.shape == (2, 5, 16, 12)
    assert (found - expected).abs().max().item() < 1e-4 * max(1.0, expected.abs().max().item())


@needs_transformers
def test_the_key_remap_round_trip():
    hf_state = _hf_model().state_dict()
    ours = sapiens2.convert.state_from_hf(hf_state)
    with torch.device("meta"):
        module = sapiens2.Sapiens2PoseNet(sapiens2.convert.config_from_hf(_hf_model().config.to_dict()))
    assert set(ours) == set(module.state_dict())
    assert ours["blocks.1.ffn.w12.weight"].shape == (256, 64)       # gate rows, then up rows
    assert ours["blocks.1.attn.wk.weight"].shape == (32, 64)        # a grouped-query layer: 2 key/value heads
    assert torch.equal(ours["blocks.1.ffn.w12.weight"][:128], hf_state["model.model.layer.1.mlp.gate_proj.weight"])
    back = sapiens2.convert.state_to_hf(ours)
    assert set(back) == set(hf_state)
    assert all(torch.equal(back[k], hf_state[k]) for k in hf_state)
    with pytest.raises(ValueError, match="no place in the native module"):
        sapiens2.convert.state_from_hf({**hf_state, "model.embeddings.mask_token": torch.zeros(1, 1, 64)})


def test_the_converter_refuses_what_the_module_lacks():
    base = {"hidden_act": "silu", "use_gated_mlp": True, "use_qk_norm": True, "use_mask_token": False,
            "head_config": {"use_pixel_shuffle": None, "upsample_out_channels": [8], "upsample_kernel_sizes": [4],
                            "conv_out_channels": [], "conv_kernel_sizes": []},
            "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads_per_layer": None,
            "num_first_full_attention_layers": 1, "num_last_full_attention_layers": 1,
            "num_key_value_attention_heads": 2, "image_size": [32, 16], "id2label": {"0": "a"}, "patch_size": 16,
            "hidden_size": 16, "intermediate_size": 32, "num_register_tokens": 1, "rope_theta": 100.0,
            "rms_norm_eps": 1e-6, "query_bias": True, "key_bias": True, "value_bias": True, "proj_bias": True,
            "mlp_bias": True}
    # transformers' own rule when the per-layer list is absent: full attention on the first and last layers
    assert sapiens2.convert.config_from_hf(base)["kv_heads"] == [4, 2, 4]
    for key, value in (("hidden_act", "gelu"), ("use_gated_mlp", False), ("use_mask_token", True)):
        with pytest.raises(ValueError, match="does not implement"):
            sapiens2.convert.config_from_hf({**base, key: value})


def test_the_quant_recipe_quantizes_every_block_linear_of_every_size():
    # the shipped files (beycanai/sapiens2-convrot README): one exclude for all sizes, group 64 for 5b
    for size, width in (("0.4b", 1024), ("5b", 2432)):
        config = {**TINY, "hidden_size": width, "intermediate_size": 4 * width, "kv_heads": [16, 8], "num_layers": 2,
                  "num_heads": 16}
        with torch.device("meta"):
            names = list(sapiens2.Sapiens2PoseNet(config).state_dict())
        excluded = {n for n in names if re.search(sapiens2.convert.EXCLUDE_LAYERS, n)}
        linear_weights = {n for n in names if re.search(r"\.(wq|wk|wv|proj|w12|w3)\.weight$", n)}
        assert len(linear_weights) == 12 and linear_weights.isdisjoint(excluded), size
        # (the shipped pattern's `(cls|storage)_tokens` misses the singular cls_token; convert_to_quant
        # quantizes 2D `.weight` tensors only, so the 3D token is left as it is all the same)
        assert {"patch_embed.weight", "ln_final.weight", "head.predictor.weight", "blocks.0.attn.q_norm.weight",
                "storage_tokens"} <= excluded
        group = sapiens2.convert.QUANT_RECIPES[size][1]
        assert width % group == 0 and 4 * width % group == 0, size
    assert sapiens2.convert.QUANT_RECIPES == {"0.4b": ("fp32", 256), "0.8b": ("bf16", 256), "1b": ("bf16", 256),
                                              "5b": ("fp32", 64)}
    command = sapiens2.convert.quant_command("5b")
    assert "-i sapiens2_pose_5b_fp32.safetensors" in command and "--convrot-group-size 64" in command


# --- the UDP crop and decode ------------------------------------------------------------------

BOX = (130.0, 60.0, 290.0, 420.0)
POINT = (201.3, 177.8)           # a keypoint in frame pixels, inside BOX


def _crop_pixel(point, center, scale):
    """Where the UDP crop puts a frame point, the official unbiased mapping written out: the crop's
    (input size - 1) pixel spacing spans `scale` frame pixels around `center`."""
    w, h = sapiens2.INPUT_SIZE
    return ((point[0] - center[0] + 0.5 * scale[0]) * (w - 1) / scale[0],
            (point[1] - center[1] + 0.5 * scale[1]) * (h - 1) / scale[1])


def test_the_udp_crop_puts_a_frame_point_where_the_formula_says():
    center, scale = sapiens2.udp_crop_params(BOX)
    # padded 1.25 and widened to the 768:1024 input aspect: the box is 160x360, so the height rules
    assert np.allclose(center, (210.0, 240.0)) and np.allclose(scale, (360 * 1.25 * 0.75, 360 * 1.25))
    frame = np.zeros((480, 400, 3), np.float32)
    yy, xx = np.mgrid[0:480, 0:400]
    frame[..., 0] = 255.0 * np.exp(-((xx - POINT[0]) ** 2 + (yy - POINT[1]) ** 2) / (2 * 3.0 ** 2))
    crop = sapiens2.udp_crop(frame.astype(np.uint8), center, scale)
    assert crop.shape == (1024, 768, 3)
    weight = crop[..., 0].astype(np.float64)
    cy, cx = np.mgrid[0:1024, 0:768]
    found = ((weight * cx).sum() / weight.sum(), (weight * cy).sum() / weight.sum())
    expected = _crop_pixel(POINT, center, scale)
    # half a crop pixel, back in frame pixels (the crop enlarges the frame 2.28 times here)
    assert abs(found[0] - expected[0]) < 0.5 and abs(found[1] - expected[1]) < 0.5


def test_the_udp_decode_returns_the_frame_point_within_half_a_pixel():
    center, scale = sapiens2.udp_crop_params(BOX)
    w, h = sapiens2.HEATMAP_SIZE
    cx, cy = _crop_pixel(POINT, center, scale)
    # the heatmap is the crop at 1/4 on the same unbiased grid: (heatmap size - 1) / (input size - 1)
    hx, hy = cx * (w - 1) / (sapiens2.INPUT_SIZE[0] - 1), cy * (h - 1) / (sapiens2.INPUT_SIZE[1] - 1)
    yy, xx = np.mgrid[0:h, 0:w]
    heatmaps = (0.9 * np.exp(-((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * 2.0 ** 2)))[None].astype(np.float32)
    keypoints = sapiens2.udp_decode(heatmaps, center, scale)
    assert keypoints.shape == (1, 3)
    assert abs(keypoints[0, 0] - POINT[0]) < 0.5 and abs(keypoints[0, 1] - POINT[1]) < 0.5
    assert keypoints[0, 2] == heatmaps[0].max()                      # the confidence is the raw heatmap peak


def test_crop_input_is_the_normalised_udp_crop():
    frame = np.random.default_rng(0).integers(0, 256, (480, 400, 3), dtype=np.uint8)
    x, center, scale = sapiens2.crop_input(frame, BOX)
    crop = sapiens2.udp_crop(frame, *sapiens2.udp_crop_params(BOX))
    expected = ((crop / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)
    assert x.shape == (3, 1024, 768) and x.dtype == np.float32
    assert np.allclose(x, expected, atol=1e-5)


# --- the 70 -> 133 table ----------------------------------------------------------------------

# COCO-WholeBody's 133 keypoint names, as mmpose defines them (configs/_base_/datasets/coco_wholebody.py)
FINGERS = ("thumb", "forefinger", "middle_finger", "ring_finger", "pinky_finger")
COCO_WHOLEBODY = (
    ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
     "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
     "right_ankle", "left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]
    + [f"face-{i}" for i in range(68)]
    + ["left_hand_root"] + [f"left_{f}{j}" for f in FINGERS for j in (1, 2, 3, 4)]
    + ["right_hand_root"] + [f"right_{f}{j}" for f in FINGERS for j in (1, 2, 3, 4)])


def _sapiens2_name(coco):
    """The Sapiens2 keypoint a COCO-WholeBody keypoint is: the hand root is the wrist, a finger's base
    joint (1) is its third_joint, joints 2-4 share the name."""
    side_root = re.fullmatch(r"(left|right)_hand_root", coco)
    if side_root:
        return f"{side_root[1]}_wrist"
    base = re.fullmatch(r"(.+(?:thumb|finger))1", coco)
    return f"{base[1]}_third_joint" if base else coco


def test_the_table_maps_every_coco_keypoint_but_the_face_to_the_sapiens2_keypoint_of_its_name():
    table, names = sapiens2.COCO_FROM_SAPIENS2, sapiens2.SAPIENS2_70_NAMES
    assert list(sapiens2.COCO_WHOLEBODY_NAMES) == COCO_WHOLEBODY
    assert sorted(table) == [i for i in range(133) if not 23 <= i <= 90]
    assert range(133)[sapiens2.FACE] == range(23, 91)
    for coco, s2 in table.items():
        assert names[s2] == _sapiens2_name(COCO_WHOLEBODY[coco]), COCO_WHOLEBODY[coco]


def test_to_coco133_moves_the_rows_and_leaves_the_face_at_0():
    kp = np.random.default_rng(1).random((2, 308, 3)).astype(np.float32)
    out = sapiens2.to_coco133(kp)
    assert out.shape == (2, 133, 3)
    assert np.array_equal(out[:, 9], kp[:, 62]) and np.array_equal(out[:, 132], kp[:, 37])   # left wrist, right pinky tip
    assert not out[:, 23:91].any()


# --- the wrapper: operations, the model file ---------------------------------------------------

def _fake_int8_convrot(state, key, group, orig_dtype="float32"):
    """`state` with Linear `key` replaced by an int8 ConvRot row-scaled weight, as
    convert_to_quant --int8 --convrot --scaling_mode row --comfy_quant writes it."""
    from comfy_kitchen.tensor import TensorWiseINT8Layout
    qdata, params = TensorWiseINT8Layout.quantize(state[f"{key}.weight"].float(), per_channel=True, convrot=True,
                                                  convrot_groupsize=group)
    conf = {"format": "int8_tensorwise", "orig_dtype": orig_dtype, "convrot": True, "convrot_groupsize": group,
            "per_row": True}
    return {**state, f"{key}.weight": qdata, f"{key}.weight_scale": params.scale,
            f"{key}.comfy_quant": torch.tensor(list(json.dumps(conf).encode()), dtype=torch.uint8)}


def test_the_wrapper_picks_quantized_ops_from_the_comfy_quant_metadata():
    pytest.importorskip("comfy_kitchen")
    from comfy.quant_ops import QuantizedTensor
    state = _tiny_state()
    plain, compute, quantized = sapiens2.build(state, TINY, CPU, dtype=torch.float32)
    assert (compute, quantized) == (torch.float32, False)
    assert "MixedPrecisionOps" not in type(plain.blocks[1].ffn.w12).__qualname__
    int8 = _fake_int8_convrot(state, "blocks.1.ffn.w12", group=64)
    assert sapiens2.quant_compute_dtype(int8) == torch.float32 and sapiens2.quant_compute_dtype(state) is None
    net, compute, quantized = sapiens2.build(int8, TINY, CPU)
    assert (compute, quantized) == (torch.float32, True)
    assert "MixedPrecisionOps" in type(net.blocks[1].ffn.w12).__qualname__
    assert isinstance(net.blocks[1].ffn.w12.weight, QuantizedTensor)
    assert not isinstance(net.blocks[0].ffn.w12.weight, QuantizedTensor)   # a layer without .comfy_quant stays plain
    x = torch.randn(1, 3, 64, 48)
    with torch.no_grad():
        a, b = plain(x), net(x)
    # one int8 layer: close to the plain module, not equal
    assert 0 < (a - b).abs().max().item() < 0.05 * a.abs().max().item()
    with pytest.raises(ValueError, match="do not force a dtype"):
        sapiens2.build(int8, TINY, CPU, dtype=torch.float32)
    with pytest.raises(ValueError, match="missing"):
        sapiens2.build({k: v for k, v in state.items() if k != "ln_final.weight"}, TINY, CPU, dtype=torch.float32)


def _write(tmp_path, state, name="sapiens2_pose_tiny_fp32.safetensors", **metadata):
    from safetensors.torch import save_file
    path = str(tmp_path / name)
    save_file({k: v.contiguous() for k, v in state.items()}, path, metadata={
        "format_version": str(sapiens2.FORMAT_VERSION), "architecture": sapiens2.ARCHITECTURE,
        "config": json.dumps(TINY), "dtype": "float32", **metadata})
    return path


def test_the_model_file_round_trip(tmp_path, monkeypatch):
    import comfy.model_management as mm
    monkeypatch.setattr(mm, "get_torch_device", lambda: CPU)
    state = _tiny_state()
    model = sapiens2.Sapiens2Pose(_write(tmp_path, state), dtype=torch.float32)
    assert model.config == TINY and not model.quantized and model.input_dtype == torch.float32
    assert model.input_shape == [1, 3, 64, 48] and model.precision == "float32 compute"
    net, _, _ = sapiens2.build(state, TINY, CPU, dtype=torch.float32)
    x = np.random.default_rng(0).standard_normal((2, 3, 64, 48)).astype(np.float32)
    with torch.no_grad():
        assert np.abs(model.run(x) - net(torch.from_numpy(x)).numpy()).max() < 1e-5


def test_the_wrapper_decodes_a_batch_to_coco133(monkeypatch):
    model = object.__new__(sapiens2.Sapiens2Pose)
    # heatmaps of 70 keypoints at 1/4 of the 1024x768 input, each keypoint's peak one cell further right
    heatmaps = np.zeros((2, 308, 256, 192), np.float32)
    for k in range(70):
        heatmaps[:, k, 100, 20 + k] = 0.8
    model.run = lambda crops: heatmaps
    center, scale = sapiens2.udp_crop_params(BOX)
    out = model.forward(np.zeros((2, 3, 1024, 768), np.float32), np.stack([center] * 2), np.stack([scale] * 2))
    expected = sapiens2.to_coco133(np.stack([sapiens2.udp_decode(h[:70], center, scale) for h in heatmaps]))
    assert out.shape == (2, 133, 3) and np.array_equal(out, expected)
    # the left wrist (COCO 9) is Sapiens2 62: its peak at heatmap column 82
    assert out[0, 9, 0] == pytest.approx(82 / 191 * scale[0] + center[0] - 0.5 * scale[0], abs=0.5)


def test_a_file_of_another_version_names_the_sapiens2_converter(tmp_path, monkeypatch):
    import comfy.model_management as mm
    monkeypatch.setattr(mm, "get_torch_device", lambda: CPU)
    path = _write(tmp_path, _tiny_state(), format_version="0")
    with pytest.raises(ValueError, match="convert the model again with scripts/convert_sapiens2.py"):
        sapiens2.Sapiens2Pose(path)
    path = _write(tmp_path, _tiny_state(), name="other.safetensors", architecture="vitpose")
    with pytest.raises(ValueError, match="expected a sapiens2_pose model file, found 'vitpose'"):
        sapiens2.Sapiens2Pose(path)


# --- the registry and the loader -------------------------------------------------------------

def test_every_sapiens2_file_is_a_registered_pose_estimator_named_as_the_widgets_name_it():
    registry = sapiens2.registry
    names = registry.names("pose_estimator")
    assert names == ["ViTPose-H"] + [f"Sapiens2 {m}" for m in sapiens2.SAPIENS2_MODELS]
    assert list(sapiens2.POSE_MODELS) == names
    files = {
        "Sapiens2 5b int8 convrot": "sapiens2_pose_5b_int8_convrot.safetensors",
        "Sapiens2 5b bf16": "sapiens2_pose_5b_bf16.safetensors",
        "Sapiens2 1b int8 convrot": "sapiens2_pose_1b_int8_convrot.safetensors",
        "Sapiens2 1b bf16": "sapiens2_pose_1b_bf16.safetensors",
        "Sapiens2 0.8b int8 convrot": "sapiens2_pose_0.8b_int8_convrot.safetensors",
        "Sapiens2 0.8b bf16": "sapiens2_pose_0.8b_bf16.safetensors",
        "Sapiens2 0.4b int8 convrot": "sapiens2_pose_0.4b_int8_convrot.safetensors",
        "Sapiens2 0.4b bf16": "sapiens2_pose_0.4b_bf16.safetensors",
    }
    for name, file in files.items():
        assert registry.get("pose_estimator", name) == registry.Entry(sapiens2.Sapiens2Pose, file,
                                                                      "beycanai/sapiens2-convrot")
    assert registry.get("architecture", "sapiens2_pose") == registry.Entry(sapiens2.Sapiens2PoseNet)


def test_the_loader_fetches_a_sapiens2_file_from_its_repo(monkeypatch):
    pytest.importorskip("folder_paths")  # the loader imports the download module (E1)
    from pose_fakes import loader

    class Model:
        def __init__(self, path):
            self.path = path

    fetched = []
    monkeypatch.setattr(loader, "_loaded", {})
    monkeypatch.setattr(loader, "detection_model_path", lambda filename, repo=None: fetched.append((filename, repo))
                        or f"<detection>/{filename}")
    monkeypatch.setattr(sapiens2.registry, "_entries", {**sapiens2.registry._entries, "pose_estimator": {
        **sapiens2.registry._entries["pose_estimator"],
        "Sapiens2 1b bf16": sapiens2.registry.Entry(Model, "sapiens2_pose_1b_bf16.safetensors", "beycanai/sapiens2-convrot")}})
    model = sapiens2.load_pose_estimator("Sapiens2 1b bf16")
    assert model.path == "<detection>/sapiens2_pose_1b_bf16.safetensors"
    assert fetched == [("sapiens2_pose_1b_bf16.safetensors", "beycanai/sapiens2-convrot")]
    with pytest.raises(ValueError, match="unknown pose model 'Sapiens2 2b bf16'"):
        sapiens2.load_pose_estimator("Sapiens2 2b bf16")
