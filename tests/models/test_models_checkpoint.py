"""The model files: a module built from its config, written with the config in the
safetensors metadata, and loaded back without anything but the file - and a file that is
not what the module expects refused with what was expected and what was found.

A miniature ViTPose config stands in for the real models; nothing here needs a model file,
onnx or a GPU.

    python -m pytest tests/models/test_models_checkpoint.py
"""
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file  # noqa: E402

from bcvideonodes.models.common import checkpoint  # noqa: E402


def conv_cfg(cin, cout, kernel, stride=1, padding=0, groups=1, act=None):
    return {"cin": cin, "cout": cout, "kernel": [kernel, kernel], "stride": [stride, stride],
            "padding": [padding, padding], "dilation": [1, 1], "groups": groups, "bias": True, "act": act}


def tiny_vitpose():
    """A 32x24 ViTPose: 8x8 patches, two blocks of width 16, a one-deconvolution head."""
    return {
        "input_size": [32, 24], "patch_embed": conv_cfg(3, 16, 8, stride=8), "num_tokens": 12, "dim": 16,
        "depth": 2, "heads": 2, "scale": 8 ** -0.5, "mlp_hidden": 32, "eps": 1e-6, "gelu": "none",
        "head": [
            {"type": "deconv", "cin": 16, "cout": 8, "kernel": [4, 4], "stride": [2, 2], "padding": [1, 1],
             "output_padding": [0, 0], "dilation": [1, 1], "bias": False},
            {"type": "bn", "channels": 8, "eps": 1e-5},
            {"type": "relu"},
            {"type": "conv", **conv_cfg(8, 5, 1)},
        ],
    }


def filled(architecture, config, seed=0):
    """The module with random weights, on the CPU."""
    net = checkpoint.build(architecture, config)
    g = torch.Generator().manual_seed(seed)
    state = {}
    for name, t in net.state_dict().items():
        if name.endswith("num_batches_tracked"):
            state[name] = torch.tensor(0, dtype=torch.long)
        elif name.endswith("running_var"):
            state[name] = torch.rand(t.shape, generator=g) + 0.5
        else:
            state[name] = torch.randn(t.shape, generator=g) * 0.2
    net.load_state_dict(state, strict=True, assign=True)
    return net.eval()


@pytest.fixture
def written(tmp_path):
    net = filled("vitpose", tiny_vitpose())
    path = str(tmp_path / "tiny.safetensors")
    checkpoint.save(net, "vitpose", path, torch.float32, extra={"source": "test"})
    return net, path


def test_a_file_round_trips_exactly(written):
    net, path = written
    loaded = checkpoint.load(path, "vitpose")
    assert loaded.config == net.config
    x = torch.randn(2, 3, 32, 24, generator=torch.Generator().manual_seed(1))
    with torch.inference_mode():
        a, b = net(x), loaded(x)
    assert a.shape == (2, 5, 8, 6)
    assert torch.equal(a, b)
    meta = checkpoint.read_metadata(path)
    assert meta["architecture"] == "vitpose" and meta["source"] == "test" and meta["dtype"] == "float32"
    assert json.loads(meta["config"]) == tiny_vitpose()


def test_the_precision_is_the_files(tmp_path):
    net = filled("vitpose", tiny_vitpose())
    path = str(tmp_path / "half.safetensors")
    checkpoint.save(net, "vitpose", path, torch.float16)
    loaded = checkpoint.load(path, "vitpose")
    floats = {t.dtype for t in loaded.state_dict().values() if t.is_floating_point()}
    assert floats == {torch.float16}
    assert checkpoint.read_metadata(path)["dtype"] == "float16"


def test_the_wrong_architecture_is_refused(written):
    _, path = written
    with pytest.raises(ValueError, match="expected a yolov10 model file, found 'vitpose'"):
        checkpoint.load(path, "yolov10")


def test_a_file_without_metadata_is_refused(tmp_path):
    path = str(tmp_path / "bare.safetensors")
    save_file({"w": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="no architecture in its metadata"):
        checkpoint.load(path, "vitpose")


def test_another_format_version_is_refused(written, tmp_path):
    _, path = written
    meta = checkpoint.read_metadata(path)
    meta["format_version"] = "0"
    other = str(tmp_path / "old.safetensors")
    save_file(load_file(path), other, metadata=meta)
    with pytest.raises(ValueError, match="expected model file format 1, found '0'"):
        checkpoint.load(other, "vitpose")


def test_a_missing_tensor_is_refused_by_name(written, tmp_path):
    _, path = written
    state = load_file(path)
    del state["blocks.1.fc2.bias"]
    other = str(tmp_path / "missing.safetensors")
    save_file(state, other, metadata=checkpoint.read_metadata(path))
    with pytest.raises(ValueError, match="missing 1 tensors: blocks.1.fc2.bias"):
        checkpoint.load(other, "vitpose")


def test_a_config_that_disagrees_with_the_weights_is_refused(written, tmp_path):
    """A file whose config says three blocks and whose weights hold two: the config builds
    a third block the weights do not fill, and the shapes of what is there must match."""
    _, path = written
    meta = checkpoint.read_metadata(path)
    config = json.loads(meta["config"])
    config["depth"] = 3
    config["mlp_hidden"] = 64
    meta["config"] = json.dumps(config)
    other = str(tmp_path / "wrong.safetensors")
    save_file(load_file(path), other, metadata=meta)
    with pytest.raises(ValueError) as err:
        checkpoint.load(other, "vitpose")
    message = str(err.value)
    assert "missing 12 tensors: blocks.2.fc1.bias" in message
    assert "blocks.0.fc1.weight: expected shape [16, 64], found [16, 32]" in message


def test_mixed_precision_is_refused(written, tmp_path):
    _, path = written
    state = load_file(path)
    state["pos_embed"] = state["pos_embed"].half()
    other = str(tmp_path / "mixed.safetensors")
    save_file(state, other, metadata=checkpoint.read_metadata(path))
    with pytest.raises(ValueError, match="expected one floating precision"):
        checkpoint.load(other, "vitpose")


def test_an_unknown_architecture_is_refused():
    with pytest.raises(ValueError, match="unknown model architecture"):
        checkpoint.build("resnet", {})


def test_a_bad_layer_type_is_refused():
    config = tiny_vitpose()
    config["head"][2] = {"type": "gelu"}
    with pytest.raises(ValueError, match="deconv, conv, bn or relu, found 'gelu'"):
        checkpoint.build("vitpose", config)
