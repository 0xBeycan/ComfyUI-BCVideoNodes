"""The runtime wrappers and the loader's cache.

The wrappers need ComfyUI (model management) and are skipped where it is not importable; on a
machine with ComfyUI run with it on the path:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/models/test_models_wrappers.py
"""
import inspect

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from test_models_checkpoint import conv_cfg, tiny_vitpose  # noqa: E402

from bcvideonodes.models.common import checkpoint  # noqa: E402


def _comfy():
    try:
        import comfy.model_management  # noqa: F401
        import comfy.model_patcher  # noqa: F401
    except Exception:
        pytest.skip("ComfyUI is not importable")
    from pose_fakes import wrappers
    return wrappers


def write(tmp_path, architecture, config, name="model.safetensors"):
    net = checkpoint.build(architecture, config)
    g = torch.Generator().manual_seed(0)
    net.load_state_dict({k: torch.randn(t.shape, generator=g) * 0.2 for k, t in net.state_dict().items()},
                        strict=True, assign=True)
    path = str(tmp_path / name)
    checkpoint.save(net, architecture, path, torch.float32)
    return path


def test_a_wrapper_refuses_another_models_file(tmp_path):
    wrappers = _comfy()
    with pytest.raises(ValueError, match="expected a yolov10 model file, found 'vitpose'"):
        wrappers.Yolo(write(tmp_path, "vitpose", tiny_vitpose()))


# -- the Yolo wrapper ------------------------------------------------------------------------

def mini_yolo():
    """The layout of test_models_extract.py's mini_yolov10 as the extractor reads it out: the
    export's backbone, the PAFPN neck without reduce convolutions and the one-to-one head, at
    tiny widths and a 32x32 input (the config `extract_yolov10` gives, key for key)."""
    def conv(cin, cout, kernel=1, stride=1, groups=1, act="silu"):
        return conv_cfg(cin, cout, kernel, stride=stride, padding=kernel // 2, groups=groups, act=act)

    def c2f(cin, cout, n, shortcut, cib=False):
        c = cout // 2
        if cib:
            convs = [conv(c, c, 3, groups=c), conv(c, 2 * c), conv(2 * c, 2 * c, 3, groups=2 * c), conv(2 * c, c),
                     conv(c, c, 3, groups=c)]
        else:
            convs = [conv(c, c, 3), conv(c, c, 3)]
        return {"type": "c2f", "cv1": conv(cin, 2 * c), "split": [c, c],
                "blocks": [{"convs": convs, "residual": shortcut} for _ in range(n)], "cv2": conv((2 + n) * c, cout)}

    def scdown(cin, cout):
        return {"type": "chain", "convs": [conv(cin, cout), conv(cout, cout, 3, stride=2, groups=cout, act=None)],
                "residual": False}

    backbone = [
        {"type": "conv", **conv(3, 8, 3, stride=2)}, {"type": "conv", **conv(8, 8, 3, stride=2)},
        c2f(8, 8, 1, True), {"type": "conv", **conv(8, 16, 3, stride=2)}, c2f(16, 16, 2, True),
        scdown(16, 16), c2f(16, 16, 1, True, cib=True), scdown(16, 32), c2f(32, 32, 1, True, cib=True),
        {"type": "spp", "conv1": conv(32, 16), "conv2": conv(64, 32), "kernels": [5, 5, 5], "cascade": True},
        {"type": "psa", "cv1": conv(32, 32), "split": [16, 16],
         "attn": {"qkv": conv(16, 32, act=None), "pe": conv(16, 16, 3, groups=16, act=None),
                  "proj": conv(16, 16, act=None), "heads": 2, "key_dim": 4, "head_dim": 8, "scale": 0.5},
         "ffn": {"type": "chain", "convs": [conv(16, 32), conv(32, 16, act=None)], "residual": False},
         "cv2": conv(32, 32)},
    ]
    neck = {"reduce": [None, None], "top_down": [c2f(48, 16, 1, False), c2f(32, 16, 1, False)],
            "down": [{"type": "conv", **conv(16, 16, 3, stride=2)}, scdown(16, 16)],
            "bottom_up": [c2f(32, 16, 1, True, cib=True), c2f(48, 32, 1, True, cib=True)], "scale": 2.0}
    head = {"box": [{"convs": [conv(ch, 8, 3), conv(8, 8, 3), conv(8, 16, act=None)], "residual": False}
                    for ch in (16, 16, 32)],
            "cls": [{"convs": [conv(ch, ch, 3, groups=ch), conv(ch, 8), conv(8, 8, 3, groups=8), conv(8, 8),
                               conv(8, 3, act=None)], "residual": False} for ch in (16, 16, 32)],
            "reg_max": 4, "num_classes": 3, "max_det": 10, "strides": [8, 16, 32]}
    return {"input_size": [32, 32], "backbone": backbone, "features": [4, 6, 10], "neck": neck, "head": head}


def test_yolo_refuses_a_detector_that_does_not_take_640(tmp_path, monkeypatch):
    wrappers = _comfy()
    import comfy.model_management as mm
    monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cpu"))
    with pytest.raises(ValueError):
        wrappers.Yolo(write(tmp_path, "yolov10", mini_yolo()))


def test_yolo_keeps_the_whole_frame_box_on_a_frame_without_a_person():
    wrappers = _comfy()
    yolo = object.__new__(wrappers.Yolo)
    # the attributes Yolo.__init__ sets after loading the net: its defaults, and the 640 input
    for name, parameter in inspect.signature(wrappers.Yolo.__init__).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            setattr(yolo, name, parameter.default)
    yolo.input_width = yolo.input_height = 640
    # [2, 300, 6] detector rows: a person on frame 0, every row below the threshold on frame 1
    rows = np.zeros((2, 300, 6), np.float32)
    rows[0, 0] = (100.0, 80.0, 300.0, 560.0, 0.92, 0.0)
    rows[1, :, 4] = 0.01
    rows[1, :3, :4] = [(100.0, 80.0, 300.0, 560.0), (400.0, 100.0, 520.0, 400.0), (0.0, 0.0, 640.0, 640.0)]
    yolo.run = lambda x: rows
    results = yolo.forward(np.zeros((2, 3, 640, 640), np.float32), np.array([[720, 1280], [160, 120]]))
    assert results[1][0]["bbox"].tolist() == [0.0, 0.0, 120.0, 160.0, -1.0] and results[1][0]["track_id"] == -1


# -- the loader ------------------------------------------------------------------------------

def test_the_loader_builds_a_model_file_once(monkeypatch):
    _comfy()
    from pose_fakes import loader

    class Model:
        def __init__(self, path):
            pass

    monkeypatch.setattr(loader, "_loaded", {})
    monkeypatch.setattr(loader, "detection_model_path", lambda filename: f"<detection>/{filename}")
    first = loader._load(Model, "a.safetensors")
    again = loader._load(Model, "a.safetensors")
    other = loader._load(Model, "b.safetensors")
    assert first is again and other is not first
