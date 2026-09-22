"""The keypoint decoding and the runtime wrappers.

The decoding needs nothing but numpy and runs anywhere. The wrappers need ComfyUI (model
management) and are skipped where it is not importable; on a machine with ComfyUI run with
it on the path:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/test_models_wrappers.py
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from preprocess.models import checkpoint  # noqa: E402
from preprocess.models.decode import SIMCC_CONF_SCALE, SIMCC_SPLIT_RATIO, decode_simcc  # noqa: E402


def simcc(height, width, peaks, value):
    """SimCC logits for one image whose keypoints peak at the given input pixels."""
    x = np.zeros((1, len(peaks), int(width * SIMCC_SPLIT_RATIO)), np.float32)
    y = np.zeros((1, len(peaks), int(height * SIMCC_SPLIT_RATIO)), np.float32)
    for k, (px, py) in enumerate(peaks):
        x[0, k, int(px * SIMCC_SPLIT_RATIO)] = value
        y[0, k, int(py * SIMCC_SPLIT_RATIO)] = value
    return x, y


def test_the_simcc_divisor_is_the_owners():
    assert SIMCC_CONF_SCALE == 4.6


def test_simcc_decodes_back_to_the_frame():
    # a 40x30 crop centred on (100, 200) covering 60x80 frame pixels: twice the input grid
    x, y = simcc(40, 30, [(15, 20), (0, 0)], 2.3)
    center, scale = np.array([[100.0, 200.0]]), np.array([[60.0, 80.0]]) / 200
    out = decode_simcc(x, y, center, scale, (40, 30))
    assert out.shape == (1, 2, 3)
    assert out[0, 0, :2] == pytest.approx([100.0, 200.0])
    assert out[0, 1, :2] == pytest.approx([70.0, 160.0])
    assert out[0, 0, 2] == pytest.approx(2.3 / SIMCC_CONF_SCALE)
    # another divisor, and the clip to 1
    assert decode_simcc(x, y, center, scale, (40, 30), conf_scale=2.3)[0, 0, 2] == pytest.approx(1.0)
    assert decode_simcc(x, y, center, scale, (40, 30), conf_scale=1.0)[0, 0, 2] == pytest.approx(1.0)


# -- wrappers (ComfyUI) --------------------------------------------------------------------

def _comfy():
    try:
        import comfy.model_management  # noqa: F401
        import comfy.model_patcher  # noqa: F401
    except Exception:
        pytest.skip("ComfyUI is not importable")
    from preprocess.models import wrappers
    return wrappers


def conv_cfg(cin, cout, kernel, stride=1, padding=0, groups=1, act="silu"):
    return {"cin": cin, "cout": cout, "kernel": [kernel, kernel], "stride": [stride, stride],
            "padding": [padding, padding], "dilation": [1, 1], "groups": groups, "bias": True, "act": act}


def tiny_rtmw(height=32, width=32, keypoints=3):
    """An RTMW at the smallest widths the layout allows; the numbers are random, the test
    is about the wrapper around it."""
    def csp(cin, cout, blocks, attention, residual):
        mid = cout // 2
        return {"short_conv": conv_cfg(cin, mid, 1), "main_conv": conv_cfg(cin, mid, 1),
                "blocks": [{"convs": [conv_cfg(mid, mid, 3, padding=1), conv_cfg(mid, mid, 5, padding=2, groups=mid),
                                      conv_cfg(mid, mid, 1)], "residual": residual}] * blocks,
                "attention": {"fc": conv_cfg(cout, cout, 1, act=None), "alpha": 1 / 6, "beta": 0.5} if attention else None,
                "final_conv": conv_cfg(cout, cout, 1)}
    stages = []
    for i, (cin, cout) in enumerate(((4, 4), (4, 8), (8, 16), (16, 32))):
        stages.append({"downsample": conv_cfg(cin, cout, 3, stride=2, padding=1),
                       "spp": {"conv1": conv_cfg(cout, cout // 2, 1), "conv2": conv_cfg(cout * 2, cout, 1),
                               "kernels": [3, 5, 7], "cascade": False} if i == 3 else None,
                       "csp": csp(cout, cout, 1, True, i < 3)})
    neck = {"reduce": [conv_cfg(32, 16, 1), conv_cfg(16, 8, 1)],
            "top_down": [{"type": "csp", **csp(32, 16, 1, False, False)}, {"type": "csp", **csp(16, 8, 1, False, False)}],
            "down": [{"type": "conv", **conv_cfg(8, 8, 3, stride=2, padding=1)},
                     {"type": "conv", **conv_cfg(16, 16, 3, stride=2, padding=1)}],
            "bottom_up": [{"type": "csp", **csp(16, 16, 1, False, False)}, {"type": "csp", **csp(32, 32, 1, False, False)}],
            "scale": 2.0}
    h2, w2 = height // 32, width // 32
    norm = {"scale": 0.5, "gain": 1.0, "eps": 1e-5}
    head = {"final_layer": conv_cfg(32, keypoints, 3, padding=1, act="relu"), "mlp_norm": norm,
            "mlp": [h2 * w2, 8], "mid_layer": conv_cfg(8, 8, 3, padding=1, act="relu"),
            "final_layer2": conv_cfg(24, keypoints, 3, padding=1, act="relu"), "mlp2_norm": norm,
            "mlp2": [4 * h2 * w2, 8],
            "gau": {"norm": norm, "uv": [16, 20], "sizes": [8, 8, 4], "gamma": [2, 4], "beta": [2, 4],
                    "out": [8, 16], "res_scale": [16], "sqrt_s": 2.0},
            "cls_x": [16, int(width * SIMCC_SPLIT_RATIO)], "cls_y": [16, int(height * SIMCC_SPLIT_RATIO)],
            "blocksize": 2}
    return {"input_size": [height, width], "stem": [conv_cfg(3, 4, 3, stride=2, padding=1),
                                                    conv_cfg(4, 4, 3, padding=1), conv_cfg(4, 4, 3, padding=1)],
            "stages": stages, "neck": neck, "head": head}


def write(tmp_path, architecture, config, name="model.safetensors"):
    net = checkpoint.build(architecture, config)
    g = torch.Generator().manual_seed(0)
    net.load_state_dict({k: torch.randn(t.shape, generator=g) * 0.2 for k, t in net.state_dict().items()},
                        strict=True, assign=True)
    path = str(tmp_path / name)
    checkpoint.save(net, architecture, path, torch.float32)
    return path


def test_rtmw_wrapper(tmp_path):
    wrappers = _comfy()
    model = wrappers.RTMW(write(tmp_path, "rtmw", tiny_rtmw()))
    wrappers.load_models(model)
    assert model.input_shape == [1, 3, 32, 32]
    assert model.conf_scale == SIMCC_CONF_SCALE
    img = np.random.default_rng(0).standard_normal((1, 3, 32, 32)).astype(np.float32)
    center, scale = np.array([[50.0, 60.0]]), np.array([[0.2, 0.2]])
    out = model(img, center, scale)
    assert out.shape == (1, 3, 3)
    simcc_x, simcc_y = model.run(img)
    vals = np.minimum(simcc_x.max(axis=2), simcc_y.max(axis=2))
    assert np.allclose(out[..., 2], np.clip(vals / SIMCC_CONF_SCALE, 0, 1))
    # the preprocess overrides the divisor by setting the attribute
    model.conf_scale = 2.0
    other = model(img, center, scale)
    assert np.allclose(other[..., :2], out[..., :2])
    assert np.allclose(other[..., 2], np.clip(vals / 2.0, 0, 1))


def test_rtmw_wrapper_refuses_other_simcc_axes(tmp_path):
    wrappers = _comfy()
    config = tiny_rtmw()
    config["head"]["cls_x"] = [16, 50]
    with pytest.raises(ValueError, match="expected SimCC axes of 32 x 2 and 32 x 2 bins, found 50 and 64"):
        wrappers.RTMW(write(tmp_path, "rtmw", config))


def test_a_wrapper_refuses_another_models_file(tmp_path):
    wrappers = _comfy()
    with pytest.raises(ValueError, match="expected a yolov10 model file, found 'rtmw'"):
        wrappers.Yolo(write(tmp_path, "rtmw", tiny_rtmw()))
    with pytest.raises(ValueError, match="expected a vitpose model file, found 'rtmw'"):
        wrappers.ViTPose(write(tmp_path, "rtmw", tiny_rtmw()))
