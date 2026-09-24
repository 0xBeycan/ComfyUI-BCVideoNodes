"""G16 and G19: the pose models and the converter's crop and person box, recorded from the
production code (specs refactor plan 8.2).

G16 the Yolo wrapper's postprocess and person selection on scripted detector rows, with the
    whole-frame box for a frame whose rows all fall below the threshold; the ViTPose and
    YOLOv10 nets' raw forwards, bit exact, on seeded tiny configs in fp32 and fp16 through a
    written and loaded model file, with their state_dict names and shapes and the metadata
    written; the ViTPose wrapper on a real module; `load_pose_models`' files and its cache; `checkpoint.build`'s unknown-architecture text; the Yolo
    wrapper's refusal of a detector that does not take 640x640.
G19 `scripts/convert_models.py`: `pose_crops` at ViTPose's crop size and `detect_person` on a
    scripted detector.

Everything runs on the CPU: the wrappers are built with ComfyUI's device set to the CPU. The
crops and the Yolo NMS go through cv2, so those digests hold for the cv2 build of the ComfyUI
venv only; they are in tests/goldens/test_models_golden.json:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/models/test_models_golden.py
"""
import inspect
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("safetensors")
mm = pytest.importorskip("comfy.model_management")

from golden import check, digest, log_text  # noqa: E402
from pose_fakes import loader, wrappers  # noqa: E402
# the tiny configs and the seeded weights of the model-file tests, shared rather than copied
from test_models_checkpoint import conv_cfg, filled, tiny_vitpose  # noqa: E402

from bcvideonodes.models.common import checkpoint  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "scripts")


@pytest.fixture
def cpu(monkeypatch):
    """ComfyUI's compute device set to the CPU while a wrapper is built, so it runs there."""
    monkeypatch.setattr(mm, "get_torch_device", lambda: torch.device("cpu"))


# --- a frozen mini YOLOv10 -------------------------------------------------------------------

def conv(cin, cout, kernel=1, stride=1, groups=1, act="silu"):
    return {"cin": cin, "cout": cout, "kernel": [kernel, kernel], "stride": [stride, stride],
            "padding": [kernel // 2, kernel // 2], "dilation": [1, 1], "groups": groups, "bias": True, "act": act}


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


def mini_yolo():
    """The layout of test_models_extract.py's mini_yolov10 as the extractor reads it out: the
    export's backbone, the PAFPN neck without reduce convolutions and the one-to-one head, at
    tiny widths and a 32x32 input (the config `extract_yolov10` gives, key for key)."""
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


def written(tmp_path, architecture, config, dtype, seed=0):
    """A model file of `architecture` with seeded weights, and the net it was written from."""
    net = filled(architecture, config, seed)
    path = str(tmp_path / f"{architecture}_{str(dtype).replace('torch.', '')}.safetensors")
    checkpoint.save(net, architecture, path, dtype, extra={"source": "golden"})
    return net, path


# --- the nets ----------------------------------------------------------------------------------

CONFIGS = {"vitpose": tiny_vitpose, "yolov10": mini_yolo}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("architecture", list(CONFIGS))
def test_raw_forward_golden(architecture, dtype, tmp_path):
    config = CONFIGS[architecture]()
    net, path = written(tmp_path, architecture, config, dtype)
    loaded = checkpoint.load(path, architecture)
    height, width = config["input_size"]
    x = torch.randn(2, 3, height, width, generator=torch.Generator().manual_seed(1)).to(dtype)
    with torch.inference_mode():
        out = loaded(x)
    name = f"{architecture}.{str(dtype).replace('torch.', '')}"
    check(__file__, f"{name}.forward", digest([digest(out)]))
    check(__file__, f"{name}.state_dict",
          digest(sorted((key, tuple(t.shape)) for key, t in loaded.state_dict().items())))
    check(__file__, f"{name}.metadata", checkpoint.read_metadata(path))


def test_vitpose_wrapper_on_a_real_module_golden(tmp_path, cpu):
    config = tiny_vitpose()
    config["head"][-1] = {"type": "conv", **conv_cfg(8, 133, 1)}   # the 133 COCO-WholeBody keypoints
    _, path = written(tmp_path, "vitpose", config, torch.float32, seed=4)
    model = wrappers.ViTPose(path)
    x = np.random.default_rng(4).standard_normal((1, 3, 32, 24)).astype(np.float32)
    center, scale = np.array([[60.0, 80.0]]), np.array([[0.6, 0.8]])
    with pytest.warns(DeprecationWarning):
        plain = model(x, center, scale)
    check(__file__, "vitpose.wrapper", digest(plain))


# --- the Yolo wrapper ----------------------------------------------------------------------------

def scripted_rows():
    """[2, 300, 6] detector rows in 640x640 input pixels, best first. Frame 0: a person, a
    second box on it that NMS drops, a second person, a car, a tiny person, a small person
    behind, one below the threshold. Frame 1: every row below the threshold."""
    rows = np.zeros((2, 300, 6), np.float32)
    rows[0, :7] = [
        (100.0, 80.0, 300.0, 560.0, 0.92, 0.0),
        (110.0, 90.0, 310.0, 570.0, 0.85, 0.0),
        (0.0, 0.0, 640.0, 640.0, 0.7, 2.0),
        (400.0, 100.0, 520.0, 400.0, 0.6, 0.0),
        (10.0, 10.0, 30.0, 40.0, 0.4, 0.0),
        (500.0, 450.0, 600.0, 630.0, 0.2, 0.0),
        (200.0, 200.0, 260.0, 300.0, 0.04, 0.0),
    ]
    rows[1, :, 4] = 0.01
    rows[1, :3, :4] = [(100.0, 80.0, 300.0, 560.0), (400.0, 100.0, 520.0, 400.0), (0.0, 0.0, 640.0, 640.0)]
    return rows


def people(results):
    return [[[(key, (value.dtype.str, value.tolist()) if isinstance(value, np.ndarray) else value)
              for key, value in person.items()] for person in frame] for frame in results]


def test_yolo_postprocess_and_selection_golden():
    yolo = object.__new__(wrappers.Yolo)
    # the attributes Yolo.__init__ sets after loading the net: its defaults, and the 640 input
    for name, parameter in inspect.signature(wrappers.Yolo.__init__).parameters.items():
        if parameter.default is not inspect.Parameter.empty:
            setattr(yolo, name, parameter.default)
    yolo.input_width = yolo.input_height = 640
    rows = scripted_rows()
    yolo.run = lambda x: rows
    shape_raw = np.array([[720, 1280], [160, 120]])
    check(__file__, "yolo.postprocess", digest(yolo.postprocess(rows[0], shape_raw[0], cat_id=yolo.cat_id)))
    results = yolo.forward(np.zeros((2, 3, 640, 640), np.float32), shape_raw)
    # frame 1 keeps the whole-frame box forward starts every frame with
    assert results[1][0]["bbox"].tolist() == [0.0, 0.0, 120.0, 160.0, -1.0] and results[1][0]["track_id"] == -1
    check(__file__, "yolo.forward", digest(people(results)))


def test_yolo_refuses_a_detector_that_does_not_take_640(tmp_path, cpu):
    _, path = written(tmp_path, "yolov10", mini_yolo(), torch.float32)
    with pytest.raises(ValueError) as err:
        wrappers.Yolo(path)
    check(__file__, "yolo.not_640", str(err.value).replace(path, "<file>"))


def test_build_refuses_an_unknown_architecture_golden():
    with pytest.raises(ValueError) as err:
        checkpoint.build("resnet", {})
    check(__file__, "checkpoint.build.unknown", str(err.value))


# --- the loader --------------------------------------------------------------------------------

def test_load_pose_models_golden(monkeypatch):
    calls = []
    monkeypatch.setattr(loader, "_load", lambda cls, filename: calls.append((cls.__name__, filename)) or filename)
    found = [loader.load_pose_models(detector) for detector in (True, False)]
    check(__file__, "load_pose_models.calls", repr(calls))
    check(__file__, "load_pose_models.results", repr(found))


def test_the_loader_cache_golden(monkeypatch, caplog):
    built = []

    class Model:
        def __init__(self, path):
            built.append(path)

    monkeypatch.setattr(loader, "_loaded", {})
    monkeypatch.setattr(loader, "detection_model_path", lambda filename: f"<detection>/{filename}")
    caplog.set_level("INFO")
    caplog.set_level("INFO", logger="BCVideoNodes")
    first = loader._load(Model, "a.safetensors")
    again = loader._load(Model, "a.safetensors")
    other = loader._load(Model, "b.safetensors")
    assert first is again and other is not first
    check(__file__, "loader.cache.built", repr(built))
    check(__file__, "loader.cache.log", digest(log_text(caplog.records)))


# --- G19: the converter ------------------------------------------------------------------------

def convert_models():
    """scripts/convert_models.py, imported the way test_models_extract.py and the P7 smoke
    import it."""
    if SCRIPTS not in sys.path:
        sys.path.insert(0, SCRIPTS)
    import convert_models

    return convert_models


def converter_frames():
    rng = np.random.default_rng(19)
    return [rng.random((160, 120, 3), dtype=np.float32) for _ in range(4)]


CONVERTER_BOXES = [
    np.array([30.0, 20.0, 90.0, 140.0, 0.9]),
    np.array([12.3, 45.6, 78.9, 150.1, 0.7]),
    np.array([-20.0, -30.0, 60.0, 100.0, 0.8]),        # past the left and the top edge
    np.array([0.0, 0.0, 120.0, 160.0, -1.0]),          # the whole frame: nobody detected
]


@pytest.mark.parametrize("input_size", [(256, 192)], ids=["256x192"])
def test_convert_pose_crops_golden(input_size):
    crops, centers, scales = convert_models().pose_crops(converter_frames(), CONVERTER_BOXES, input_size)
    name = f"convert.pose_crops.{input_size[0]}x{input_size[1]}"
    check(__file__, f"{name}.crops", digest([digest(c) for c in crops]))
    check(__file__, f"{name}.centers", digest([digest(c) for c in centers]))
    check(__file__, f"{name}.scales", digest([digest(s) for s in scales]))


class ScriptedYolo:
    """The detector net as detect_person calls it: [1, 3, 640, 640] in, [1, N, 6] rows out."""

    def __init__(self, rows):
        self.rows = torch.tensor(rows, dtype=torch.float32)[None]
        self.seen = []

    def __call__(self, x):
        self.seen.append(digest(x))
        return self.rows


PERSON_ROWS = {
    "no_person": [(0.0, 0.0, 640.0, 640.0, 0.8, 2.0), (10.0, 10.0, 50.0, 50.0, 0.3, 1.0)],
    "person_below": [(0.0, 0.0, 640.0, 640.0, 0.8, 2.0), (100.0, 200.0, 300.0, 500.0, 0.03, 0.0)],
    "person_above": [(0.0, 0.0, 640.0, 640.0, 0.8, 2.0), (100.0, 200.0, 300.0, 500.0, 0.7, 0.0),
                     (50.0, 60.0, 90.0, 99.0, 0.5, 0.0)],
}


@pytest.mark.parametrize("case", list(PERSON_ROWS))
def test_convert_detect_person_golden(case):
    yolo = ScriptedYolo(PERSON_ROWS[case])
    box = convert_models().detect_person(yolo, converter_frames()[0], torch.device("cpu"))
    check(__file__, f"convert.detect_person.{case}", repr((box.dtype.str, box.tolist())))
    check(__file__, f"convert.detect_person.{case}.input", digest(yolo.seen))
