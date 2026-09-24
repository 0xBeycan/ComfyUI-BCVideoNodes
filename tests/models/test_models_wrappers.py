"""The runtime wrappers.

The wrappers need ComfyUI (model management) and are skipped where it is not importable; on a
machine with ComfyUI run with it on the path:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/models/test_models_wrappers.py
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")

from test_models_checkpoint import tiny_vitpose  # noqa: E402

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
