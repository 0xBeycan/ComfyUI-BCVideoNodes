"""The loaders' unloads the full clear calls (models/common/loader.py, models/sam3_1_multiplex/
loader.py): every kept model dropped, the bytes of its weights reported, idempotent, and the next
load builds it again."""
import types
import weakref

import pytest

from full_clear_fakes import full_clear

torch = pytest.importorskip("torch")


def wrapper(n):
    """A built pose model or detector as the loader keeps it: its module in `.net` (n float32 weights)."""
    return types.SimpleNamespace(net=torch.nn.Linear(n, 1, bias=False))


def test_the_pose_loader_drops_every_built_model(monkeypatch):
    built = {"vitpose.safetensors": wrapper(1000), "yolo.safetensors": wrapper(10)}
    refs = [weakref.ref(w.net.weight) for w in built.values()]
    monkeypatch.setattr(full_clear, "_loaded", built)
    del built
    assert full_clear.pose_unload() == {"vitpose.safetensors": 4000, "yolo.safetensors": 40}
    assert full_clear._loaded == {} and all(r() is None for r in refs)
    assert full_clear.pose_unload() == {}


def test_the_sam_loader_drops_its_model_and_text_encoder(monkeypatch):
    loaded = full_clear.sam3_loaded
    model, clip = torch.nn.Linear(100, 1, bias=False), torch.nn.Linear(50, 1, bias=False)
    refs = [weakref.ref(model.weight), weakref.ref(clip.weight)]
    monkeypatch.setitem(loaded, "name", "sam.safetensors")
    monkeypatch.setitem(loaded, "model", types.SimpleNamespace(model=model))  # a ModelPatcher
    monkeypatch.setitem(loaded, "clip", types.SimpleNamespace(cond_stage_model=clip))  # comfy.sd.CLIP
    del model, clip
    assert full_clear.sam3_unload() == {"sam.safetensors": 600}
    assert loaded == {"name": None, "model": None, "clip": None} and all(r() is None for r in refs)
    assert full_clear.sam3_unload() == {}
