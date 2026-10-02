"""nodes/full_clear.py: the hook ComfyUI-BCNodes' full clear calls, in the server's
bc_full_clear_hooks list (the locked contract: a list on PromptServer.instance, created by whichever
pack registers first, of zero-argument callables returning {model name: bytes})."""
import types

import pytest

from full_clear_fakes import full_clear

torch = pytest.importorskip("torch")


def test_registers_once_creating_the_list_when_first(monkeypatch):
    server = types.SimpleNamespace()
    monkeypatch.setattr(full_clear, "prompt_server", lambda: server)
    hook = full_clear.register_full_clear_hook()
    assert full_clear.FULL_CLEAR_HOOKS == "bc_full_clear_hooks"
    assert server.bc_full_clear_hooks == [hook] and hook is full_clear.drop_cached_models
    full_clear.register_full_clear_hook()
    assert server.bc_full_clear_hooks == [hook]  # idempotent


def test_joins_the_list_another_pack_created(monkeypatch):
    def theirs():
        return {}

    server = types.SimpleNamespace(bc_full_clear_hooks=[theirs])
    monkeypatch.setattr(full_clear, "prompt_server", lambda: server)
    full_clear.register_full_clear_hook()
    assert server.bc_full_clear_hooks == [theirs, full_clear.drop_cached_models]


def test_no_server_no_hook(monkeypatch):
    monkeypatch.setattr(full_clear, "prompt_server", lambda: None)
    assert full_clear.register_full_clear_hook() is None


def test_the_hook_drops_both_loaders(monkeypatch):
    monkeypatch.setattr(full_clear, "_loaded", {"vitpose.safetensors": types.SimpleNamespace(net=torch.nn.Linear(10, 1, bias=False))})
    sam = full_clear.sam3_loaded
    monkeypatch.setitem(sam, "name", "sam.safetensors")
    monkeypatch.setitem(sam, "model", types.SimpleNamespace(model=torch.nn.Linear(20, 1, bias=False)))
    monkeypatch.setitem(sam, "clip", None)
    assert full_clear.drop_cached_models() == {"vitpose.safetensors": 40, "sam.safetensors": 80}
    assert full_clear.drop_cached_models() == {}
