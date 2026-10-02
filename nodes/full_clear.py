"""The pack's full-clear hook: ComfyUI-BCNodes' Process Monitor drops the models this pack keeps
between runs through it, so its full clear brings ComfyUI back to its startup state.

The contract (locked, shared with ComfyUI-BCNodes): `bc_full_clear_hooks`, a list attribute on
PromptServer.instance, created by whichever pack registers first. Each entry is a zero-argument
callable that drops that pack's cached models and returns {model name: bytes it held} (counted from
the tensors it dropped; {} when nothing was loaded); idempotent, and the next node call loads the
model again as it does without a clear.
"""
from .common import prompt_server

FULL_CLEAR_HOOKS = "bc_full_clear_hooks"


def drop_cached_models():
    """This pack's hook: drops the person detector and the pose models (models/common/loader.py) and
    SAM 3.1 Multiplex (models/sam3_1_multiplex/loader.py); {model name: bytes of its weights}."""
    from ..models.common import loader
    from ..models.sam3_1_multiplex import loader as sam3_1_multiplex

    return {**loader.unload(), **sam3_1_multiplex.unload()}


def register_full_clear_hook():
    """Adds drop_cached_models to the server's bc_full_clear_hooks, creating the list when this pack
    registers first; returns the hook, None outside ComfyUI."""
    server = prompt_server()
    if server is None:
        return None
    hooks = getattr(server, FULL_CLEAR_HOOKS, None)
    if hooks is None:
        hooks = []
        setattr(server, FULL_CLEAR_HOOKS, hooks)
    if drop_cached_models not in hooks:
        hooks.append(drop_cached_models)
    return drop_cached_models
