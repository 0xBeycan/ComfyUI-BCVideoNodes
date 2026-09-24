"""G9: how the samplers call a core node outside the graph. `_call_node` on a V3 node (FUNCTION a
classmethod, outputs in `.args`), a V1 node returning a tuple and a V1 node returning a
`{"result": ...}` dict; `_with_schema_defaults` filling the required and optional defaults only
where the schema gives one; the text of the missing-node RuntimeError. The fake core classes sit
in the stubbed `sys.modules["nodes"]`; the reprs are pinned as they are.

ComfyUI itself is stubbed; skipped when torch is not installed.
"""

import sys

import pytest

torch = pytest.importorskip("torch")

from golden import check  # noqa: E402
from sampler_fakes import node_module  # noqa: E402,F401


class Outputs:
    """A V3 NodeOutput: the values live in .args."""

    def __init__(self, *args):
        self.args = args


SCHEMA = {
    "required": {
        "given": ("INT", {"default": 1}),
        "combo": (["low", "high"], {"default": "high"}),
        "no_default": ("FLOAT", {"min": 0.0}),
        "bare": ("IMAGE",),
        "options_not_a_dict": ("STRING", "text"),
        "not_a_definition": "MODEL",
        "number": ("FLOAT", {"default": 0.5, "step": 0.1}),
    },
    "optional": {
        "switch": ("BOOLEAN", {"default": False}),
        "mask": ("MASK", {}),
    },
    "hidden": {"unique_id": ("UNIQUE_ID", {"default": "never filled"})},
}


class V3Node:
    FUNCTION = "execute"

    @classmethod
    def INPUT_TYPES(cls):
        return SCHEMA

    @classmethod
    def execute(cls, **kwargs):
        return Outputs("v3", kwargs)


class V1TupleNode:
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"strength": ("FLOAT", {"default": 1.0})}, "optional": None}

    def run(self, **kwargs):
        return ("v1 tuple", kwargs)


class V1DictNode:
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"strength": ("FLOAT", {"default": 2.0})}}

    def run(self, **kwargs):
        return {"ui": {"text": ["shown, not returned"]}, "result": ("v1 dict", kwargs)}


@pytest.fixture
def core(node_module, monkeypatch):
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    for node_id, cls in (("V3Node", V3Node), ("V1TupleNode", V1TupleNode), ("V1DictNode", V1DictNode)):
        monkeypatch.setitem(mappings, node_id, cls)
    return node_module


def test_call_node_golden(core):
    results = [
        core._call_node("V3Node", given=7, no_default=0.25, bare="image"),
        core._call_node("V3Node", number=3.0, given=0),
        core._call_node("V1TupleNode"),
        core._call_node("V1TupleNode", strength=0.3, extra="passed through"),
        core._call_node("V1DictNode"),
    ]
    check(__file__, "call_node", repr(results))


def test_with_schema_defaults_golden(core):
    given = {"no_default": 0.0, "given": 5, "switch": True}
    filled = core._with_schema_defaults(V3Node, given)
    empty = core._with_schema_defaults(V1TupleNode, {})
    check(__file__, "with_schema_defaults", repr([filled, filled is given, empty]))


def test_missing_core_node_golden(core):
    with pytest.raises(RuntimeError) as error:
        core._call_node("NotACoreNode", given=1)
    check(__file__, "missing node", repr((type(error.value).__name__, str(error.value))))
