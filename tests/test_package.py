"""The package must import with nothing but pytest installed (no torch, no
ComfyUI): this is how pytest collects it and how tooling sees it."""

import importlib
import inspect

import pytest

NODES = {
    "WanAnimateLongVideoSampler": ("Wan Animate Long Video Sampler", "WanAnimateToVideo"),
    "WanAnimate2LongVideoSampler": ("Wan Animate 2 Long Video Sampler", "WanAnimate2ToVideo"),
}


def test_mappings():
    nodes = importlib.import_module("nodes")
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(NODES)
    assert set(nodes.NODE_DISPLAY_NAME_MAPPINGS) == set(NODES)


@pytest.mark.parametrize("node_id", sorted(NODES))
def test_node_contract(node_id):
    nodes = importlib.import_module("nodes")
    display_name, animate_node = NODES[node_id]
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] == display_name

    cls = nodes.NODE_CLASS_MAPPINGS[node_id]
    assert cls.__name__ == node_id
    assert cls.ANIMATE_NODE == animate_node
    assert cls.CATEGORY == "WanAnimate"
    assert cls.DESCRIPTION
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES) == 3
    assert callable(getattr(cls, cls.FUNCTION))
    assert inspect.ismethod(cls.INPUT_TYPES)
    # the node-specific inputs are widgets/links the core node also has
    required, optional = cls._animate_inputs(16384)
    assert required and optional
    assert not (set(required) & set(optional))


def test_module_top_level_has_no_heavy_imports():
    import types

    nodes = importlib.import_module("nodes")
    imported = {name for name, value in vars(nodes).items() if isinstance(value, types.ModuleType)}
    assert imported <= {"inspect", "logging"}
