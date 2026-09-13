"""The package must import with nothing but pytest installed (no torch, no
ComfyUI): this is how pytest collects it and how tooling sees it."""

import importlib
import inspect


def test_mappings_and_node_contract():
    nodes = importlib.import_module("nodes")
    assert set(nodes.NODE_CLASS_MAPPINGS) == {"WanAnimate2LongVideoSampler"}
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["WanAnimate2LongVideoSampler"] == "Wan Animate 2 Long Video Sampler"

    cls = nodes.NODE_CLASS_MAPPINGS["WanAnimate2LongVideoSampler"]
    assert cls.CATEGORY == "WanAnimate2"
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES) == 3
    assert callable(getattr(cls, cls.FUNCTION))
    assert inspect.ismethod(cls.INPUT_TYPES)


def test_module_top_level_has_no_heavy_imports():
    import types

    nodes = importlib.import_module("nodes")
    imported = {name for name, value in vars(nodes).items() if isinstance(value, types.ModuleType)}
    assert imported <= {"inspect", "logging"}
