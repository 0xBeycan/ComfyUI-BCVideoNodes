"""The package must import with nothing but pytest installed (no torch, no
ComfyUI): this is how ComfyUI's loader and tooling see it, and how
tests/conftest.py binds it."""

import importlib
import inspect

import pytest

SAMPLERS = {
    "BCVWanAnimateLongVideoSampler": ("Wan Animate Long Video Sampler", "WanAnimateToVideo"),
    "BCVWanAnimate2LongVideoSampler": ("Wan Animate 2 Long Video Sampler", "WanAnimate2ToVideo"),
}

# id -> (display name, category, output names)
PREPROCESS = {
    "BCVPoseDetection": ("Pose Detection", "BCVideoNodes", ("pose_images", "pose_data", "bboxes", "key_frame_body_points")),
    "BCVPoseConfig": ("Pose Config", "BCVideoNodes", ("pose_config",)),
    "BCVSAM3VideoTrack": ("SAM 3.1 Multiplex Video Track", "BCVideoNodes", ("mask",)),
    "BCVSAM3Config": ("SAM 3.1 Multiplex Config", "BCVideoNodes", ("sam3_config",)),
    "BCVFaceCrop": ("Face Crop", "BCVideoNodes", ("face_images", "face_bboxes")),
    "BCVPoseGuard": ("Pose Guard", "BCVideoNodes", ("pose_data", "report", "metrics", "timeline")),
    "BCVMaskGuard": ("Mask Guard", "BCVideoNodes", ("mask", "report", "metrics", "timeline")),
    # the source node's outputs in the source order, then the individual nodes' new ones
    "BCVWanAnimatePreprocess": ("WanAnimate Preprocess", "BCVideoNodes/Wan/Animate",
                                ("pose_images", "face_images", "mask", "pose_data", "bboxes", "key_frame_body_points", "face_bboxes")),
    "BCVWanAnimatePreprocessGuard": ("WanAnimate Preprocess Guard", "BCVideoNodes/Wan/Animate",
                                     ("mask", "pose_data", "report", "metrics", "timeline")),
}


def test_mappings():
    nodes = importlib.import_module("bcvideonodes")
    assert set(nodes.NODE_CLASS_MAPPINGS) == set(SAMPLERS) | set(PREPROCESS)
    assert set(nodes.NODE_DISPLAY_NAME_MAPPINGS) == set(SAMPLERS) | set(PREPROCESS)


@pytest.mark.parametrize("node_id", sorted(SAMPLERS))
def test_node_contract(node_id):
    nodes = importlib.import_module("bcvideonodes")
    display_name, animate_node = SAMPLERS[node_id]
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] == display_name

    cls = nodes.NODE_CLASS_MAPPINGS[node_id]
    assert cls.__name__ == node_id
    assert cls.ANIMATE_NODE == animate_node
    assert cls.CATEGORY == "BCVideoNodes/Wan/Animate"
    assert cls.DESCRIPTION
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES) == 3
    assert callable(getattr(cls, cls.FUNCTION))
    assert inspect.ismethod(cls.INPUT_TYPES)
    # the node-specific inputs are widgets/links the core node also has
    required, optional = cls._animate_inputs(16384)
    assert required and optional
    assert not (set(required) & set(optional))


@pytest.mark.parametrize("node_id", sorted(PREPROCESS))
def test_preprocess_node_contract(node_id):
    nodes = importlib.import_module("bcvideonodes")
    display_name, category, outputs = PREPROCESS[node_id]
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS[node_id] == display_name

    cls = nodes.NODE_CLASS_MAPPINGS[node_id]
    assert cls.__name__ == node_id
    assert cls.CATEGORY == category
    assert cls.DESCRIPTION
    assert cls.RETURN_NAMES == outputs
    assert len(cls.RETURN_TYPES) == len(cls.RETURN_NAMES)
    assert callable(getattr(cls, cls.FUNCTION))
    assert inspect.ismethod(cls.INPUT_TYPES)


def test_module_top_level_has_no_heavy_imports():
    import os
    import types

    root = importlib.import_module("bcvideonodes")
    folder = os.path.join(root.__path__[0], "nodes")
    names = sorted("bcvideonodes.nodes" + ("" if f == "__init__.py" else "." + f[:-3])
                   for f in os.listdir(folder) if f.endswith(".py"))
    for module in [root] + [importlib.import_module(name) for name in names]:
        # pack modules (the root binds `nodes`, nodes/ binds its modules) are not imports
        imported = {name for name, value in vars(module).items() if isinstance(value, types.ModuleType)
                    and value.__name__.split(".")[0] != "bcvideonodes"}
        assert imported <= {"inspect", "logging"}, module.__name__
