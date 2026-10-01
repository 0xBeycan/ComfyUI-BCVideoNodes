"""The model registry (models/common/registry.py) and the pose families the model packages fill: the
architecture, pose_estimator and person_detector names in registration order, the class and the
file each one holds, and the names the loader reads them by. The animate family is
tests/models/test_animate_registry.py.

    python -m pytest tests/models/test_registry.py
"""
import pytest

pytest.importorskip("torch")

from bcvideonodes.models.common import registry  # noqa: E402
from bcvideonodes.models.sapiens2.net import Sapiens2PoseNet  # noqa: E402
from bcvideonodes.models.sapiens2.wrapper import Sapiens2Pose  # noqa: E402
from bcvideonodes.models.vitpose.net import ViTPoseNet  # noqa: E402
from bcvideonodes.models.vitpose.wrapper import ViTPose  # noqa: E402
from bcvideonodes.models.yolo.net import YOLOv10Net  # noqa: E402
from bcvideonodes.models.yolo.wrapper import Yolo  # noqa: E402


def entries(family):
    return [(name, registry.get(family, name)) for name in registry.names(family)]


def test_the_architectures_are_the_checkpoint_modules_in_order():
    # the order of checkpoint.ARCHITECTURES before the registry
    assert entries("architecture") == [("vitpose", registry.Entry(ViTPoseNet)), ("yolov10", registry.Entry(YOLOv10Net)),
                                       ("sapiens2_pose", registry.Entry(Sapiens2PoseNet))]


def test_the_pose_estimators_are_vitpose_h_then_the_sapiens2_files():
    sapiens2 = [(f"Sapiens2 {size} {precision}", registry.Entry(
        Sapiens2Pose, f"sapiens2_pose_{size}_{precision.replace(' ', '_')}.safetensors", "beycanai/sapiens2-convrot"))
        for size in ("5b", "1b", "0.8b", "0.4b") for precision in ("int8 convrot", "bf16")]
    assert entries("pose_estimator") == [
        ("ViTPose-H", registry.Entry(ViTPose, "vitpose_h_wholebody_fp16.safetensors")),
        *sapiens2,
    ]


def test_the_person_detector_is_yolov10x():
    assert entries("person_detector") == [("YOLOv10x", registry.Entry(Yolo, "yolov10x_fp32.safetensors"))]


def test_the_loader_names_are_registered():
    pytest.importorskip("folder_paths")  # the loader imports the download module (E1)
    from bcvideonodes.models.common import loader

    assert loader.DETECTOR in registry.names("person_detector")
    assert loader.POSE_ESTIMATOR in registry.names("pose_estimator")


def test_the_registry_api(monkeypatch):
    monkeypatch.setattr(registry, "_entries", {family: {} for family in registry.FAMILIES})
    registry.register("pose_estimator", "b", int, file="b.safetensors")
    registry.register("pose_estimator", "a", str)
    assert registry.names("pose_estimator") == ["b", "a"] and registry.names("architecture") == []
    registry.register("pose_estimator", "d", float, file="d.safetensors", repo="someone/models")
    assert registry.get("pose_estimator", "b") == registry.Entry(int, "b.safetensors")
    assert registry.get("pose_estimator", "a") == registry.Entry(str, None)
    assert registry.get("pose_estimator", "d") == registry.Entry(float, "d.safetensors", "someone/models")
    assert registry.get("pose_estimator", "b").repo is None
    with pytest.raises(ValueError, match="registered twice"):
        registry.register("pose_estimator", "a", float)
    assert registry.get("pose_estimator", "a").implementation is str
    with pytest.raises(ValueError, match="unknown model family 'segmenter'"):
        registry.register("segmenter", "x", int)
    with pytest.raises(ValueError, match="unknown model family"):
        registry.names("segmenter")
    with pytest.raises(KeyError):
        registry.get("pose_estimator", "c")
