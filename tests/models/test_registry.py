"""The model registry (models/common/registry.py) and the pose families the model packages fill: the
architecture, pose_estimator and person_detector names in registration order, the class and the
file each one holds, and the Pose Detection combo list built from them. The animate family is
tests/models/test_animate_registry.py.

    python -m pytest tests/models/test_registry.py
"""
import pytest

pytest.importorskip("torch")

from bcvideonodes.models.common import registry  # noqa: E402
from bcvideonodes.models.rtmw.net import RTMWNet  # noqa: E402
from bcvideonodes.models.rtmw.wrapper import RTMW  # noqa: E402
from bcvideonodes.models.vitpose.net import ViTPoseNet  # noqa: E402
from bcvideonodes.models.vitpose.wrapper import ViTPose  # noqa: E402
from bcvideonodes.models.yolo.net import YOLOv10Net  # noqa: E402
from bcvideonodes.models.yolo.wrapper import Yolo  # noqa: E402


def entries(family):
    return [(name, registry.get(family, name)) for name in registry.names(family)]


def test_the_architectures_are_the_checkpoint_modules_in_order():
    # the order of checkpoint.ARCHITECTURES before the registry
    assert entries("architecture") == [("vitpose", registry.Entry(ViTPoseNet)), ("rtmw", registry.Entry(RTMWNet)),
                                       ("yolov10", registry.Entry(YOLOv10Net))]


def test_the_pose_estimators_are_the_combo_values_in_order():
    assert entries("pose_estimator") == [
        ("ViTPose-H", registry.Entry(ViTPose, "vitpose_h_wholebody_fp16.safetensors")),
        ("RTMW-l", registry.Entry(RTMW, "rtmw_l_wholebody_384x288_fp32.safetensors")),
    ]


def test_the_person_detector_is_yolov10x():
    assert entries("person_detector") == [("YOLOv10x", registry.Entry(Yolo, "yolov10x_fp32.safetensors"))]


def test_the_pose_model_combo_is_a_new_list_of_the_estimators():
    pytest.importorskip("folder_paths")  # the loader imports the download module (E1)
    from bcvideonodes.models.common import loader

    assert loader.DETECTOR in registry.names("person_detector")
    names = loader.pose_model_names()
    assert names == ["ViTPose-H", "RTMW-l"]
    names.append("changed")
    assert loader.pose_model_names() == ["ViTPose-H", "RTMW-l"]


def test_the_registry_api(monkeypatch):
    monkeypatch.setattr(registry, "_entries", {family: {} for family in registry.FAMILIES})
    registry.register("pose_estimator", "b", int, file="b.safetensors")
    registry.register("pose_estimator", "a", str)
    assert registry.names("pose_estimator") == ["b", "a"] and registry.names("architecture") == []
    assert registry.get("pose_estimator", "b") == registry.Entry(int, "b.safetensors")
    assert registry.get("pose_estimator", "a") == registry.Entry(str, None)
    with pytest.raises(ValueError, match="registered twice"):
        registry.register("pose_estimator", "a", float)
    assert registry.get("pose_estimator", "a").implementation is str
    with pytest.raises(ValueError, match="unknown model family 'segmenter'"):
        registry.register("segmenter", "x", int)
    with pytest.raises(ValueError, match="unknown model family"):
        registry.names("segmenter")
    with pytest.raises(KeyError):
        registry.get("pose_estimator", "c")
