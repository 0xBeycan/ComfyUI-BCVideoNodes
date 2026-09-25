"""The preprocess nodes' wiring: every widget reaches the function, the config nodes follow
their dataclasses, the guards return what the pipelines/guard package returns, and each
WanAnimate wrapper computes exactly what its individual nodes compute when chained. Fake models,
synthetic frames. Two of the modules the nodes load import ComfyUI at their top
(models/common/download.py and pipelines/sam3_1_multiplex/track.py), so this runs where ComfyUI
is importable (with the ComfyUI root on PYTHONPATH) and is skipped elsewhere:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_nodes_wiring.py
"""
import dataclasses
import inspect
import json

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("folder_paths")

from names import nodes, spec  # noqa: E402
from bcvideonodes.pipelines import guard  # noqa: E402
from sam3_1_multiplex_fakes import sam3  # noqa: E402

from guard_fakes import MASK as MASK_THRESHOLDS  # noqa: E402
from guard_fakes import POSE as POSE_THRESHOLDS  # noqa: E402
from guard_fakes import LEGS, clip, drop_keypoints  # noqa: E402
from pose_fakes import FakeDetector, FakePose, frames, loader, pose  # noqa: E402


PREPROCESS_NODES = ["BCVPoseDetection", "BCVPoseConfig", "BCVSAM3VideoTrack", "BCVSAM3Config", "BCVFaceCrop",
                    "BCVPoseGuard", "BCVMaskGuard", "BCVWanAnimatePreprocess", "BCVWanAnimatePreprocessGuard",
                    "BCVSCAIL2ColoredMask", "BCVSCAIL2Preprocess"]


def widget_defaults(node_id):
    return {name: options[1]["default"] if len(options) > 1 and "default" in options[1] else options[0][0]
            for name, options in spec(node_id)["required"].items() if name not in ("images", "mask", "pose_data")}


@pytest.mark.parametrize("node_id", PREPROCESS_NODES)
def test_every_input_reaches_the_function(node_id):
    cls = nodes.NODE_CLASS_MAPPINGS[node_id]
    types = spec(node_id)
    params = inspect.signature(getattr(cls, cls.FUNCTION)).parameters
    names = set(types["required"]) | set(types.get("optional", {}))
    if any(p.kind is p.VAR_KEYWORD for p in params.values()):
        assert {n for n in params if n != "self" and params[n].kind is not params[n].VAR_KEYWORD} <= names
    else:
        assert {n for n in params if n != "self"} == names
    # an optional link defaults to None; an optional widget to the widget's own default
    for name, options in types.get("optional", {}).items():
        if name in params:
            widget_default = options[1].get("default") if len(options) > 1 else None
            assert params[name].default == widget_default, name


@pytest.mark.parametrize("node_id, config_cls", [("BCVPoseConfig", pose.PoseConfig), ("BCVSAM3Config", sam3.SAM3Config)])
def test_config_nodes_follow_their_dataclass(node_id, config_cls):
    widgets = spec(node_id)["required"]
    fields = dataclasses.fields(config_cls)
    assert list(widgets) == [f.name for f in fields]
    for f in fields:
        options = widgets[f.name][1]
        assert options["default"] == f.default
        assert options.get("tooltip"), f"{config_cls.__name__}.{f.name} has no description"
    built = nodes.NODE_CLASS_MAPPINGS[node_id]().build(**widget_defaults(node_id))[0]
    assert built == config_cls()


def test_a_config_field_the_node_cannot_show_raises():
    @dataclasses.dataclass
    class Odd:
        points: tuple = (1, 2)

    with pytest.raises(TypeError, match="Odd.points is a <class 'tuple'>"):
        nodes._config_inputs(Odd)


# --- guards --------------------------------------------------------------------------------

def thresholds(config):
    return dataclasses.asdict(config)


def test_the_guard_widgets_are_the_guard_configs():
    assert list(spec("BCVPoseGuard")["required"]) == ["pose_data", "pose_guard"] + [f.name for f in dataclasses.fields(guard.PoseGuardConfig)]
    assert list(spec("BCVMaskGuard")["required"]) == ["mask", "pose_data", "mask_guard"] + [f.name for f in dataclasses.fields(guard.MaskGuardConfig)]
    both = spec("BCVWanAnimatePreprocessGuard")["required"]
    assert list(both)[:4] == ["mask", "pose_data", "pose_guard", "mask_guard"]
    assert set(both) == set(spec("BCVPoseGuard")["required"]) | set(spec("BCVMaskGuard")["required"])


def test_the_guard_nodes_return_what_the_guard_does():
    masks, pose_data = clip()
    out = nodes.BCVPoseGuard().check(pose_data, True, **thresholds(POSE_THRESHOLDS))
    direct = guard.check_pose(pose_data, POSE_THRESHOLDS)
    assert out[0] is pose_data and out[1:3] == direct[1:3] and torch.equal(out[3], direct[3])
    out = nodes.BCVMaskGuard().check(masks, pose_data, True, **thresholds(MASK_THRESHOLDS))
    direct = guard.check_mask(masks, pose_data, MASK_THRESHOLDS)
    assert out[0] is masks and out[1:3] == direct[1:3] and torch.equal(out[3], direct[3])


@pytest.mark.parametrize("switches", [(True, True), (False, True), (True, False), (False, False)])
def test_the_guard_wrapper_is_the_two_guards_combined(switches):
    masks, pose_data = clip()
    drop_keypoints(pose_data, range(18, 24), LEGS)   # a pose fault, so the switches matter
    pose_guard, mask_guard = switches
    values = {**thresholds(POSE_THRESHOLDS), **thresholds(MASK_THRESHOLDS)}
    try:
        wrapped = nodes.BCVWanAnimatePreprocessGuard().check(masks, pose_data, pose_guard, mask_guard, **values)
    except guard.GuardFailed as failure:
        wrapped = failure
    # the chain: each node measuring without stopping, then the combined report
    pose_metrics = guard.check_pose(pose_data, POSE_THRESHOLDS, pose_guard, stop_on_fail=False)[2]
    mask_metrics = guard.check_mask(masks, pose_data, MASK_THRESHOLDS, mask_guard, stop_on_fail=False)[2]
    if pose_guard:
        assert isinstance(wrapped, guard.GuardFailed) and "pose_incomplete" in str(wrapped)
        with pytest.raises(guard.GuardFailed) as chained:
            guard.combine_guards(pose_metrics, mask_metrics)
        assert str(wrapped) == str(chained.value)
        return
    report, metrics, timeline = guard.combine_guards(pose_metrics, mask_metrics)
    assert wrapped[0] is masks and wrapped[1] is pose_data
    assert wrapped[2] == report and wrapped[3] == metrics and torch.equal(wrapped[4], timeline)
    # the unconnected-node chain measures the same: the individual nodes' metrics are the inputs
    assert json.loads(nodes.BCVPoseGuard().check(pose_data, False, **thresholds(POSE_THRESHOLDS))[2]) == json.loads(pose_metrics)


# --- the preprocess wrapper ----------------------------------------------------------------

class FakeSAM3:
    """track() stand-in: a mask from the images, marked by what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, model, images, pose_data=None, bboxes=None, positive_coords=None, negative_coords=None,
                 mode=sam3.MODE_PROMPT, prompt=sam3.PROMPT, max_objects=1, object_index=-1, config=None):
        self.calls.append({"mode": mode, "prompt": prompt, "pose_data": pose_data is not None, "config": config})
        return (images[..., 0] > 0.5).float() + (0.0 if pose_data is None else 0.25)


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(pose, "_to_device", lambda *models: None)
    monkeypatch.setattr(loader, "load_pose_models", lambda detector=True: (FakeDetector() if detector else None, FakePose()))
    fake = FakeSAM3()
    monkeypatch.setattr(sam3, "track", fake)
    monkeypatch.setattr(sam3, "load_sam3", lambda *args: ("model", "clip"))
    return fake


def same(a, b):
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if hasattr(a, "__dict__"):
        return same(vars(a), vars(b))
    return a == b


@pytest.mark.parametrize("mode", list(sam3.MODES))
def test_the_preprocess_wrapper_is_the_three_nodes_chained(fake_models, mode):
    images = frames()
    widgets = dict(body_stick_width=-1, hand_stick_width=-1, draw_head=True, draw_threshold=0.5)
    config = pose.PoseConfig(temporal=False)
    wrapped = nodes.BCVWanAnimatePreprocess().process(images, face_padding=8, mode=mode, prompt=sam3.PROMPT,
                                                      pose_config=config, **widgets)

    pose_images, pose_data, bboxes, key_points = nodes.BCVPoseDetection().detect(images, pose_config=config, **widgets)
    (mask,) = nodes.BCVSAM3VideoTrack().track(images, mode, sam3.PROMPT, 1, -1,
                                             pose_data=pose_data if mode == sam3.MODE_BOX_KEYPOINT else None)
    face_images, face_bboxes = nodes.BCVFaceCrop().crop(images, pose_data, 8)
    chained = (pose_images, face_images, mask, pose_data, bboxes, key_points, face_bboxes)

    assert len(wrapped) == len(nodes.BCVWanAnimatePreprocess.RETURN_NAMES)
    for name, a, b in zip(nodes.BCVWanAnimatePreprocess.RETURN_NAMES, wrapped, chained):
        assert same(a, b), name
    call = fake_models.calls[0]
    assert call["mode"] == mode and call["pose_data"] == (mode == sam3.MODE_BOX_KEYPOINT)
