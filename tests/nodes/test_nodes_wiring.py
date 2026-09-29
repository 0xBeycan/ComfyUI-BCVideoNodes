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
from guard_fakes import clip, origin, place_keypoints  # noqa: E402
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
    assert list(spec("BCVMaskGuard")["required"]) == ["mask", "mask_guard"] + [f.name for f in dataclasses.fields(guard.MaskGuardConfig)]
    assert list(spec("BCVMaskGuard")["optional"]) == ["pose_data"]
    assert "Optional, but it gives the best result" in spec("BCVMaskGuard")["optional"]["pose_data"][1]["tooltip"]
    both = spec("BCVWanAnimatePreprocessGuard")["required"]
    assert list(both)[:4] == ["mask", "pose_data", "pose_guard", "mask_guard"]
    assert set(both) == set(spec("BCVPoseGuard")["required"]) | set(spec("BCVMaskGuard")["required"])


def test_the_keypoint_mask_fail_widgets_come_last():
    # added after the thresholds saved workflows hold, so those keep their widget positions
    for key in ("BCVMaskGuard", "BCVWanAnimatePreprocessGuard"):
        required = spec(key)["required"]
        assert list(required)[-2:] == ["head_out_eyes_ears", "large_loss_area"], key
        count, share = required["head_out_eyes_ears"], required["large_loss_area"]
        assert count[0] == "INT" and {**count[1], "tooltip": None} == {"default": 2, "min": 1, "max": 4, "step": 1, "tooltip": None}
        assert share[0] == "FLOAT" and {**share[1], "tooltip": None} == {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                                                                         "tooltip": None}
        assert count[1]["tooltip"].startswith("mask_head_out (fail)") and share[1]["tooltip"].startswith("mask_loss_large (fail)")


def test_the_guard_wrapper_says_it_takes_the_final_mask():
    # its mask checks are measured on the final mask, and its tooltips say so; the widget itself
    # (type, default, range) is the Mask Guard's
    both, mask = spec("BCVWanAnimatePreprocessGuard")["required"], spec("BCVMaskGuard")["required"]
    assert both["mask"][0] == "MASK" and "BlockifyMask" in both["mask"][1]["tooltip"]
    wrapped, raw = both["max_mask_loss"][1], mask["max_mask_loss"][1]
    assert "holds a drawn keypoint on every frame" in wrapped["tooltip"] and "leaves out" in raw["tooltip"]
    assert {**wrapped, "tooltip": None} == {**raw, "tooltip": None}


@pytest.mark.parametrize("with_pose", [True, False])
def test_the_guard_nodes_return_what_the_guard_does(with_pose):
    masks, pose_data = clip()
    out = nodes.BCVPoseGuard().check(pose_data, True, **thresholds(POSE_THRESHOLDS))
    direct = guard.check_pose(pose_data, POSE_THRESHOLDS)
    assert out[0] is pose_data and out[1:3] == direct[1:3] and torch.equal(out[3], direct[3])
    pose_data = pose_data if with_pose else None
    out = nodes.BCVMaskGuard().check(masks, True, pose_data=pose_data, **thresholds(MASK_THRESHOLDS))
    direct = guard.check_mask(masks, pose_data, MASK_THRESHOLDS)
    assert out[0] is masks and out[1:3] == direct[1:3] and torch.equal(out[3], direct[3])


def test_the_guard_wrapper_reads_the_spike_threshold_of_its_pose_guard():
    # a wrist that jumps 40 px off the body for a frame: a spike at the default 0.08 of the frame
    # height (26 px), and a limb end the mask lost when max_limb_spike is raised above the jump
    masks, pose_data = clip()
    pts = pose_data["pose_metas_original"][20]["keypoints_body"].copy()
    pts[4, 0] -= 40 / masks.shape[2]
    pose_data["pose_metas_original"][20]["keypoints_body"] = pts
    for spike, flagged in ((0.08, "pose_spike"), (0.2, "mask_missed_limb")):
        values = {**thresholds(POSE_THRESHOLDS), "max_limb_spike": spike, **thresholds(MASK_THRESHOLDS)}
        flags = json.loads(nodes.BCVWanAnimatePreprocessGuard().check(masks, pose_data, True, True, **values)[3])["flags"]
        assert list(flags) == [flagged], flags


@pytest.mark.parametrize("key", ["BCVMaskGuard", "BCVWanAnimatePreprocessGuard"])
def test_the_guard_nodes_read_the_keypoint_mask_fail_widgets(key):
    # her right ear drawn 60 px left of her rectangle from frame 20 on, beyond the final mask too:
    # one ear alone is no check at the default head_out_eyes_ears 2 (19 of 20 drawn keypoints
    # inside), and fails at 1
    masks, pose_data = clip()
    for i in range(20, masks.shape[0]):
        x1, y1 = origin(i)
        place_keypoints(pose_data, [i], 16, x1 - 60, y1 + 18)
    for count, flags in ((2, {}), (1, {"mask_head_out": list(range(20, masks.shape[0]))})):
        values = {**thresholds(MASK_THRESHOLDS), "head_out_eyes_ears": count}
        if key == "BCVMaskGuard":
            metrics = nodes.BCVMaskGuard().check(masks, False, pose_data=pose_data, **values)[2]
        else:
            metrics = nodes.BCVWanAnimatePreprocessGuard().check(masks, pose_data, False, False,
                                                                 **thresholds(POSE_THRESHOLDS), **values)[3]
        record = json.loads(metrics)
        assert record["flags"] == flags and record["thresholds"]["head_out_eyes_ears"] == count, (count, record["flags"])


@pytest.mark.parametrize("switches", [(True, True), (False, True), (True, False), (False, False)])
def test_the_guard_wrapper_is_the_two_guards_combined(switches):
    masks, pose_data = clip()
    pts = pose_data["pose_metas_original"][12]["keypoints_body"].copy()
    pts[guard.TORSO, 0] += 0.5                       # a torso jump, a pose fault that stops, so the switches matter
    pose_data["pose_metas_original"][12]["keypoints_body"] = pts
    pose_guard, mask_guard = switches
    values = {**thresholds(POSE_THRESHOLDS), **thresholds(MASK_THRESHOLDS)}
    try:
        wrapped = nodes.BCVWanAnimatePreprocessGuard().check(masks, pose_data, pose_guard, mask_guard, **values)
    except guard.GuardFailed as failure:
        wrapped = failure
    # the chain: each group measuring without stopping, then the combined report; the wrapper's
    # mask is the final mask of the Wan Animate workflow
    pose_metrics = guard.check_pose(pose_data, POSE_THRESHOLDS, pose_guard, stop_on_fail=False)[2]
    mask_metrics = guard.check_mask(masks, pose_data, MASK_THRESHOLDS, mask_guard, stop_on_fail=False, final=True)[2]
    if pose_guard:
        assert isinstance(wrapped, guard.GuardFailed) and "pose_jump" in str(wrapped)
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
    config = pose.PoseConfig()
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
