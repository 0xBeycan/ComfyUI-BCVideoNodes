"""Golden digests of the guard (G6, with its log lines, G18): the report, metrics JSON and
timeline of the pose group, the mask group and their combination on every scenario, the
GuardFailed texts, the combine disagreement message in full, and the paths the A/B test scripts
take (pose_data through pickle, the mask back from booleans as float32, the config left out,
stop_on_fail off, the timeline as uint8). Recorded once from the production code before the
refactor moved anything; see tests/golden.py for the storage and the one way to re-record.

The timelines are drawn by cv2 (anti-aliased lines and text), so their digests hold for the cv2
build of the ComfyUI venv only. Run with that venv:

    python -m pytest tests/pipelines/test_guard_golden.py
"""
import json
import logging
import pickle

import numpy as np
import pytest
import torch

from bcvideonodes.pipelines import guard
from golden import check, digest, log_text
from guard_fakes import LEGS, MASK, POSE, H, N, W, clip, drop_keypoints


def faults():
    """clip() with a fault for every check of both groups, each on its own frames, plus what the
    guard keeps as data only (a missed detection, a second person) and a hand the frame edge cut
    off that holds its own keypoints."""
    masks, pose_data = clip()
    metas, detections = pose_data["pose_metas_original"], pose_data["detections"]
    masks[0:3] = 0                                   # mask_empty, and mask_unstable on frame 3
    masks[5, 170:250, 0:30] = 1.0                    # mask_fragmented: an object against the left edge
    masks[8, 5:25, 5:25] = 1.0                       # mask_specks
    masks[9, 0:20, W - 30:] = 1.0                    # a hand at the corner, which her hand keypoints claim
    metas[9]["keypoints_right_hand"] = np.tile([(W - 15) / W, 10 / H, 0.9], (21, 1))
    masks[10, 200:280, 170:210] = 1.0                # mask_attached_leak: background joined to her side
    detections[15]["bbox"] = [0.0, 0.0, 40.0, 40.0]  # subject_switch
    drop_keypoints(pose_data, range(15, 25), [4])    # pose_limb_gap: the right forearm
    drop_keypoints(pose_data, range(18, 24), LEGS)   # pose_incomplete
    masks[26:29, 200:, :] = 0                        # mask_missing_keypoints: the lowest keypoint row cut off
    masks[33, :, 200:] = 1.0                         # mask_leak: a stripe beside her
    drop_keypoints(pose_data, range(35, 38), list(range(10, 20)))   # body_not_drawn
    detections[38] = {"bbox": [0.0, 0.0, float(W), float(H)], "score": -1.0, "persons": 0}
    for i in (20, 21):
        detections[i]["persons"] = 2
    # (frame, body keypoints, dx, dy), the shift as fractions of the frame
    for i, index, dx, dy in ((12, guard.TORSO, 0.5, 0.0),    # pose_jump
                             (30, 7, 0.0, -0.2),             # pose_spike: the left wrist out and back
                             (7, 3, 0.3, 0.0)):              # mask_missed_limb: the right elbow off the body
        pts = metas[i]["keypoints_body"].copy()
        pts[index, 0] += dx
        pts[index, 1] += dy
        metas[i]["keypoints_body"] = pts
    return masks, pose_data


def unpickled(pose_data):
    """pose_data as the A/B scripts load a run's dump: through pickle, without pose_metas (the
    dump leaves the drawing objects out; clip() has none)."""
    return pickle.loads(pickle.dumps(pose_data, protocol=pickle.HIGHEST_PROTOCOL))


def from_bits(masks):
    """The mask as the A/B dump saves it (booleans) and the scripts hand it back: float32 0/1."""
    return torch.from_numpy((masks.numpy() > 0.5).astype(np.float32))


def guards(masks, pose_data, configs=(POSE, MASK), enabled=True, stop_on_fail=True):
    """{"<group>.<output>": value} of check_pose, check_mask and combine_guards over them, each
    with `enabled` and `stop_on_fail`. `configs` None leaves the config out, as the A/B scripts do."""
    pose_config, mask_config = ((), ()) if configs is None else ((configs[0],), (configs[1],))
    _, *pose = guard.check_pose(pose_data, *pose_config, enabled=enabled, stop_on_fail=stop_on_fail)
    _, *mask = guard.check_mask(masks, pose_data, *mask_config, enabled=enabled, stop_on_fail=stop_on_fail)
    both = guard.combine_guards(pose[1], mask[1], stop_on_fail=stop_on_fail)
    return {f"{group}.{output}": value
            for group, result in (("pose", pose), ("mask", mask), ("preprocess", both))
            for output, value in zip(("report", "metrics", "timeline"), result)}


def clean():
    return guards(*clip())


def faults_combined():
    out = guards(*faults(), stop_on_fail=False)
    assert set(json.loads(out["preprocess.metrics"])["flags"]) == set(guard.POSE_CHECKS + guard.MASK_CHECKS)
    return out


def switches_off():
    return guards(*faults(), enabled=False)   # stops on a failure, and nothing can fail


def single_2d_frame():
    masks, pose_data = clip()
    return guards(masks[0], {key: value[:1] if isinstance(value, list) else value for key, value in pose_data.items()})


def zero_frames():
    masks, pose_data = clip()
    pose_data["pose_metas_original"], pose_data["detections"] = [], []
    return guards(masks[:0], pose_data)


def ab_compare():
    """ab_compare.py and ab_sweep.py: both groups measured with the switches off, never stopping."""
    masks, pose_data = faults()
    return guards(from_bits(masks), unpickled(pose_data), configs=None, enabled=False, stop_on_fail=False)


def analyze_run():
    """analyze_run.py: both groups on, never stopping, and the timeline it writes out as uint8."""
    masks, pose_data = faults()
    out = guards(from_bits(masks), unpickled(pose_data), configs=None, stop_on_fail=False)
    out["preprocess.timeline_uint8"] = (out["preprocess.timeline"][0].numpy() * 255).astype(np.uint8)
    return out


SCENARIOS = {f.__name__: f for f in (clean, faults_combined, switches_off, single_2d_frame, zero_frames,
                                     ab_compare, analyze_run)}


def capture(caplog):
    caplog.set_level(logging.INFO)
    caplog.set_level(logging.INFO, logger="BCVideoNodes")


def logged(caplog):
    return log_text(r for r in caplog.records if r.name in ("BCVideoNodes", "root"))


@pytest.mark.parametrize("name", SCENARIOS)
def test_guard_outputs(name, caplog):
    capture(caplog)
    for key, value in SCENARIOS[name]().items():
        check(__file__, f"{name}.{key}", digest(value))
    check(__file__, f"{name}.log", digest(logged(caplog)))


def test_guard_failed_texts(caplog):
    capture(caplog)
    masks, pose_data = faults()
    _, _, pose_metrics, _ = guard.check_pose(pose_data, POSE, stop_on_fail=False)
    _, _, mask_metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    for group, run in (("pose", lambda: guard.check_pose(pose_data, POSE)),
                       ("mask", lambda: guard.check_mask(masks, pose_data, MASK)),
                       ("preprocess", lambda: guard.combine_guards(pose_metrics, mask_metrics))):
        with pytest.raises(guard.GuardFailed) as failure:
            run()
        check(__file__, f"stop_on_fail.{group}.GuardFailed", digest(str(failure.value)))
    check(__file__, "stop_on_fail.log", digest(logged(caplog)))


@pytest.mark.parametrize("named, edit", [("frame", {"frame": 99, "box_iou_prev": 0.0}),   # both shared keys differ
                                         ("box_iou_prev", {"box_iou_prev": 0.0})])
def test_combine_disagreement_message(named, edit):
    masks, pose_data = clip()
    _, _, pose_metrics, _ = guard.check_pose(pose_data, POSE, stop_on_fail=False)
    _, _, mask_metrics, _ = guard.check_mask(masks, pose_data, MASK, stop_on_fail=False)
    record = json.loads(mask_metrics)
    record["frames"][5].update(edit)
    with pytest.raises(ValueError) as failure:
        guard.combine_guards(pose_metrics, json.dumps(record))
    check(__file__, f"disagreement.{named}", str(failure.value))


# --- the SCAIL-2 guard ------------------------------------------------------------------------

from scail2_fakes import scail2  # noqa: E402


def scail2_guard(driving, reference, replacement_mode, enabled=True, stop_on_fail=False):
    """{"scail2.<output>": value} of check_scail2 over the masks rendered by SCAIL-2 Colored Mask."""
    pose_video_mask, reference_image_mask = scail2.colored_masks(driving, replacement_mode, reference)
    _, _, *out = scail2.check_scail2(pose_video_mask, reference_image_mask, enabled=enabled, stop_on_fail=stop_on_fail)
    return {f"scail2.{output}": value for output, value in zip(("report", "metrics", "timeline"), out)}


def scail2_faults():
    """clip()'s person with every SCAIL-2 warning: blank frames, a detached object, a speck, and a
    split landscape reference placed apart from the first frame, in replacement mode."""
    masks = clip()[0]
    masks[[5, 6, 20]] = 0
    masks[12, 280:320, 200:240] = 1.0
    masks[14, 300:316, 220:236] = 1.0
    reference = torch.zeros(1, H, 2 * W)
    reference[0, 40:280, 100:200] = 1.0
    reference[0, 280:320, 400:440] = 1.0
    return masks, reference


SCAIL2_SCENARIOS = {
    "scail2_clean_animation": lambda: scail2_guard(clip()[0], clip()[0][:1], False, stop_on_fail=True),
    "scail2_clean_replacement": lambda: scail2_guard(clip()[0], clip()[0][:1], True, stop_on_fail=True),
    "scail2_warnings": lambda: scail2_guard(*scail2_faults(), True, stop_on_fail=True),
    "scail2_failures": lambda: scail2_guard(torch.zeros(N, H, W), torch.zeros(1, H, W), False),
    "scail2_switch_off": lambda: scail2_guard(torch.zeros(N, H, W), torch.zeros(1, H, W), False, enabled=False,
                                              stop_on_fail=True),
    "scail2_zero_frames": lambda: scail2_guard(clip()[0][:0], clip()[0][:1], False, stop_on_fail=True),
}


@pytest.mark.parametrize("name", SCAIL2_SCENARIOS)
def test_scail2_guard_outputs(name, caplog):
    capture(caplog)
    out = SCAIL2_SCENARIOS[name]()
    flags = json.loads(out["scail2.metrics"])
    if name == "scail2_warnings":
        assert set(flags["flags"]) | set(flags["reference"]["flags"]) == scail2.WARNINGS & set(scail2.SCAIL2_CHECKS)
    for key, value in out.items():
        check(__file__, f"{name}.{key}", digest(value))
    check(__file__, f"{name}.log", digest(logged(caplog)))


def test_scail2_guard_failed_text(caplog):
    capture(caplog)
    pose_video_mask, reference_image_mask = scail2.colored_masks(torch.zeros(N, H, W), False, torch.zeros(1, H, W))
    with pytest.raises(guard.GuardFailed) as failure:
        scail2.check_scail2(pose_video_mask, reference_image_mask)
    check(__file__, "stop_on_fail.scail2.GuardFailed", str(failure.value))
    check(__file__, "stop_on_fail.scail2.log", digest(logged(caplog)))
