"""combine_guards: the one report of the whole preprocess from the metrics of both groups."""
import json

from .common import MASK_CHECKS, POSE_CHECKS, PREPROCESS_ROW, PreprocessRow
from .report import _finish
from .timeline import PREPROCESS_PANELS


def combine_guards(pose_metrics, mask_metrics, stop_on_fail=True):
    """The one report of the whole preprocess from the `metrics` of `check_pose` and
    `check_mask` on the same clip: every frame's measurements side by side, the checks of both
    groups in the order they first fired, one timeline. With `stop_on_fail` a failed enabled
    check of either group raises GuardFailed with the combined report.

    Returns (report, metrics JSON, timeline IMAGE)."""
    pose, mask = json.loads(pose_metrics), json.loads(mask_metrics)
    if pose.get("guard") != "pose" or mask.get("guard") != "mask":
        raise ValueError(f"expected the metrics of check_pose and check_mask, got the metrics of "
                         f"{pose.get('guard')!r} and {mask.get('guard')!r}")
    if len(pose["frames"]) != len(mask["frames"]):
        raise ValueError(f"the pose was checked on {len(pose['frames'])} frames, the mask on {len(mask['frames'])}; "
                         "check the same clip")
    rows: list[PreprocessRow] = []
    for p, m in zip(pose["frames"], mask["frames"]):
        # the shared keys in the pose row's order, not a set's: the message names the same key on every run
        for key in p:
            if key in m and p[key] != m[key]:
                raise ValueError(f"frame {p['frame']}: the pose and the mask checks disagree on {key} "
                                 f"({p[key]} and {m[key]}); check the same clip with the same pose_data")
        merged = {**p, **m}
        rows.append({key: merged[key] for key in PREPROCESS_ROW})
    order = POSE_CHECKS + MASK_CHECKS
    both = {**pose["flags"], **mask["flags"]}
    flags = {name: both[name] for name in sorted(both, key=lambda name: (both[name][0], order.index(name)))}
    thresholds = {**pose["thresholds"], **mask["thresholds"]}
    enabled = set(pose["enabled"]) | set(mask["enabled"])
    return _finish("Preprocess guard", None, rows, flags, thresholds, enabled, PREPROCESS_PANELS, stop_on_fail)
