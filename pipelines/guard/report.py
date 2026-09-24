"""The report of a guard run, and the step that finishes one: the report text, the metrics JSON,
the timeline, and the stop on a failed enabled check."""
import json
from typing import Union

from ...libs import log
from .common import WARNINGS, GuardFailed, MaskRow, PoseRow, PreprocessRow
from .timeline import timeline_image


def _span(start, end):
    """One run of `_ranges`: '1-3', or '7' when it is one frame."""
    return f"{start}-{end}" if end > start else str(start)


def _ranges(frames):
    """[1, 2, 3, 7, 9, 10] -> '1-3, 7, 9-10'"""
    out, start, prev = [], None, None
    for f in frames:
        if start is None:
            start = prev = f
        elif f == prev + 1:
            prev = f
        else:
            out.append(_span(start, prev))
            start = prev = f
    if start is not None:
        out.append(_span(start, prev))
    return ", ".join(out)


def longest_run(frames):
    """The longest run of consecutive frame numbers in a sorted list."""
    longest = run = 0
    prev = None
    for f in frames:
        run = run + 1 if prev is not None and f == prev + 1 else 1
        longest = max(longest, run)
        prev = f
    return longest


# The per-frame list a check's report line counts the names of.
REPORT_NAMES = {"pose_incomplete": "lost_limbs", "pose_spike": "limb_spikes", "pose_limb_gap": "limb_gaps",
                "mask_missing_keypoints": "missed_keypoints", "mask_missed_limb": "missed_limbs"}


def write_report(title, rows: Union[list[PoseRow], list[MaskRow], list[PreprocessRow]], flags, enabled):
    """The report text and whether the enabled checks all passed."""
    n = len(rows)
    failed = [name for name in flags if name in enabled and name not in WARNINGS]
    lines = [f"{title}: {'FAILED' if failed else 'passed'} - "
             f"{len(failed)} check(s) failed on {len({i for name in failed for i in flags[name]})}/{n} frames"]
    for name, frames in flags.items():
        kind = "warning" if name in WARNINGS else ("fail" if name in enabled else "off")
        line = f"- {name} ({kind}): {len(frames)} frame(s), longest run {longest_run(frames)}: {_ranges(frames)}"
        key = REPORT_NAMES.get(name)
        if key:
            missed = {}
            for i in frames:
                for k in rows[i][key]:
                    missed[k] = missed.get(k, 0) + 1
            line += " | missed: " + ", ".join(f"{k} x{c}" for k, c in sorted(missed.items(), key=lambda kv: -kv[1]))
        lines.append(line)
    return "\n".join(lines), not failed


def _finish(title, guard, rows: Union[list[PoseRow], list[MaskRow], list[PreprocessRow]], flags, thresholds,
            enabled, panels, stop_on_fail):
    """Report, metrics and timeline of one guard run; stops the workflow on a failed enabled
    check when `stop_on_fail`. `guard` names the group in the metrics (None for the combined
    run, whose metrics keep the layout they always had)."""
    report, passed = write_report(title, rows, flags, enabled)
    record = {"guard": guard} if guard else {}
    record.update({"thresholds": thresholds, "enabled": sorted(enabled), "flags": flags, "frames": rows})
    metrics = json.dumps(record)
    timeline = timeline_image(rows, flags, panels)
    if stop_on_fail:
        # whoever decides whether the workflow stops is the one that logs the report
        log.info(report.replace("\n", "\n    "))
        if not passed:
            raise GuardFailed(report)
    return report, metrics, timeline
