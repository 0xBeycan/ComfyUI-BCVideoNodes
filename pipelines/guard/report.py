"""The report of a guard run, and the step that finishes one: the report text, the metrics JSON,
the timeline, and the stop on a failed enabled check."""
import json
from typing import Union

from ...libs import log
from .common import WARNINGS, GuardFailed, MaskRow, PoseRow, PreprocessRow, Scail2Row
from .timeline import timeline_image


def longest_run(frames):
    """The longest run of consecutive frame numbers in a sorted list."""
    longest = run = 0
    prev = None
    for f in frames:
        run = run + 1 if prev is not None and f == prev + 1 else 1
        longest = max(longest, run)
        prev = f
    return longest


def _kind(name, enabled):
    """What the check `name` is in a report: a warning, a fail (enabled) or off."""
    return "warning" if name in WARNINGS else ("fail" if name in enabled else "off")


# The per-frame list a check's report line counts the names of.
REPORT_NAMES = {"pose_incomplete": "lost_limbs", "pose_spike": "limb_spikes", "pose_limb_gap": "limb_gaps",
                "mask_missing_keypoints": "missed_keypoints", "mask_missed_limb": "missed_limbs"}


def write_report(title, rows: Union[list[PoseRow], list[MaskRow], list[PreprocessRow], list[Scail2Row]], flags, enabled):
    """The report text and whether the enabled checks all passed."""
    n = len(rows)
    failed = [name for name in flags if name in enabled and name not in WARNINGS]
    lines = [f"{title}: {'FAILED' if failed else 'passed'} - "
             f"{len(failed)} check(s) failed on {len({i for name in failed for i in flags[name]})}/{n} frames"]
    for name, frames in flags.items():
        line = f"- {name} ({_kind(name, enabled)}): {len(frames)} frame(s), longest run {longest_run(frames)}: {log.frame_ranges(frames)}"
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
            enabled, panels, stop_on_fail, note=None):
    """Report, metrics and timeline of one guard run; stops the workflow on a failed enabled
    check when `stop_on_fail`. `guard` names the group in the metrics (None for the combined
    run, whose metrics keep the layout they always had); `note` is a last report line."""
    report, passed = write_report(title, rows, flags, enabled)
    if note:
        report += f"\n- {note}"
    record = {"guard": guard} if guard else {}
    record.update({"thresholds": thresholds, "enabled": sorted(enabled), "flags": flags, "frames": rows})
    metrics = json.dumps(record)
    timeline = timeline_image(rows, flags, panels)
    _stop(report, passed, stop_on_fail)
    return report, metrics, timeline


def _stop(report, passed, stop_on_fail):
    """With `stop_on_fail`, logs the report and stops the workflow (GuardFailed) when it did not
    pass: whoever decides whether the workflow stops is the one that logs the report."""
    if stop_on_fail:
        log.info(report.replace("\n", "\n    "))
        if not passed:
            raise GuardFailed(report)
