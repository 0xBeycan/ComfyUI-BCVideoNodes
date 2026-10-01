"""The report of a guard run, and the step that finishes one: the report text, the metrics JSON,
the timeline, and the stop on a failed enabled check; with a reference image, the reference
record's lines and its place in the metrics."""
import json
from typing import Union

from ...libs import log
from .common import WARNINGS, GuardFailed, MaskReference, MaskRow, PoseRow, PreprocessRow, Scail2Row
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
REPORT_NAMES = {"pose_spike": "limb_spikes", "mask_head_out": "head_out", "mask_limb_out": "limbs_out"}


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


def _value(value):
    return "n/a" if value is None else f"{value:.3f}"


# What the Mask Guard's reference check says (reference.mask_reference), from its record.
MASK_REFERENCE_LINES = {
    "reference_misaligned": "IoU {iou} with mask frame 0; replacement expects the reference posed and placed like the "
                            "first frame",
}


def reference_report(reference: MaskReference, enabled):
    """The report lines of the Mask Guard's reference record: its measurements, then its checks."""
    line = (f"reference image: area {_value(reference['area'])}, cropped {_value(reference['cropped'])}, "
            f"IoU with mask frame 0 {_value(reference['iou_first_frame'])}, "
            f"scale vs mask frame 0 {_value(reference['scale_first_frame'])}")
    if not reference["area"]:
        line += "; SAM 3.1 Multiplex found no person on it"
    iou = _value(reference["iou_first_frame"])
    return "\n".join([line] + [f"- {name} ({_kind(name, enabled)}): " + MASK_REFERENCE_LINES[name].format(iou=iou)
                               for name in reference["flags"]])


def _finish(title, guard, rows: Union[list[PoseRow], list[MaskRow], list[PreprocessRow]], flags, thresholds,
            enabled, panels, stop_on_fail, note=None, reference: MaskReference = None):
    """Report, metrics and timeline of one guard run; stops the workflow on a failed enabled
    check when `stop_on_fail`. `guard` names the group in the metrics (None for the combined
    run, whose metrics keep the layout they always had); `note` is a report line after the checks.
    `reference`, the record of a connected reference image, adds its lines last and its
    "reference" key before "frames"; without one the report and the metrics are as they were."""
    report, passed = write_report(title, rows, flags, enabled)
    if note:
        report += f"\n- {note}"
    record = {"guard": guard} if guard else {}
    record.update({"thresholds": thresholds, "enabled": sorted(enabled), "flags": flags})
    if reference is not None:
        report += "\n" + reference_report(reference, enabled)
        record["reference"] = reference
    record["frames"] = rows
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
