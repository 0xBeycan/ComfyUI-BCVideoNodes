"""G8: the chunk planner tables. plan_chunks / produced_frames / format_plan over every total
1..400 for each chunk and overlap of the grid, the snap functions over -3..199, and every
ValueError text the planner raises; each table is pinned as the md5 of its JSON.

Pure Python like the planner: no torch, no ComfyUI.
"""

import hashlib
import json

from bcvideonodes.libs.chunking import (
    format_plan,
    next_chunk_length,
    overlap_for_motion_frames,
    plan_chunks,
    produced_frames,
    snap_down,
    snap_up,
)
from golden import check

TOTALS = range(1, 401)
CHUNKS = (5, 9, 33, 49, 81)
OVERLAPS = (0, 1, 5, 13)


def md5_json(value):
    return hashlib.md5(json.dumps(value).encode()).hexdigest()


def plan_row(total, chunk, overlap):
    """[total, plan, produced, plan text], or [total, the ValueError text] when the planner refuses."""
    try:
        plan = plan_chunks(total, chunk, overlap)
    except ValueError as error:
        return [total, "ValueError: {}".format(error)]
    produced = produced_frames(plan, overlap)
    return [total, plan, produced, format_plan(plan, produced, total, total, overlap)]


def error_text(fn, args):
    try:
        fn(*args)
    except ValueError as error:
        return str(error)
    raise AssertionError("{}{} raised no ValueError".format(fn.__name__, args))


def test_plan_tables_golden():
    for chunk in CHUNKS:
        for overlap in OVERLAPS:
            table = [plan_row(total, chunk, overlap) for total in TOTALS]
            check(__file__, "plans chunk={} overlap={}".format(chunk, overlap), md5_json(table))


def test_snap_tables_golden():
    table = [[frames, snap_down(frames), snap_up(frames), overlap_for_motion_frames(frames)] for frames in range(-3, 200)]
    check(__file__, "snaps", md5_json(table))


def test_value_error_texts_golden():
    cases = [
        ("plan_chunks", plan_chunks, (0, 81, 1)),
        ("plan_chunks", plan_chunks, (-4, 81, 1)),
        ("plan_chunks", plan_chunks, (0, 5, 5)),  # the total is checked before the overlap
        ("plan_chunks", plan_chunks, (100, 7, 5)),  # 7 snaps down to 5
        ("plan_chunks", plan_chunks, (100, 81, 81)),
        ("next_chunk_length", next_chunk_length, (0, 100, 5, 5)),
        ("next_chunk_length", next_chunk_length, (0, 100, 84, 81)),
        ("next_chunk_length", next_chunk_length, (100, 100, 5, 13)),  # refused even with nothing left to produce
    ]
    table = [[label, list(args), error_text(fn, args)] for label, fn, args in cases]
    check(__file__, "value errors", md5_json(table))
