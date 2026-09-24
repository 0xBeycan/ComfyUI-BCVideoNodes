"""Golden digests: a value recorded once from the production code, compared on every run.

`check(file, key, value)` compares `value` with the one recorded under `key` in
tests/goldens/<file>.json (`file` is the golden test's name or its `__file__`) and fails with the
key, the recorded value and the found one. With BCV_GOLDEN_RECORD=1 a key missing from the file
is written; a recorded key is never overwritten, so a recorded value changes only by an edit of
the JSON file. Values go through JSON, so compare digests and strings, not tuples against lists.

`digest(value)` is the md5 hex of an array or tensor's dtype, shape and bytes, of bytes as they
are, of a string's UTF-8, and of anything else's repr (which keeps tuple vs list, float repr
and dict order). `log_text(records)` is the log lines with their times normalised to "Ns".

Standard library only at module level; numpy and torch are read off the value itself.
"""
import hashlib
import json
import os
import re

GOLDENS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goldens")
RECORD = os.environ.get("BCV_GOLDEN_RECORD") == "1"


def _array_bytes(value):
    """(prefix, bytes) for a numpy array or a torch tensor, else None."""
    if type(value).__module__ == "numpy" and hasattr(value, "dtype") and hasattr(value, "tobytes"):
        import numpy as np

        return f"{value.dtype.str}{value.shape}", np.ascontiguousarray(value).tobytes()
    if type(value).__module__.startswith("torch") and hasattr(value, "untyped_storage"):
        import torch

        flat = value.detach().cpu().contiguous().reshape(-1)
        return f"{value.dtype}{tuple(value.shape)}", flat.view(torch.uint8).numpy().tobytes()
    return None


def digest(value):
    array = _array_bytes(value)
    if array is not None:
        prefix, data = array
        data = prefix.encode() + data
    elif isinstance(value, bytes):
        data = value
    elif isinstance(value, str):
        data = value.encode()
    else:
        data = repr(value).encode()
    return hashlib.md5(data).hexdigest()


def log_text(records):
    """The messages of `records` (e.g. caplog.records), one per line, times as "Ns"."""
    return "\n".join(re.sub(r"\d+\.\ds", "Ns", record.getMessage()) for record in records)


def _path(file):
    return os.path.join(GOLDENS, os.path.splitext(os.path.basename(file))[0] + ".json")


def check(file, key, value):
    path = _path(file)
    recorded = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            recorded = json.load(f)
    found = json.loads(json.dumps(value))
    if key not in recorded:
        if not RECORD:
            raise AssertionError(f"golden {os.path.basename(path)}: nothing recorded under {key!r} "
                                 f"(found {found!r}); record it once with BCV_GOLDEN_RECORD=1")
        recorded[key] = found
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recorded, f, indent=1, sort_keys=True)
            f.write("\n")
        return
    if recorded[key] != found:
        raise AssertionError(f"golden {os.path.basename(path)}[{key!r}]: recorded {recorded[key]!r}, "
                             f"found {found!r}")
