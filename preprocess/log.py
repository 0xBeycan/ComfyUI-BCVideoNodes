"""Console logging for the preprocess: one line when a step starts, one when it ends with
its duration and what it produced, all under ComfyUI's own logging (so they show up in
the ComfyUI console and log file with the usual [INFO] prefix)."""
import logging
import time
from contextlib import contextmanager

_log = logging.getLogger("BCVideoNodes")
PREFIX = "[BCVideoNodes]"


def info(message):
    _log.info(f"{PREFIX} {message}")


def warning(message):
    _log.warning(f"{PREFIX} {message}")


@contextmanager
def step(name, result=None):
    """Logs `name` when entered and `name: done in Ns` when left. `result` is a dict the
    step can fill in; its items are appended to the closing line."""
    info(f"{name} ...")
    t0 = time.perf_counter()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        seconds = time.perf_counter() - t0
        details = ", ".join(f"{k} {v}" for k, v in (result or {}).items())
        info(f"{name}: {'failed after' if failed else 'done in'} {seconds:.1f}s" + (f" ({details})" if details else ""))
