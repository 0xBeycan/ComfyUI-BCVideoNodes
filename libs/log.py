"""Console logging for the preprocess: one line when a step starts, one when it ends with
its duration and what it produced, all under ComfyUI's own logging (so they show up in
the ComfyUI console and log file with the usual [INFO] prefix). The long-video sampler's
lines beside its progress bar (active_bar, log_beside_bar) are here too."""
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


# The long-video sampler's console lines go to the root logger, beside the tqdm bar the
# sampler is driving; kept apart from the [BCVideoNodes] lines above.
def active_bar(total_steps):
    """The console tqdm bar the k-diffusion sampler is driving right now.

    It is created inside the sampler loop, so the only handle is tqdm's own
    registry of live bars; match on the step count. None when tqdm is not
    installed or the bar is disabled."""
    try:
        from tqdm import tqdm
        bars = list(tqdm._instances)
    except (ImportError, AttributeError):
        return None
    for bar in bars:
        if getattr(bar, "total", None) == total_steps and not getattr(bar, "disable", False):
            return bar
    return None


def log_beside_bar(message, *args):
    """logging.info that does not tear through a live tqdm bar: the bar is
    cleared, the line printed, the bar redrawn on its next update."""
    try:
        from tqdm import tqdm
        context = tqdm.external_write_mode()
    except ImportError:
        logging.info(message, *args)
        return
    with context:
        logging.info(message, *args)
