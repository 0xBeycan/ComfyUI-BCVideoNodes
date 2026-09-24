"""G7: the wan_beta sigmas, bit exact, over a grid of steps, shift and denoise (denoise 0.0
included). They pin today's values, which differ from WanVideoWrapper's euler/beta by a few
float32 ULPs at denoise 1.0 and by rule below it (kept by ruling, plan 12.D F1).

Recorded in the ComfyUI venv on CPU: the digests depend on the scipy and torch builds.
ComfyUI itself is stubbed; skipped when torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from golden import check, digest  # noqa: E402
from sampler_fakes import node_module  # noqa: E402,F401

STEPS = (1, 4, 6, 10, 30)
SHIFTS = (1.0, 5.0, 8.0)
DENOISES = (1.0, 0.75, 0.5, 0.0)


def test_wan_beta_sigmas_golden(node_module):
    pytest.importorskip("scipy")
    for steps in STEPS:
        for shift in SHIFTS:
            for denoise in DENOISES:
                sigmas = node_module.wan_beta_sigmas(steps, shift, denoise)
                check(__file__, "steps={} shift={} denoise={}".format(steps, shift, denoise),
                      [digest(sigmas), str(sigmas.dtype)])
