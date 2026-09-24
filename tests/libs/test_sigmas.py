"""The wan_beta sigmas against the diffusers schedule they reproduce. ComfyUI itself is
stubbed; skipped when torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import node_module  # noqa: E402,F401


def test_wan_beta_sigmas_match_diffusers(node_module):
    pytest.importorskip("scipy")
    # diffusers FlowMatchEulerDiscreteScheduler(shift, use_beta_sigmas=True).set_timesteps(steps), recorded from diffusers 0.40
    expected = {
        (4, 5.0): [1.0, 0.7313, 0.2931, 0.0244, 0.0],
        (6, 5.0): [1.0, 0.8801, 0.6462, 0.3783, 0.1443, 0.0244, 0.0],
        (4, 8.0): [1.0, 0.7412, 0.3190, 0.0602, 0.0],
        (4, 3.0): [1.0, 0.7270, 0.2819, 0.0089, 0.0],
    }
    for (steps, shift), sigmas in expected.items():
        got = node_module.wan_beta_sigmas(steps, shift).tolist()
        assert [round(v, 4) for v in got] == sigmas, (steps, shift, got)
    # denoise < 1 keeps the tail of a longer schedule, like BasicScheduler
    assert node_module.wan_beta_sigmas(2, 5.0, denoise=0.5).tolist() == node_module.wan_beta_sigmas(4, 5.0).tolist()[-3:]
