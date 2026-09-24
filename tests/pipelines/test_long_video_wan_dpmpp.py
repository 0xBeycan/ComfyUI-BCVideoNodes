"""wan_dpmpp, the sampler_name entry for the official Wan DPM-Solver++ 2M: the chunk loop builds it
as core's SamplerDPMPP_2M_SDE node builds it with eta 0, s_noise 1 and the midpoint solver, and wraps
it in the step logger like a KSamplerSelect sampler. The expected sampler is core's own node's; the
loop runs under the sampler_fakes stubs. Runs where ComfyUI is importable (with the ComfyUI root on
PYTHONPATH).
"""
import sys

import pytest

torch = pytest.importorskip("torch")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("comfy_extras.nodes_custom_sampler")

from sampler_fakes import FakeSamplerCustom, node_module, run  # noqa: E402,F401


class RecordingSamplerCustom(FakeSamplerCustom):
    samplers = []

    @classmethod
    def EXECUTE_NORMALIZED(cls, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        cls.samplers.append(sampler)
        return FakeSamplerCustom.EXECUTE_NORMALIZED(model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image)


def test_wan_dpmpp_is_core_s_dpmpp_2m_sde_at_eta_0(request, monkeypatch):
    # core's node and ComfyUI's ksampler, taken before the node_module fixture stubs comfy.*
    import comfy.samplers
    import comfy_extras.nodes_custom_sampler as core

    expected = core.SamplerDPMPP_2M_SDE.execute(solver_type="midpoint", eta=0.0, s_noise=1.0, noise_device="cpu").args[0]
    ksampler = comfy.samplers.ksampler

    module = request.getfixturevalue("node_module")
    monkeypatch.setattr(sys.modules["comfy.samplers"], "ksampler", ksampler, raising=False)
    mappings = sys.modules["nodes"].NODE_CLASS_MAPPINGS
    monkeypatch.setitem(mappings, "SamplerCustom", RecordingSamplerCustom)
    monkeypatch.delitem(mappings, "KSamplerSelect")  # wan_dpmpp does not go through it
    RecordingSamplerCustom.samplers = []

    run(module, pose_frames=160, sampler_name="wan_dpmpp")
    samplers = RecordingSamplerCustom.samplers
    assert len(samplers) == 2 and samplers[0] is samplers[1]
    assert isinstance(samplers[0], module._StepLogger)
    built = samplers[0]._sampler
    assert type(built) is type(expected)
    assert built.sampler_function is expected.sampler_function
    assert built.extra_options == expected.extra_options == {"eta": 0.0, "s_noise": 1.0, "solver_type": "midpoint"}
    assert built.inpaint_options == expected.inpaint_options
