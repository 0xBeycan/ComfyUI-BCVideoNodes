"""wan_beta and wan_dpmpp, the schedule and the sampler the long-video samplers offer beside
ComfyUI's own: the sigmas WanVideoWrapper's euler/beta scheduler samples with, and the official
Wan DPM-Solver++."""

WAN_BETA = "wan_beta"
# the sampler_name entry for the DPM-Solver++ (2M) the official Wan pipelines sample with (Wan
# fm_solvers.py FlowDPMSolverMultistepScheduler, arXiv 2211.01095), in its flow-matching form:
# ComfyUI's dpmpp_2m_sde with eta 0 (no noise, so the deterministic 2M) and the midpoint solver
# reproduces it; the chunk loop builds it as core's SamplerDPMPP_2M_SDE does. Plain dpmpp_2m steps
# in -log(sigma) (the variance-exploding form), not in the flow model's half-log-SNR, and the
# listed dpmpp_2m_sde samples with eta 1 (stochastic).
WAN_DPMPP = "wan_dpmpp"


def wan_beta_sigmas(steps, shift, denoise=1.0, alpha=0.6, beta=0.6):
    """The sigmas WanVideoWrapper's 'euler/beta' scheduler samples with:
    diffusers FlowMatchEulerDiscreteScheduler(shift, use_beta_sigmas=True).

    Not ComfyUI's 'beta' scheduler. diffusers shifts first and then spreads
    Beta(0.6, 0.6) quantiles between the shifted extremes, so shift only
    moves sigma_min (1/1000 shifted twice: once in __init__, once in
    set_timesteps) and the steps stay evenly spread. ComfyUI's beta takes
    the quantiles on the timestep axis and reads them off the shifted table,
    which at shift 5 and 4 steps gives 1 / .959 / .834 / .518 / 0 against
    the wrapper's 1 / .731 / .293 / .024 / 0.
    """
    import numpy
    import scipy.stats
    import torch

    total = int(steps / denoise) if 0.0 < denoise < 1.0 else int(steps)
    sigma_min = 1.0 / 1000
    for _ in range(2):
        sigma_min = shift * sigma_min / (1 + (shift - 1) * sigma_min)
    quantiles = scipy.stats.beta.ppf(1 - numpy.linspace(0, 1, total), alpha, beta)
    sigmas = [float(sigma_min + q * (1.0 - sigma_min)) for q in quantiles] + [0.0]
    return torch.FloatTensor(sigmas[-(int(steps) + 1):])
