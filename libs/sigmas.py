"""wan_beta, the schedule the long-video samplers offer beside ComfyUI's own: the sigmas
WanVideoWrapper's euler/beta scheduler samples with."""

WAN_BETA = "wan_beta"


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
