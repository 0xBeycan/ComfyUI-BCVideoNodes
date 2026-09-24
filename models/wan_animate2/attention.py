"""Wan Animate 2's seed-frame attention bias, installed as an attention override."""


def seed_frame_attention_bias(log_scale):
    """Wan Animate 2's ``log_scale`` (wanxiang/models/wan_animate_2_model.py
    _score_mod_impl): in every generation self-attention the keys of latent
    frame 1 - the previous chunk's last frame on chained chunks, grey fill on
    the first - get ``log_scale`` added to their logits. The official distilled
    checkpoint runs with -1.3 (infer/wan_animate_2_distillation.yaml), the
    base one with 0.0; core has no equivalent, so distilled weights attend to
    the seed frame e^1.3 = 3.7x harder than they were trained to.

    Installed as transformer_options["optimized_attention_override"]. The
    generation calls are told apart by shape: per frame (q = hw tokens, k =
    all gen tokens, plus that frame's pose tokens) or the whole clip when the
    pose branch is windowed out; cross-attention and the pose branch have
    other shapes and pass through untouched."""

    def override(func, q, k, v, **kwargs):
        options = kwargs.get("transformer_options") or {}
        grid = options.get("grid_sizes")
        if options.get("block_type") == "double" and grid is not None:
            frames, gh, gw = grid
            hw = gh * gw
            tokens = frames * hw
            lq, lk = q.shape[1], k.shape[1]
            if (lq == hw and lk in (tokens, tokens + hw)) or (lq == tokens and lk == tokens):
                import comfy.ldm.modules.attention

                bias = q.new_zeros(1, 1, 1, lk)
                bias[..., hw:2 * hw] = log_scale
                return comfy.ldm.modules.attention.attention_pytorch(q, k, v, mask=bias, **kwargs)
        return func(q, k, v, **kwargs)

    return override
