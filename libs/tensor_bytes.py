"""What a model's weights hold in memory: the bytes of its parameters and buffers, each storage once."""
import torch


def module_bytes(*modules):
    """Bytes of the parameters and buffers of `modules` (nn.Modules; None is skipped), each storage
    counted once across all of them (tied weights, views of one buffer): what dropping them frees when
    nothing else holds them. Meta tensors hold no memory."""
    seen, total = set(), 0
    for module in modules:
        if module is None:
            continue
        for t in module.state_dict(keep_vars=True).values():
            if not isinstance(t, torch.Tensor) or t.device.type == "meta":
                continue
            try:
                storage = t.untyped_storage()
                key, size = (str(t.device), storage.data_ptr()), storage.nbytes()
            except (RuntimeError, NotImplementedError):  # a tensor subclass without a plain storage
                key, size = id(t), t.numel() * t.element_size()
            if key not in seen:
                seen.add(key)
                total += size
    return total
