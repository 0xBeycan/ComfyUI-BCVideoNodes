"""The model files: one safetensors file per model, holding the native module's own state
dict and, in the safetensors metadata, which module it is and the config that module is
built from. The file needs nothing else to be loaded, and nothing reads ONNX at runtime.

A file that is not what the module expects is refused with what was expected and what was
found. There is no fallback: a model that quietly half-loads is a model nobody can debug.
"""
import json

import torch

from . import registry

# bump when a module's config or parameter layout changes incompatibly; files written by
# another format version are refused rather than guessed at
FORMAT_VERSION = 1


def build(architecture, config, device="meta"):
    """The module for `architecture` from its config, with empty parameters on `device`
    (the meta device by default, so building costs no memory until the weights arrive)."""
    if architecture not in registry.names("architecture"):
        raise ValueError(f"unknown model architecture {architecture!r}; expected one of {sorted(registry.names('architecture'))}")
    with torch.device(device):
        return registry.get("architecture", architecture).implementation(config).eval()


def save(net, architecture, path, dtype, extra=None):
    """Writes `net`'s state dict, floating tensors cast to `dtype`, with the architecture
    and the config the net was built from in the metadata. `extra` is more metadata
    (string values), e.g. where the weights came from."""
    from safetensors.torch import save_file
    if type(net) is not registry.get("architecture", architecture).implementation:
        raise TypeError(f"{type(net).__name__} is not the {architecture} module ({registry.get('architecture', architecture).implementation.__name__})")
    tensors = {name: (t.to(dtype) if t.is_floating_point() else t).contiguous().cpu()
               for name, t in net.state_dict().items()}
    metadata = {"format_version": str(FORMAT_VERSION), "architecture": architecture,
                "config": json.dumps(net.config), "dtype": str(dtype).replace("torch.", "")}
    metadata.update(extra or {})
    save_file(tensors, path, metadata=metadata)


def read_metadata(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        return f.metadata() or {}


def load(path, architecture, device="cpu"):
    """The module stored in `path`, which has to be an `architecture` file, with its
    weights on `device` in the file's own precision."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device=str(device)) as f:
        metadata = f.metadata() or {}
        found = metadata.get("architecture")
        if found != architecture:
            raise ValueError(f"{path}: expected a {architecture} model file, found "
                             f"{'no architecture in its metadata' if found is None else repr(found)}")
        version = metadata.get("format_version")
        if version != str(FORMAT_VERSION):
            raise ValueError(f"{path}: expected model file format {FORMAT_VERSION}, found {version!r}; "
                             f"convert the model again with scripts/convert_models.py")
        if "config" not in metadata:
            raise ValueError(f"{path}: expected the model config in the file metadata, found none")
        net = build(architecture, json.loads(metadata["config"]))
        state = {name: f.get_tensor(name) for name in f.keys()}
    check_state(net, state, path)
    net.load_state_dict(state, strict=True, assign=True)
    return net


def check_state(net, state, source):
    """Raises when `state` does not have exactly the tensors `net` expects, in shape and
    in one floating precision, listing every difference."""
    expected = {name: tuple(t.shape) for name, t in net.state_dict().items()}
    found = {name: tuple(t.shape) for name, t in state.items()}
    problems = []
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    if missing:
        problems.append(f"missing {len(missing)} tensors: {', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}")
    if unexpected:
        problems.append(f"{len(unexpected)} unexpected tensors: {', '.join(unexpected[:8])}"
                        f"{' ...' if len(unexpected) > 8 else ''}")
    for name in sorted(set(expected) & set(found)):
        if expected[name] != found[name]:
            problems.append(f"{name}: expected shape {list(expected[name])}, found {list(found[name])}")
    dtypes = sorted({str(t.dtype) for t in state.values() if t.is_floating_point()})
    if len(dtypes) > 1:
        problems.append(f"expected one floating precision, found {', '.join(dtypes)}")
    if problems:
        raise ValueError(f"{source} does not match the {type(net).__name__} its config describes:\n  "
                         + "\n  ".join(problems))
