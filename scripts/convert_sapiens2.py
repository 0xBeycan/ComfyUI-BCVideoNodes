"""Convert a transformers Sapiens2 pose checkpoint (HF facebook/sapiens2-pose-<size>: config.json +
model.safetensors, fp32) to the model file Sapiens2 Pose loads: the native module's state dict in bf16,
with the pack's model-file metadata (format_version, architecture, config, dtype;
models/common/checkpoint.py) plus where it came from.

Offline only (needs huggingface_hub to fetch; transformers is not needed). The file written is loaded
back the way the node loads it (models/sapiens2/wrapper.build, strict) before the script reports it.
The files the node downloads are in beycanai/sapiens2-convrot.

    PYTHONPATH=/path/to/ComfyUI python scripts/convert_sapiens2.py --repo facebook/sapiens2-pose-1b \\
        --out /path/to/out [--revision <commit>]
    python scripts/convert_sapiens2.py --print-quant --repo facebook/sapiens2-pose-5b

Writes <out>/sapiens2_pose_<size>_bf16.safetensors (--dtype fp32: ..._fp32.safetensors). The int8 ConvRot
file is made from one of them by convert_to_quant; --print-quant prints the command the shipped file of that
size was made with (QUANT_RECIPES). The full recipe is in the README of beycanai/sapiens2-convrot.

Key remap (HF -> ours; names follow the official sapiens2 backbone where it has them):
  model.embeddings.cls_token                    -> cls_token
  model.embeddings.register_tokens              -> storage_tokens
  model.embeddings.patch_embeddings.*           -> patch_embed.*
  model.model.layer.<i>.norm1.weight            -> blocks.<i>.ln1.weight
  model.model.layer.<i>.attention.{q,k,v,o}_proj.* -> blocks.<i>.attn.{wq,wk,wv,proj}.*
  model.model.layer.<i>.attention.{q,k}_norm.weight -> blocks.<i>.attn.{q,k}_norm.weight
  model.model.layer.<i>.layer_scale1.lambda1    -> blocks.<i>.attn.gamma
  model.model.layer.<i>.norm2.weight            -> blocks.<i>.ln2.weight
  model.model.layer.<i>.mlp.gate_proj.* + up_proj.* -> blocks.<i>.ffn.w12.* (gate rows first, then up)
  model.model.layer.<i>.mlp.down_proj.*         -> blocks.<i>.ffn.w3.*
  model.norm.weight                             -> ln_final.weight
  decode_head.upsample_layers.<j>.convolution.* -> head.upsample.<j>.*
  decode_head.conv_layers.<j>.convolution.*     -> head.conv.<j>.*
  decode_head.predictor.*                       -> head.predictor.*
Any other key is refused. state_to_hf is the exact inverse (tests check the round trip).
"""
import argparse
import json
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The pack, bound as the package `bcvideonodes` the way ComfyUI and tests/conftest.py bind it, so
# the relative imports of its modules resolve; its __init__ (the nodes) is not run. A process that
# has bound it already keeps that binding.
if "bcvideonodes" not in sys.modules:
    _pack = types.ModuleType("bcvideonodes")
    _pack.__path__ = [ROOT]
    sys.modules["bcvideonodes"] = _pack

CONVERTER_VERSION = "1"
# convert_to_quant --exclude-layers, the same for every size: the patch embedding, the tokens, every norm
# and the head; every Linear of every transformer block is quantized
EXCLUDE_LAYERS = r"^patch_embed\.|^(cls|storage)_tokens|\.(ln1|ln2|q_norm|k_norm)\.|^ln_final\.|^head\."
# size -> (the source file ctq quantizes: this script's --dtype, the ConvRot group size). 5b's width 2432
# does not divide by 256, so it uses group 64 (2432 = 38 x 64); an fp32-source file is afterwards set to
# compute and store like a bf16-source one (orig_dtype bfloat16, non-scale floats in bf16; see the HF card)
QUANT_RECIPES = {"0.4b": ("fp32", 256), "0.8b": ("bf16", 256), "1b": ("bf16", 256), "5b": ("fp32", 64)}

# (HF name, our name); {i} is a layer / stage index, {p} is weight or bias
_RENAMES = [
    ("model.embeddings.cls_token", "cls_token"),
    ("model.embeddings.register_tokens", "storage_tokens"),
    ("model.embeddings.patch_embeddings.{p}", "patch_embed.{p}"),
    ("model.model.layer.{i}.norm1.weight", "blocks.{i}.ln1.weight"),
    ("model.model.layer.{i}.attention.q_proj.{p}", "blocks.{i}.attn.wq.{p}"),
    ("model.model.layer.{i}.attention.k_proj.{p}", "blocks.{i}.attn.wk.{p}"),
    ("model.model.layer.{i}.attention.v_proj.{p}", "blocks.{i}.attn.wv.{p}"),
    ("model.model.layer.{i}.attention.o_proj.{p}", "blocks.{i}.attn.proj.{p}"),
    ("model.model.layer.{i}.attention.q_norm.weight", "blocks.{i}.attn.q_norm.weight"),
    ("model.model.layer.{i}.attention.k_norm.weight", "blocks.{i}.attn.k_norm.weight"),
    ("model.model.layer.{i}.layer_scale1.lambda1", "blocks.{i}.attn.gamma"),
    ("model.model.layer.{i}.norm2.weight", "blocks.{i}.ln2.weight"),
    ("model.model.layer.{i}.mlp.down_proj.{p}", "blocks.{i}.ffn.w3.{p}"),
    ("model.norm.weight", "ln_final.weight"),
    ("decode_head.upsample_layers.{i}.convolution.{p}", "head.upsample.{i}.{p}"),
    ("decode_head.conv_layers.{i}.convolution.{p}", "head.conv.{i}.{p}"),
    ("decode_head.predictor.{p}", "head.predictor.{p}"),
]
# the fused SwiGLU input projection: HF gate_proj and up_proj, ours w12 = gate rows then up rows
_W12 = ("model.model.layer.{i}.mlp.{part}_proj.{p}", "blocks.{i}.ffn.w12.{p}")


def _regex(template):
    return re.compile(re.escape(template).replace(r"\{i\}", r"(?P<i>\d+)").replace(r"\{p\}", r"(?P<p>weight|bias)")
                      .replace(r"\{part\}", r"(?P<part>gate|up)"))


def _rename(key, pairs):
    """`key` renamed by the first (source, target) template pair it matches, None when none does."""
    for source, target in pairs:
        m = _regex(source).fullmatch(key)
        if m:
            return target.format(**m.groupdict())
    return None


def config_from_hf(hf):
    """Our module config from a transformers Sapiens2 pose config.json dict; refuses what the module does
    not implement."""
    head = hf["head_config"]
    unsupported = {
        "hidden_act": (hf.get("hidden_act"), "silu"), "use_gated_mlp": (hf.get("use_gated_mlp"), True),
        "use_qk_norm": (hf.get("use_qk_norm"), True), "use_mask_token": (bool(hf.get("use_mask_token")), False),
        "head_config.use_pixel_shuffle": (bool(head.get("use_pixel_shuffle")), False),
    }
    wrong = {k: v for k, (v, want) in unsupported.items() if v != want}
    if wrong:
        raise ValueError(f"this Sapiens2 config uses what the native module does not implement: {wrong}")
    layers, heads = hf["num_hidden_layers"], hf["num_attention_heads"]
    kv = hf.get("num_key_value_heads_per_layer")
    if kv is None:  # HF Sapiens2Config.__post_init__ rule
        first, last = hf.get("num_first_full_attention_layers", 8), hf.get("num_last_full_attention_layers", 8)
        kv = [heads if i < first or i >= layers - last else hf["num_key_value_attention_heads"] for i in range(layers)]
    image = hf["image_size"] if isinstance(hf["image_size"], (list, tuple)) else [hf["image_size"]] * 2
    labels = hf.get("num_labels") or len(hf.get("id2label") or {})
    if not labels:
        raise ValueError("the config gives no keypoint count (num_labels / id2label)")
    return {
        "input_size": list(image), "patch_size": hf["patch_size"], "num_channels": hf.get("num_channels", 3),
        "hidden_size": hf["hidden_size"], "num_layers": layers, "num_heads": heads, "kv_heads": list(kv),
        "intermediate_size": hf["intermediate_size"], "num_register_tokens": hf["num_register_tokens"],
        "rope_theta": hf["rope_theta"], "rms_norm_eps": hf["rms_norm_eps"],
        "qkv_bias": [hf["query_bias"], hf["key_bias"], hf["value_bias"]], "proj_bias": hf["proj_bias"],
        "mlp_bias": hf["mlp_bias"], "num_keypoints": labels,
        "head": {"upsample_channels": list(head["upsample_out_channels"]),
                 "upsample_kernels": list(head["upsample_kernel_sizes"]),
                 "conv_channels": list(head["conv_out_channels"]), "conv_kernels": list(head["conv_kernel_sizes"])},
    }


def state_from_hf(hf_state):
    """Our state dict from a transformers Sapiens2ForPoseEstimation state dict (see the module docstring)."""
    import torch
    out, gates, unmapped = {}, {}, []
    for key, value in hf_state.items():
        m = _regex(_W12[0]).fullmatch(key)
        if m:
            gates.setdefault((m["i"], m["p"]), {})[m["part"]] = value
            continue
        name = _rename(key, _RENAMES)
        if name is None:
            unmapped.append(key)
        else:
            out[name] = value
    if unmapped:
        raise ValueError(f"{len(unmapped)} HF tensors have no place in the native module: {unmapped[:8]}")
    for (layer, kind), parts in gates.items():
        if set(parts) != {"gate", "up"}:
            raise ValueError(f"layer {layer} mlp {kind}: expected gate_proj and up_proj, found {sorted(parts)}")
        out[_W12[1].format(i=layer, p=kind)] = torch.cat((parts["gate"], parts["up"]), dim=0)
    return out


# The repos' model.safetensors are in the official sapiens2 layout (backbone.* / decode_head.deconv_layers.*),
# which transformers renames on load (conversion_mapping.py, "sapiens2" + "Sapiens2ForPoseEstimation").
# The same renames, official -> transformers names, so state_from_hf takes either file.
_OFFICIAL = [
    (r"^backbone\.cls_token$", "model.embeddings.cls_token"),
    (r"^backbone\.storage_tokens$", "model.embeddings.register_tokens"),
    (r"^backbone\.patch_embed\.projection\.(weight|bias)$", r"model.embeddings.patch_embeddings.\1"),
    (r"^backbone\.ln1\.weight$", "model.norm.weight"),
    (r"^backbone\.blocks\.(\d+)\.ln1\.weight$", r"model.model.layer.\1.norm1.weight"),
    (r"^backbone\.blocks\.(\d+)\.ln2\.weight$", r"model.model.layer.\1.norm2.weight"),
    (r"^backbone\.blocks\.(\d+)\.attn\.proj\.(weight|bias)$", r"model.model.layer.\1.attention.o_proj.\2"),
    (r"^backbone\.blocks\.(\d+)\.attn\.w(q|k|v)\.(weight|bias)$", r"model.model.layer.\1.attention.\2_proj.\3"),
    (r"^backbone\.blocks\.(\d+)\.attn\.(q|k)_norm\.weight$", r"model.model.layer.\1.attention.\2_norm.weight"),
    (r"^backbone\.blocks\.(\d+)\.attn\.gamma\.weight$", r"model.model.layer.\1.layer_scale1.lambda1"),
    (r"^backbone\.blocks\.(\d+)\.ffn\.w3\.(weight|bias)$", r"model.model.layer.\1.mlp.down_proj.\2"),
    (r"^decode_head\.conv_pose\.(weight|bias)$", r"decode_head.predictor.\1"),
]
# transformers ignores the rope periods buffer too (inv_freq is computed from the config)
_OFFICIAL_DROPPED = re.compile(r"^backbone\.rope_embed\.periods$")


def hf_from_official(state):
    """transformers-named state dict from an official-layout sapiens2 pose state dict (the HF repos' file)."""
    import torch
    out, unknown = {}, []
    for key, value in state.items():
        if _OFFICIAL_DROPPED.fullmatch(key):
            continue
        m = re.fullmatch(r"backbone\.blocks\.(\d+)\.ffn\.w12\.(weight|bias)", key)
        if m:  # fused gate + up rows, as transformers' Chunk(dim=0)
            gate, up = value.chunk(2, dim=0)
            out[f"model.model.layer.{m[1]}.mlp.gate_proj.{m[2]}"] = gate
            out[f"model.model.layer.{m[1]}.mlp.up_proj.{m[2]}"] = up
            continue
        # head: the Sequential index of every conv is 3 * its place (conv, norm, activation)
        m = re.fullmatch(r"decode_head\.(deconv|conv)_layers\.(\d+)\.(weight|bias)", key)
        if m:
            index = int(m[2])
            if index % 3:
                unknown.append(key)
                continue
            kind = "upsample_layers" if m[1] == "deconv" else "conv_layers"
            out[f"decode_head.{kind}.{index // 3}.convolution.{m[3]}"] = value
            continue
        for source, target in _OFFICIAL:
            if re.fullmatch(source, key):
                out[re.sub(source, target, key)] = value
                break
        else:
            unknown.append(key)
    if unknown:
        raise ValueError(f"{len(unknown)} tensors of the official layout have no transformers name: {unknown[:8]}")
    return out


def load_hf_state(path):
    """The repo's model.safetensors as a transformers-named state dict, whichever layout the file uses."""
    from safetensors.torch import load_file
    state = load_file(path)
    if any(k.startswith("backbone.") for k in state):
        state = hf_from_official(state)
    return state


def state_to_hf(state):
    """The inverse of state_from_hf."""
    out = {}
    for key, value in state.items():
        m = _regex(_W12[1]).fullmatch(key)
        if m:
            for part, half in zip(("gate", "up"), value.chunk(2, dim=0)):
                out[_W12[0].format(i=m["i"], part=part, p=m["p"])] = half
            continue
        name = _rename(key, [(ours, hf) for hf, ours in _RENAMES])
        if name is None:
            raise ValueError(f"{key} has no HF name")
        out[name] = value
    return out


def quant_command(size):
    """The convert_to_quant (ctq 1.3.4) command the shipped int8 ConvRot file of `size` was made with;
    the full recipe, and why each size has its source and group, is in beycanai/sapiens2-convrot's README."""
    source, group = QUANT_RECIPES[size]
    return (f"ctq -i sapiens2_pose_{size}_{source}.safetensors -o sapiens2_pose_{size}_int8_convrot.safetensors "
            f"--int8 --convrot --convrot-group-size {group} --scaling_mode row --comfy_quant --save-quant-metadata "
            f"--simple --low-memory --exclude-layers '{EXCLUDE_LAYERS}'")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, help="HF repo, e.g. facebook/sapiens2-pose-1b")
    p.add_argument("--revision", default=None)
    p.add_argument("--out", default=".")
    p.add_argument("--size", default=None, help="size in the file name (default: from the repo name)")
    p.add_argument("--print-quant", action="store_true", help="print the convert_to_quant command of this size and stop")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="precision of the file written (fp32: the source for an int8 quantization from full precision)")
    args = p.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    size = args.size or args.repo.rsplit("-", 1)[-1]
    if args.print_quant:
        print(quant_command(size))
        return
    config_path = hf_hub_download(args.repo, "config.json", revision=args.revision)
    with open(config_path) as f:
        config = config_from_hf(json.load(f))

    import torch
    from safetensors.torch import load_file, save_file

    from bcvideonodes.models.common import checkpoint
    from bcvideonodes.models.sapiens2 import wrapper

    revision = args.revision or HfApi().model_info(args.repo).sha
    weights_path = hf_hub_download(args.repo, "model.safetensors", revision=revision)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    state = {k: v.to(dtype).contiguous() for k, v in state_from_hf(load_hf_state(weights_path)).items()}
    path = os.path.join(args.out, f"sapiens2_pose_{size}_{args.dtype}.safetensors")
    os.makedirs(args.out, exist_ok=True)
    save_file(state, path, metadata={
        "format_version": str(checkpoint.FORMAT_VERSION), "architecture": wrapper.ARCHITECTURE,
        "config": json.dumps(config), "dtype": str(dtype).replace("torch.", ""), "source": f"{args.repo}@{revision} model.safetensors",
        "converter_version": CONVERTER_VERSION})
    # load it back the way the node does (strict: every tensor has its place)
    back = load_file(path)
    # the device the operations are picked for: core refuses a cpu device on an NVIDIA host
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    config = checkpoint.model_config(checkpoint.read_metadata(path), path, wrapper.ARCHITECTURE, wrapper.CONVERTER)
    wrapper.build(back, config, device, source=path)
    print(f"wrote {path} ({os.path.getsize(path) / 2**30:.2f} GiB, {len(state)} tensors)")
    if size in QUANT_RECIPES:
        print(quant_command(size))


if __name__ == "__main__":
    main()
