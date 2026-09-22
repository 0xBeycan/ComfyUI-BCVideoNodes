"""Architecture and weights out of the upstream ONNX exports, for the offline conversion.

Each extractor walks one export node by node, matches it against the structure its native
module implements, and returns (config, state): the config dict the module is built from
and its state dict, keyed by the module's own parameter names. Every edge is checked by
tensor name, so an export that is not the model stops the walk with the node where it
diverged instead of producing a module with the right shapes and the wrong wiring.

This is the only code that reads ONNX. It runs offline (scripts/convert_models.py); the
nodes load the safetensors files it produces and never import onnx.
"""
import os

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper


class Mismatch(Exception):
    """The export stopped looking like the model being extracted."""


# -- reading the graph ---------------------------------------------------------------------

class Node:
    __slots__ = ("op", "name", "inputs", "outputs", "attrs")

    def __init__(self, proto):
        self.op = proto.op_type
        self.name = proto.name
        self.inputs = list(proto.input)
        self.outputs = list(proto.output)
        self.attrs = {a.name: _attribute(a) for a in proto.attribute}


def _attribute(a):
    v = onnx.helper.get_attribute_value(a)
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, TensorProto):
        return torch.from_numpy(numpy_helper.to_array(v).copy())
    if isinstance(v, list) and v and isinstance(v[0], bytes):
        return [s.decode() for s in v]
    return v


def _read_tensor(t, base_dir):
    """Initializer -> numpy, reading external data straight from its file."""
    if t.data_location == TensorProto.EXTERNAL:
        info = {e.key: e.value for e in t.external_data}
        dtype = onnx.helper.tensor_dtype_to_np_dtype(t.data_type)
        count = int(np.prod(t.dims)) if len(t.dims) else 1
        path = os.path.join(base_dir, info["location"])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"the ONNX model stores its weights in '{info['location']}', "
                                    f"which is missing next to the .onnx file (looked in {base_dir})")
        arr = np.fromfile(path, dtype=dtype, count=count, offset=int(info.get("offset", 0)))
        if arr.size != count:
            raise ValueError(f"{path}: expected {count} values for {t.name}, read {arr.size}")
        return arr.reshape(tuple(t.dims))
    return numpy_helper.to_array(t)


def _constant_value(n):
    attrs = {a.name: _attribute(a) for a in n.attribute}
    if "value" in attrs:
        return attrs["value"]
    for key in ("value_float", "value_int", "value_floats", "value_ints"):
        if key in attrs:
            return torch.tensor(attrs[key])
    raise ValueError(f"Constant node {n.name}: unsupported value attribute {list(attrs)}")


class OnnxGraph:
    """A parsed ONNX model: constant tensors (CPU torch tensors), nodes in exported order
    and the input / output names."""

    def __init__(self, path):
        model = onnx.load(path, load_external_data=False)
        base_dir = os.path.dirname(os.path.abspath(path))
        self.path = path
        self.tensors = {}
        for t in model.graph.initializer:
            arr = _read_tensor(t, base_dir)
            if not arr.flags.writeable:  # in-file raw data comes back as a read-only view
                arr = arr.copy()
            self.tensors[t.name] = torch.from_numpy(np.ascontiguousarray(arr))
        self.nodes = []
        for n in model.graph.node:
            if n.op_type == "Constant":
                self.tensors[n.output[0]] = _constant_value(n)
            else:
                self.nodes.append(Node(n))
        self.inputs = [i.name for i in model.graph.input if i.name not in self.tensors]
        self.outputs = [o.name for o in model.graph.output]
        self.input_shapes = {
            i.name: [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]
            for i in model.graph.input if i.name in self.inputs
        }

    def producer(self, name):
        for n in self.nodes:
            if name in n.outputs:
                return n
        return None

    def consumers(self, name):
        return [n for n in self.nodes if name in n.inputs]

    def input_size(self):
        """(height, width) of the one image input, which has to be fixed in the export."""
        if len(self.inputs) != 1:
            raise Mismatch(f"expected one input, found {len(self.inputs)}")
        shape = self.input_shapes[self.inputs[0]]
        if len(shape) != 4 or not all(isinstance(v, int) and v > 0 for v in shape[2:]) or shape[1] != 3:
            raise Mismatch(f"expected an [N, 3, H, W] input with fixed H and W, found {shape}")
        return [shape[2], shape[3]]


# -- walking it ----------------------------------------------------------------------------

class Walk:
    """A cursor over the graph's nodes in their exported order. Every helper below consumes
    the nodes it expects and raises Mismatch the moment one is missing or reads the wrong
    tensor."""

    COMMUTATIVE = ("Add", "Mul")

    def __init__(self, graph):
        self.nodes = graph.nodes
        self.tensors = graph.tensors
        self.i = 0

    def peek(self, ahead=0):
        i = self.i + ahead
        return self.nodes[i].op if i < len(self.nodes) else None

    def upcoming(self, op, within):
        return any(n.op == op for n in self.nodes[self.i:self.i + within])

    def next_node(self, op):
        """The next `op` node at or after the cursor, without consuming anything."""
        return next((n for n in self.nodes[self.i:] if n.op == op), None)

    def take(self, op, *sources):
        """The next node, which has to be `op` and has to read exactly `sources` - the
        inputs that are not initializers, in order, except for the commutative ops."""
        if self.i >= len(self.nodes):
            raise Mismatch(f"expected {op} reading {list(sources)}, found the end of the graph")
        node = self.nodes[self.i]
        if node.op != op:
            raise Mismatch(f"node {self.i} ({node.name}): expected {op}, found {node.op}")
        self.i += 1
        data = [n for n in node.inputs if n and n not in self.tensors]
        expected = list(sources)
        if op in self.COMMUTATIVE:
            data, expected = sorted(data), sorted(expected)
        if data != expected:
            raise Mismatch(f"node {self.i - 1} ({node.name}): expected {op} reading {expected}, found {data}")
        return node

    def weight(self, node, idx):
        if idx >= len(node.inputs) or node.inputs[idx] not in self.tensors:
            raise Mismatch(f"{node.name}: expected a constant as input {idx}")
        return self.tensors[node.inputs[idx]]

    def const(self, node):
        """The node's one constant input, for the scalars the exporter folds into Mul/Div."""
        consts = [self.tensors[n] for n in node.inputs if n and n in self.tensors]
        if len(consts) != 1:
            raise Mismatch(f"{node.name}: expected one constant input, found {len(consts)}")
        return consts[0]

    def scalar(self, node):
        value = self.const(node)
        if value.numel() != 1:
            raise Mismatch(f"{node.name}: expected a scalar constant, found shape {list(value.shape)}")
        return float(value)

    def ints(self, node, idx):
        return [int(v) for v in self.weight(node, idx).reshape(-1).tolist()]


def _nest(prefix, params):
    return {f"{prefix}.{k}": v for k, v in params.items()}


def _symmetric(pads):
    half = len(pads) // 2
    if pads[:half] != pads[half:]:
        raise Mismatch(f"expected symmetric padding, found {pads}")
    return list(pads[:half])


def conv(walk, src, act=None):
    """A Conv and the activation the export spells out after it: Sigmoid/Mul for SiLU, or a
    plain Relu. Returns its config, its parameters and the tensor it leaves behind."""
    node = walk.take("Conv", src)
    w = walk.weight(node, 1)
    if w.dim() != 4 or node.attrs.get("auto_pad", "NOTSET") != "NOTSET":
        raise Mismatch(f"{node.name}: expected a 2-D convolution with explicit pads")
    has_bias = len(node.inputs) > 2 and bool(node.inputs[2])
    groups = node.attrs.get("group", 1)
    cfg = {"cin": int(w.shape[1]) * groups, "cout": int(w.shape[0]), "kernel": [int(k) for k in w.shape[2:]],
           "stride": list(node.attrs.get("strides", [1, 1])),
           "padding": _symmetric(node.attrs.get("pads", [0, 0, 0, 0])),
           "dilation": list(node.attrs.get("dilations", [1, 1])), "groups": groups, "bias": has_bias, "act": act}
    params = {"weight": w}
    if has_bias:
        params["bias"] = walk.weight(node, 2)
    out = node.outputs[0]
    if act == "silu":
        sigmoid = walk.take("Sigmoid", out)
        out = walk.take("Mul", out, sigmoid.outputs[0]).outputs[0]
    elif act == "relu":
        out = walk.take("Relu", out).outputs[0]
    return cfg, params, out


def concat(walk, axis, *sources):
    node = walk.take("Concat", *sources)
    if node.attrs.get("axis") != axis:
        raise Mismatch(f"{node.name}: expected a concat on axis {axis}, found {node.attrs.get('axis')}")
    return node.outputs[0]


def upsample(walk, src):
    """The export's nearest Resize by a constant scale. It is a plain F.interpolate only
    because the mode is asymmetric/floor on an integer factor, which is what is checked."""
    node = walk.take("Resize", src)
    if (node.attrs.get("mode", "nearest") != "nearest" or len(node.inputs) != 3
            or node.attrs.get("coordinate_transformation_mode") != "asymmetric"
            or node.attrs.get("nearest_mode", "round_prefer_floor") != "floor"):
        raise Mismatch(f"{node.name}: expected a nearest asymmetric/floor Resize by scales")
    scales = [float(s) for s in walk.weight(node, 2).tolist()]
    if (len(scales) != 4 or scales[0] != 1.0 or scales[1] != 1.0 or scales[2] != scales[3]
            or scales[2] != int(scales[2])):
        raise Mismatch(f"{node.name}: expected an integer spatial scale, found {scales}")
    return scales[2], node.outputs[0]


def maxpool(walk, src):
    node = walk.take("MaxPool", src)
    kernel = node.attrs.get("kernel_shape", [])
    if (len(kernel) != 2 or kernel[0] != kernel[1] or node.attrs.get("strides", [1, 1]) != [1, 1]
            or _symmetric(node.attrs.get("pads", [0, 0, 0, 0])) != [kernel[0] // 2] * 2
            or node.attrs.get("ceil_mode", 0) or node.attrs.get("dilations", [1, 1]) != [1, 1]):
        raise Mismatch(f"{node.name}: expected a square stride-1 'same' max pool")
    return kernel[0], node.outputs[0]


def spp(walk, src):
    """SPP (RTMW: every pool reads the same map) or SPPF (YOLO: each pool reads the last)."""
    c1, p1, x = conv(walk, src, "silu")
    kernels, pooled = [], [x]
    cascade = None
    while walk.peek() == "MaxPool":
        node = walk.nodes[walk.i]
        reads_last = node.inputs[0] == pooled[-1] and len(pooled) > 1
        if len(pooled) > 1:
            if cascade is None:
                cascade = reads_last
            elif cascade != reads_last:
                raise Mismatch(f"{node.name}: the pools mix reading the map and reading the last pool")
        k, out = maxpool(walk, pooled[-1] if cascade else x)
        kernels.append(k)
        pooled.append(out)
    if not kernels:
        raise Mismatch("expected max pools after the SPP convolution")
    c2, p2, out = conv(walk, concat(walk, 1, *pooled), "silu")
    cfg = {"conv1": c1, "conv2": c2, "kernels": kernels, "cascade": bool(cascade)}
    return cfg, {**_nest("conv1", p1), **_nest("conv2", p2)}, out


# -- ViTPose -------------------------------------------------------------------------------

def _shape_values(graph, name):
    """Constant entries of a shape tensor (None for runtime dims), through a Concat if needed."""
    if name in graph.tensors:
        return [int(v) for v in graph.tensors[name].reshape(-1).tolist()]
    node = graph.producer(name)
    if node is None or node.op != "Concat":
        return []
    values = []
    for inp in node.inputs:
        if inp in graph.tensors:
            values += [int(v) for v in graph.tensors[inp].reshape(-1).tolist()]
        else:
            values.append(None)
    return values


def _plain_symmetric(pads):
    half = len(pads) // 2
    return list(pads[:half]) if pads[:half] == pads[half:] else None


def extract_vitpose(graph):
    """(config, state) of a ViTPose export: depth and widths from the weight shapes, the
    head count from the qkv reshape, the attention scale, LayerNorm epsilon and the patch
    embedding / deconvolution hyper-parameters from the node attributes."""
    t = graph.tensors
    depth = 0
    while f"backbone.blocks.{depth}.norm1.weight" in t:
        depth += 1
    if depth == 0 or "backbone.patch_embed.proj.weight" not in t or "backbone.last_norm.weight" not in t:
        raise Mismatch("no ViTPose backbone tensors (backbone.blocks.N / patch_embed / last_norm)")
    dim = int(t["backbone.last_norm.weight"].shape[0])

    # MatMul weights are anonymous; the Add that applies the named bias identifies each Linear
    linears = {}
    for node in graph.nodes:
        if node.op != "MatMul" or node.inputs[1] not in t:
            continue
        for add in graph.consumers(node.outputs[0]):
            bias = next((n for n in add.inputs if n in t and n.endswith(".bias")), None)
            if add.op == "Add" and bias:
                linears[bias[:-len(".bias")]] = (t[node.inputs[1]], t[bias])
    names = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")
    missing = [f"backbone.blocks.{i}.{n}" for i in range(depth) for n in names if f"backbone.blocks.{i}.{n}" not in linears]
    if missing:
        raise Mismatch(f"linear layers not found: {missing[:4]}")

    # patch embedding: the Conv on the patch weight
    conv_node = next((n for n in graph.nodes if n.op == "Conv" and n.inputs[1] == "backbone.patch_embed.proj.weight"), None)
    if conv_node is None:
        raise Mismatch("no patch embedding Conv")
    pads = _plain_symmetric(conv_node.attrs.get("pads", [0, 0, 0, 0]))
    if pads is None or conv_node.attrs.get("group", 1) != 1:
        raise Mismatch("patch embedding with asymmetric pads or groups")
    w = t["backbone.patch_embed.proj.weight"]
    patch_embed = {"cin": int(w.shape[1]), "cout": int(w.shape[0]), "kernel": [int(k) for k in w.shape[2:]],
                   "stride": list(conv_node.attrs.get("strides", [1, 1])), "padding": pads,
                   "dilation": list(conv_node.attrs.get("dilations", [1, 1])), "groups": 1, "bias": True, "act": None}

    # position embedding: the constant [1, N, C] / [1, 1, C] terms added to the tokens
    pos_terms = [t[n] for node in graph.nodes if node.op == "Add"
                 for n in node.inputs if n in t and t[n].dim() == 3 and t[n].shape[-1] == dim]
    if not pos_terms:
        raise Mismatch("no position embedding")
    pos_embed = pos_terms[0]
    for term in pos_terms[1:]:
        pos_embed = pos_embed + term

    # heads from the qkv reshape [B, N, 3, heads, head_dim], scale from the q multiply
    heads, scale = None, None
    qkv_add = next(n for n in graph.nodes if n.op == "Add" and "backbone.blocks.0.attn.qkv.bias" in n.inputs)
    for reshape in graph.consumers(qkv_add.outputs[0]):
        if reshape.op != "Reshape":
            continue
        shape = _shape_values(graph, reshape.inputs[1])
        if 3 in shape and shape.index(3) + 1 < len(shape):
            heads = shape[shape.index(3) + 1]
    for node in graph.nodes:
        if node.op == "Mul" and node.name.startswith("/backbone/blocks.0/attn/"):
            const = next((t[n] for n in node.inputs if n in t and t[n].numel() == 1), None)
            if const is not None:
                scale = float(const)
                break
    if heads is None or dim % heads:
        raise Mismatch(f"no head count that divides the width {dim} (found {heads})")

    eps = 1e-6
    sqrt = next((n for n in graph.nodes if n.op == "Sqrt"), None)
    if sqrt is not None:
        add = graph.producer(sqrt.inputs[0])
        if add is not None and add.op == "Add":
            eps = float(next(t[n] for n in add.inputs if n in t))
    gelu = "none" if any(n.op == "Erf" for n in graph.nodes) else "tanh"

    state = {"patch_embed.weight": w, "patch_embed.bias": t["backbone.patch_embed.proj.bias"],
             "pos_embed": pos_embed, "last_norm.weight": t["backbone.last_norm.weight"],
             "last_norm.bias": t["backbone.last_norm.bias"]}
    for i in range(depth):
        p = f"backbone.blocks.{i}."
        for norm in ("norm1", "norm2"):
            state[f"blocks.{i}.{norm}.weight"] = t[p + norm + ".weight"]
            state[f"blocks.{i}.{norm}.bias"] = t[p + norm + ".bias"]
        for src, dst in (("attn.qkv", "qkv"), ("attn.proj", "proj"), ("mlp.fc1", "fc1"), ("mlp.fc2", "fc2")):
            state[f"blocks.{i}.{dst}.weight"], state[f"blocks.{i}.{dst}.bias"] = linears[p + src]

    head, head_state = _vitpose_head(graph, t)
    state.update(_nest("head", head_state))
    config = {"input_size": graph.input_size(), "patch_embed": patch_embed, "num_tokens": int(pos_embed.shape[1]),
              "dim": dim, "depth": depth, "heads": heads, "scale": scale,
              "mlp_hidden": int(linears["backbone.blocks.0.mlp.fc1"][0].shape[1]), "eps": eps, "gelu": gelu,
              "head": head}
    return config, state


def _vitpose_head(graph, t):
    """The keypoint head: the ConvTranspose / BatchNormalization / Relu / Conv chain after the backbone."""
    start = next((i for i, n in enumerate(graph.nodes) if n.op == "ConvTranspose"), None)
    if start is None:
        raise Mismatch("no ConvTranspose head")
    layers, state = [], {}
    for node in graph.nodes[start:]:
        i = len(layers)
        if node.op in ("ConvTranspose", "Conv"):
            pads = _plain_symmetric(node.attrs.get("pads", [0, 0, 0, 0]))
            w = t.get(node.inputs[1])
            has_bias = len(node.inputs) > 2 and bool(node.inputs[2])
            if pads is None or w is None:
                raise Mismatch(f"{node.name}: asymmetric pads or a computed weight")
            if node.op == "ConvTranspose":
                if node.attrs.get("group", 1) != 1 or "output_shape" in node.attrs:
                    raise Mismatch(f"{node.name}: grouped or output_shape deconvolution")
                layer = {"type": "deconv", "cin": int(w.shape[0]), "cout": int(w.shape[1]),
                         "kernel": [int(k) for k in w.shape[2:]], "stride": list(node.attrs.get("strides", [1, 1])),
                         "padding": pads, "output_padding": list(node.attrs.get("output_padding", [0, 0])),
                         "dilation": list(node.attrs.get("dilations", [1, 1])), "bias": has_bias}
            else:
                groups = node.attrs.get("group", 1)
                layer = {"type": "conv", "cin": int(w.shape[1]) * groups, "cout": int(w.shape[0]),
                         "kernel": [int(k) for k in w.shape[2:]], "stride": list(node.attrs.get("strides", [1, 1])),
                         "padding": pads, "dilation": list(node.attrs.get("dilations", [1, 1])), "groups": groups,
                         "bias": has_bias, "act": None}
            state[f"{i}.weight"] = w
            if has_bias:
                state[f"{i}.bias"] = t[node.inputs[2]]
        elif node.op == "BatchNormalization":
            scale, bias, mean, var = (t.get(n) for n in node.inputs[1:5])
            if any(v is None for v in (scale, bias, mean, var)):
                raise Mismatch(f"{node.name}: batch norm with computed statistics")
            layer = {"type": "bn", "channels": int(scale.shape[0]), "eps": float(node.attrs.get("epsilon", 1e-5))}
            state.update({f"{i}.weight": scale, f"{i}.bias": bias, f"{i}.running_mean": mean,
                          f"{i}.running_var": var, f"{i}.num_batches_tracked": torch.tensor(0, dtype=torch.long)})
        elif node.op == "Relu":
            layer = {"type": "relu"}
        else:
            raise Mismatch(f"{node.name}: unexpected {node.op} in the keypoint head")
        layers.append(layer)
        if node.outputs[0] in graph.outputs:
            return layers, state
    raise Mismatch("the keypoint head never reaches the graph output")


# -- RTMW ----------------------------------------------------------------------------------

def _csp_block(walk, src):
    convs, params, x = [], {}, src
    for k in range(3):
        c, p, x = conv(walk, x, "silu")
        convs.append(c)
        params.update(_nest(f"convs.{k}", p))
    residual = walk.peek() == "Add"
    if residual:
        x = walk.take("Add", x, src).outputs[0]
    return {"convs": convs, "residual": residual}, params, x


def _channel_attention(walk, src):
    pool = walk.take("GlobalAveragePool", src)
    fc, p, out = conv(walk, pool.outputs[0])
    gate = walk.take("HardSigmoid", out)
    cfg = {"fc": fc, "alpha": float(gate.attrs.get("alpha", 0.2)), "beta": float(gate.attrs.get("beta", 0.5))}
    return cfg, _nest("fc", p), walk.take("Mul", src, gate.outputs[0]).outputs[0]


def _csp_layer(walk, src):
    """short_conv and main_conv both read the layer's input, so the blocks are however many
    convolutions run before the concat that joins the two halves back together."""
    short_conv, p_short, short = conv(walk, src, "silu")
    main_conv, p_main, x = conv(walk, src, "silu")
    params = {**_nest("short_conv", p_short), **_nest("main_conv", p_main)}
    blocks = []
    while walk.peek() == "Conv":
        block, p, x = _csp_block(walk, x)
        params.update(_nest(f"blocks.{len(blocks)}", p))
        blocks.append(block)
    out = concat(walk, 1, x, short)
    attention = None
    if walk.peek() == "GlobalAveragePool":
        attention, p, out = _channel_attention(walk, out)
        params.update(_nest("attention", p))
    final_conv, p, out = conv(walk, out, "silu")
    params.update(_nest("final_conv", p))
    cfg = {"type": "csp", "short_conv": short_conv, "main_conv": main_conv, "blocks": blocks,
           "attention": attention, "final_conv": final_conv}
    return cfg, params, out


def _flatten(walk, src):
    """[B, K, H, W] -> [B, K, H * W]. The export computes the shape with Shape/Slice/Concat
    because the batch is dynamic; the nodes still have to be consumed and checked."""
    shape = walk.take("Shape", src)
    sliced = walk.take("Slice", shape.outputs[0])
    if walk.weight(sliced, 1).tolist() != [0] or walk.weight(sliced, 2).tolist() != [2]:
        raise Mismatch(f"{sliced.name}: expected the first two dims of the shape")
    node = walk.take("Concat", sliced.outputs[0])
    if node.attrs.get("axis") != 0 or walk.const(node).tolist() != [-1]:
        raise Mismatch(f"{node.name}: expected [B, K, -1]")
    return walk.take("Reshape", src, node.outputs[0]).outputs[0]


def _scale_norm(walk, src):
    square = walk.take("Mul", src, src)
    total = walk.take("ReduceSum", square.outputs[0])
    if total.attrs.get("axes") != [2] or not total.attrs.get("keepdims", 1):
        raise Mismatch(f"{total.name}: expected a sum over axis 2 keeping dims")
    root = walk.take("Sqrt", total.outputs[0])
    scaled = walk.take("Mul", root.outputs[0])
    scale = walk.scalar(scaled)
    clip = walk.take("Clip", scaled.outputs[0])
    if len(clip.inputs) > 2 and clip.inputs[2]:
        raise Mismatch(f"{clip.name}: expected a lower bound only")
    eps = walk.scalar(clip)
    divided = walk.take("Div", src, clip.outputs[0])
    gained = walk.take("Mul", divided.outputs[0])
    return {"scale": scale, "gain": walk.scalar(gained), "eps": eps}, gained.outputs[0]


def _projection(walk, node):
    w = walk.weight(node, 1)
    if w.dim() != 2:
        raise Mismatch(f"{node.name}: expected a 2-D projection weight")
    return [int(v) for v in w.shape], {"weight": w}


def _gau(walk, src):
    norm, x = _scale_norm(walk, src)
    uv = walk.take("MatMul", x)
    sigmoid = walk.take("Sigmoid", uv.outputs[0])
    activated = walk.take("Mul", uv.outputs[0], sigmoid.outputs[0])
    split = walk.take("Split", activated.outputs[0])
    sizes = list(split.attrs.get("split", []))
    if split.attrs.get("axis") != 2 or len(sizes) != 3 or len(split.outputs) != 3:
        raise Mismatch(f"{split.name}: expected a three-way split on axis 2")
    u, v, base = split.outputs

    offset = walk.take("Unsqueeze", base)
    if offset.attrs.get("axes") != [2]:
        raise Mismatch(f"{offset.name}: expected an unsqueeze on axis 2")
    scaled = walk.take("Mul", offset.outputs[0])
    gamma = walk.const(scaled)
    shifted = walk.take("Add", scaled.outputs[0])
    beta = walk.const(shifted)
    heads = walk.take("Split", shifted.outputs[0])
    if heads.attrs.get("axis") != 2 or list(heads.attrs.get("split", [])) != [1, 1]:
        raise Mismatch(f"{heads.name}: expected a [1, 1] split on axis 2")
    q = walk.take("Squeeze", heads.outputs[0])
    k = walk.take("Squeeze", heads.outputs[1])
    if q.attrs.get("axes") != [2] or k.attrs.get("axes") != [2]:
        raise Mismatch(f"{q.name}: expected squeezes on axis 2")
    transposed = walk.take("Transpose", k.outputs[0])
    if transposed.attrs.get("perm") != [0, 2, 1]:
        raise Mismatch(f"{transposed.name}: expected perm [0, 2, 1]")

    qk = walk.take("MatMul", q.outputs[0], transposed.outputs[0])
    divided = walk.take("Div", qk.outputs[0])
    sqrt_s = walk.scalar(divided)
    rectified = walk.take("Relu", divided.outputs[0])
    squared = walk.take("Mul", rectified.outputs[0], rectified.outputs[0])
    weighted = walk.take("MatMul", squared.outputs[0], v)
    gated = walk.take("Mul", u, weighted.outputs[0])
    out = walk.take("MatMul", gated.outputs[0])
    residual = walk.take("Mul", src)
    res_scale = walk.const(residual)
    joined = walk.take("Add", residual.outputs[0], out.outputs[0])
    uv_shape, uv_params = _projection(walk, uv)
    out_shape, out_params = _projection(walk, out)
    cfg = {"norm": norm, "uv": uv_shape, "sizes": [int(s) for s in sizes], "gamma": list(gamma.shape),
           "beta": list(beta.shape), "out": out_shape, "res_scale": list(res_scale.shape), "sqrt_s": sqrt_s}
    params = {**_nest("uv", uv_params), "gamma": gamma, "beta": beta, **_nest("out", out_params),
              "res_scale": res_scale}
    return cfg, params, joined.outputs[0]


def _simcc_head(walk, p1, p2):
    final_layer, pf, x = conv(walk, p2, "relu")
    mlp_norm, x = _scale_norm(walk, _flatten(walk, x))
    mlp = walk.take("MatMul", x)

    shuffle = walk.take("DepthToSpace", p2)
    if shuffle.attrs.get("mode", "DCR") != "CRD":
        raise Mismatch(f"{shuffle.name}: expected a CRD pixel shuffle")
    mid_layer, pm, y = conv(walk, shuffle.outputs[0], "relu")
    final_layer2, pf2, y = conv(walk, concat(walk, 1, y, p1), "relu")
    mlp2_norm, y = _scale_norm(walk, _flatten(walk, y))
    mlp2 = walk.take("MatMul", y)

    gau, pg, z = _gau(walk, concat(walk, 2, mlp.outputs[0], mlp2.outputs[0]))
    cls_x = walk.take("MatMul", z)
    cls_y = walk.take("MatMul", z)
    cfg, params = {"final_layer": final_layer, "mlp_norm": mlp_norm, "mid_layer": mid_layer,
                   "final_layer2": final_layer2, "mlp2_norm": mlp2_norm, "gau": gau,
                   "blocksize": int(shuffle.attrs["blocksize"])}, {}
    for name, node in (("mlp", mlp), ("mlp2", mlp2), ("cls_x", cls_x), ("cls_y", cls_y)):
        cfg[name], p = _projection(walk, node)
        params.update(_nest(name, p))
    params.update({**_nest("final_layer", pf), **_nest("mid_layer", pm), **_nest("final_layer2", pf2),
                   **_nest("gau", pg)})
    return cfg, params, [cls_x.outputs[0], cls_y.outputs[0]]


def extract_rtmw(graph):
    """(config, state) of an RTMW export: a CSPNeXt backbone (stem, four stages of a CSP
    layer, an SPP bottleneck in the last), a CSPNeXt-PAFPN neck over three levels, and a
    SimCC head that ends in a gated attention unit."""
    if len(graph.inputs) != 1 or len(graph.outputs) != 2:
        raise Mismatch(f"expected one input and two outputs, found {len(graph.inputs)} and {len(graph.outputs)}")
    walk = Walk(graph)
    state = {}

    x = graph.inputs[0]
    stem = []
    for i in range(3):
        c, p, x = conv(walk, x, "silu")
        stem.append(c)
        state.update(_nest(f"stem.{i}", p))

    # four stages; the stage that has an SPP bottleneck has it between the downsample and
    # the CSP layer, which puts its first MaxPool three nodes ahead of the cursor
    stages, feats = [], []
    for i in range(4):
        downsample, p, x = conv(walk, x, "silu")
        state.update(_nest(f"stages.{i}.downsample", p))
        spp_cfg = None
        if walk.upcoming("MaxPool", 5):
            spp_cfg, p, x = spp(walk, x)
            if spp_cfg["cascade"]:
                raise Mismatch("RTMW's SPP pools the same map with every kernel; this one cascades")
            state.update(_nest(f"stages.{i}.spp", p))
        csp, p, x = _csp_layer(walk, x)
        csp.pop("type")
        state.update(_nest(f"stages.{i}.csp", p))
        stages.append({"downsample": downsample, "spp": spp_cfg, "csp": csp})
        feats.append(x)

    reduce2, p_r2, r2 = conv(walk, feats[3], "silu")
    scale, up = upsample(walk, r2)
    top_down0, p_t0, p1 = _csp_layer(walk, concat(walk, 1, up, feats[2]))
    reduce1, p_r1, r1 = conv(walk, p1, "silu")
    scale1, up1 = upsample(walk, r1)
    if scale1 != scale:
        raise Mismatch(f"the two upsamples scale by {scale} and {scale1}")
    top_down1, p_t1, p0 = _csp_layer(walk, concat(walk, 1, up1, feats[1]))
    down0, p_d0, d0 = conv(walk, p0, "silu")
    bottom_up0, p_b0, n1 = _csp_layer(walk, concat(walk, 1, d0, r1))
    down1, p_d1, d1 = conv(walk, n1, "silu")
    bottom_up1, p_b1, n2 = _csp_layer(walk, concat(walk, 1, d1, r2))
    neck = {"reduce": [reduce2, reduce1], "top_down": [top_down0, top_down1],
            "down": [{"type": "conv", **down0}, {"type": "conv", **down1}], "bottom_up": [bottom_up0, bottom_up1],
            "scale": scale}
    for name, p in (("reduce.0", p_r2), ("reduce.1", p_r1), ("top_down.0", p_t0), ("top_down.1", p_t1),
                    ("down.0", p_d0), ("down.1", p_d1), ("bottom_up.0", p_b0), ("bottom_up.1", p_b1)):
        state.update(_nest(f"neck.{name}", p))

    head, p, outputs = _simcc_head(walk, n1, n2)
    state.update(_nest("head", p))
    if walk.i != len(walk.nodes) or outputs != list(graph.outputs):
        raise Mismatch(f"the head ends at node {walk.i} of {len(walk.nodes)}, reading {outputs} "
                       f"where the graph returns {graph.outputs}")
    config = {"input_size": graph.input_size(), "stem": stem, "stages": stages, "neck": neck, "head": head}
    return config, state


# -- YOLOv10 -------------------------------------------------------------------------------

def _chain(walk, src, acts):
    """Convolutions with the given activations, one reading the next, no residual."""
    convs, params, x = [], {}, src
    for k, act in enumerate(acts):
        c, p, x = conv(walk, x, act)
        convs.append(c)
        params.update(_nest(f"convs.{k}", p))
    return {"type": "chain", "convs": convs, "residual": False}, params, x


def _split(walk, src, parts, axis=1):
    node = walk.take("Split", src)
    if node.attrs.get("axis") != axis or len(node.outputs) != parts:
        raise Mismatch(f"{node.name}: expected a {parts}-way split on axis {axis}")
    return walk.ints(node, 1), node.outputs


def _c2f(walk, src):
    """cv1, a two-way split, blocks one after the other on the second half, a concat of the
    two halves and every block's output, cv2. Where one block ends is read from the concat:
    a block closed by an Add with its input has a residual, one that is not does not."""
    cv1, p1, x = conv(walk, src, "silu")
    sizes, (a, b) = _split(walk, x, 2)
    closing = walk.next_node("Concat")
    if closing is None or closing.inputs[:2] != [a, b]:
        raise Mismatch(f"after {x}: expected a concat of the split halves and the block outputs")
    params = _nest("cv1", p1)
    blocks, cur = [], b
    for end in closing.inputs[2:]:
        block_in, convs, residual = cur, [], False
        while cur != end:
            if walk.peek() == "Conv" and not residual:
                c, p, cur = conv(walk, cur, "silu")
                params.update(_nest(f"blocks.{len(blocks)}.convs.{len(convs)}", p))
                convs.append(c)
            elif walk.peek() == "Add" and convs and not residual:
                cur = walk.take("Add", cur, block_in).outputs[0]
                residual = True
            else:
                raise Mismatch(f"node {walk.i}: expected the block to reach {end}, found {walk.peek()}")
        blocks.append({"convs": convs, "residual": residual})
    if not blocks:
        raise Mismatch(f"{closing.name}: a C2f without blocks")
    cv2, p2, out = conv(walk, concat(walk, 1, a, b, *closing.inputs[2:]), "silu")
    params.update(_nest("cv2", p2))
    return {"type": "c2f", "cv1": cv1, "split": sizes, "blocks": blocks, "cv2": cv2}, params, out


def _psa(walk, src):
    cv1, p1, x = conv(walk, src, "silu")
    sizes, (a, b) = _split(walk, x, 2)
    params = _nest("cv1", p1)

    qkv, pq, y = conv(walk, b, None)
    reshape = walk.take("Reshape", y)
    shape = walk.ints(reshape, 1)
    if len(shape) != 4:
        raise Mismatch(f"{reshape.name}: expected a [B, heads, channels, N] reshape, found {shape}")
    heads = shape[1]
    parts, (q, k, v) = _split(walk, reshape.outputs[0], 3, axis=2)
    key_dim, key_dim2, head_dim = parts
    if key_dim != key_dim2 or heads * head_dim != sizes[1] or heads * (2 * key_dim + head_dim) != qkv["cout"]:
        raise Mismatch(f"attention layout {parts} x {heads} heads does not fit {sizes[1]} channels")
    qt = walk.take("Transpose", q)
    if qt.attrs.get("perm") != [0, 1, 3, 2]:
        raise Mismatch(f"{qt.name}: expected perm [0, 1, 3, 2]")
    qk = walk.take("MatMul", qt.outputs[0], k)
    scaled = walk.take("Mul", qk.outputs[0])
    scale = walk.scalar(scaled)
    soft = walk.take("Softmax", scaled.outputs[0])
    if soft.attrs.get("axis", -1) != -1:
        raise Mismatch(f"{soft.name}: expected a softmax over the last axis")
    at = walk.take("Transpose", soft.outputs[0])
    if at.attrs.get("perm") != [0, 1, 3, 2]:
        raise Mismatch(f"{at.name}: expected perm [0, 1, 3, 2]")
    weighted = walk.take("MatMul", v, at.outputs[0])
    back = walk.take("Reshape", weighted.outputs[0]).outputs[0]
    v_map = walk.take("Reshape", v).outputs[0]
    pe, pp, pe_out = conv(walk, v_map, None)
    summed = walk.take("Add", back, pe_out).outputs[0]
    proj, pj, attn_out = conv(walk, summed, None)
    b1 = walk.take("Add", b, attn_out).outputs[0]
    ffn, pf, ffn_out = _chain(walk, b1, ["silu", None])
    b2 = walk.take("Add", b1, ffn_out).outputs[0]
    cv2, p2, out = conv(walk, concat(walk, 1, a, b2), "silu")
    attn = {"qkv": qkv, "pe": pe, "proj": proj, "heads": heads, "key_dim": key_dim, "head_dim": head_dim,
            "scale": scale}
    params.update({**_nest("attn.qkv", pq), **_nest("attn.pe", pp), **_nest("attn.proj", pj),
                   **_nest("ffn", pf), **_nest("cv2", p2)})
    return {"type": "psa", "cv1": cv1, "split": sizes, "attn": attn, "ffn": ffn, "cv2": cv2}, params, out


def _scdown(walk, src):
    """A 1x1 convolution and a strided depthwise convolution without activation."""
    return _chain(walk, src, ["silu", None])


def _plain_conv(walk, src):
    c, p, out = conv(walk, src, "silu")
    return {"type": "conv", **c}, p, out


def _detect(walk, levels):
    """YOLOv10's one-to-one head, the distribution focal loss decode, the anchors and the
    top-k the export bakes in. Returns config, parameters and the output tensor name."""
    box, cls, params, level_outs = [], [], {}, []
    for i, src in enumerate(levels):
        b, pb, bo = _chain(walk, src, ["silu", "silu", None])
        c, pc, co = _chain(walk, src, ["silu", "silu", "silu", "silu", None])
        b.pop("type")
        c.pop("type")
        box.append(b)
        cls.append(c)
        params.update({**_nest(f"box.{i}", pb), **_nest(f"cls.{i}", pc)})
        level_outs.append(concat(walk, 1, bo, co))
    box_ch, cls_ch = box[0]["convs"][-1]["cout"], cls[0]["convs"][-1]["cout"]
    if box_ch % 4 or any(b["convs"][-1]["cout"] != box_ch for b in box) or any(c["convs"][-1]["cout"] != cls_ch for c in cls):
        raise Mismatch("the head levels disagree on the box / class channels")
    reg_max, num_classes = box_ch // 4, cls_ch

    flat = []
    for out in level_outs:
        node = walk.take("Reshape", out)
        if walk.ints(node, 1)[1:] != [box_ch + cls_ch, -1]:
            raise Mismatch(f"{node.name}: expected [B, {box_ch + cls_ch}, -1]")
        flat.append(node.outputs[0])
    joined = concat(walk, 2, *flat)
    sizes, (dist, logits) = _split(walk, joined, 2)
    if sizes != [box_ch, cls_ch]:
        raise Mismatch(f"expected the head split [{box_ch}, {cls_ch}], found {sizes}")

    # distribution focal loss
    node = walk.take("Reshape", dist)
    shape = walk.ints(node, 1)
    if shape[1:3] != [4, reg_max]:
        raise Mismatch(f"{node.name}: expected [B, 4, {reg_max}, A], found {shape}")
    anchors_total = shape[3]
    node = walk.take("Transpose", node.outputs[0])
    if node.attrs.get("perm") != [0, 2, 1, 3]:
        raise Mismatch(f"{node.name}: expected perm [0, 2, 1, 3]")
    node = walk.take("Softmax", node.outputs[0])
    if node.attrs.get("axis") != 1:
        raise Mismatch(f"{node.name}: expected a softmax over the bins (axis 1)")
    dfl = walk.take("Conv", node.outputs[0])
    dfl_w = walk.weight(dfl, 1)
    if list(dfl_w.shape) != [1, reg_max, 1, 1] or (len(dfl.inputs) > 2 and dfl.inputs[2]):
        raise Mismatch(f"{dfl.name}: expected a bias-free [1, {reg_max}, 1, 1] expectation convolution")
    params["dfl"] = dfl_w
    node = walk.take("Reshape", dfl.outputs[0])
    _, (lt, rb) = _split(walk, node.outputs[0], 2)

    # anchors and strides are export constants; the module computes them from the map
    # sizes, which is only the same thing if they are the grid centres level by level
    sub = walk.take("Sub", lt)
    if sub.inputs[1] != lt:
        raise Mismatch(f"{sub.name}: expected anchors - distance")
    anchors = walk.tensors[sub.inputs[0]]
    add = walk.take("Add", rb)
    other = walk.tensors.get(next(n for n in add.inputs if n != rb))
    if other is None or other.shape != anchors.shape or not torch.equal(other, anchors):
        raise Mismatch(f"{add.name}: expected anchors + distance")
    xyxy = concat(walk, 1, sub.outputs[0], add.outputs[0])
    mul = walk.take("Mul", xyxy)
    stride_const = walk.const(mul)
    sig = walk.take("Sigmoid", logits)
    pred = concat(walk, 1, mul.outputs[0], sig.outputs[0])

    # top-k
    node = walk.take("Transpose", pred)
    if node.attrs.get("perm") != [0, 2, 1]:
        raise Mismatch(f"{node.name}: expected perm [0, 2, 1]")
    sizes, (boxes, scores) = _split(walk, node.outputs[0], 2, axis=-1)
    if sizes != [4, num_classes]:
        raise Mismatch(f"expected the prediction split [4, {num_classes}], found {sizes}")
    rmax = walk.take("ReduceMax", scores)
    if rmax.attrs.get("axes") != [-1] or rmax.attrs.get("keepdims", 1):
        raise Mismatch(f"{rmax.name}: expected a max over the classes")
    top = walk.take("TopK", rmax.outputs[0])
    max_det = walk.ints(top, 1)[0]
    if top.attrs.get("axis", -1) != -1 or not top.attrs.get("largest", 1) or not top.attrs.get("sorted", 1):
        raise Mismatch(f"{top.name}: expected a sorted largest top-k")
    idx = walk.take("Unsqueeze", top.outputs[1]).outputs[0]
    tiled = walk.take("Tile", idx)
    top_boxes = walk.take("GatherElements", boxes, tiled.outputs[0]).outputs[0]
    tiled = walk.take("Tile", idx)
    top_scores = walk.take("GatherElements", scores, tiled.outputs[0]).outputs[0]
    flat_scores = walk.take("Flatten", top_scores).outputs[0]
    top2 = walk.take("TopK", flat_scores)
    if walk.ints(top2, 1)[0] != max_det:
        raise Mismatch(f"{top2.name}: the two top-k disagree on max_det")
    mod = walk.take("Mod", top2.outputs[1])
    if int(walk.const(mod)) != num_classes or mod.attrs.get("fmod", 0):
        raise Mismatch(f"{mod.name}: expected index % {num_classes}")
    div = walk.take("Div", top2.outputs[1])
    if int(walk.const(div)) != num_classes:
        raise Mismatch(f"{div.name}: expected index // {num_classes}")
    idx2 = walk.take("Unsqueeze", div.outputs[0]).outputs[0]
    tiled = walk.take("Tile", idx2)
    final_boxes = walk.take("GatherElements", top_boxes, tiled.outputs[0]).outputs[0]
    score_col = walk.take("Unsqueeze", top2.outputs[0]).outputs[0]
    label_col = walk.take("Unsqueeze", mod.outputs[0]).outputs[0]
    label_col = walk.take("Cast", label_col).outputs[0]
    out = concat(walk, -1, final_boxes, score_col, label_col)

    cfg = {"box": box, "cls": cls, "reg_max": reg_max, "num_classes": num_classes, "max_det": max_det}
    return cfg, params, out, anchors, stride_const, anchors_total


def _level_sizes(input_size, strides):
    return [(input_size[0] // s, input_size[1] // s) for s in strides]


def extract_yolov10(graph):
    """(config, state) of a YOLOv10 export (Ultralytics, one-to-one head with the top-k
    baked in): backbone Conv, Conv, C2f, Conv, C2f, SCDown, C2f, SCDown, C2f, SPPF, PSA;
    the PAFPN neck without reduce convolutions; the detection head."""
    if len(graph.inputs) != 1 or len(graph.outputs) != 1:
        raise Mismatch(f"expected one input and one output, found {len(graph.inputs)} and {len(graph.outputs)}")
    input_size = graph.input_size()
    walk = Walk(graph)
    x = graph.inputs[0]
    layout = [_plain_conv, _plain_conv, _c2f, _plain_conv, _c2f, _scdown, _c2f, _scdown, _c2f, spp, _psa]
    features = [4, 6, 10]
    backbone, state, outs = [], {}, []
    for i, step in enumerate(layout):
        cfg, p, x = step(walk, x)
        if step is spp:
            if not cfg["cascade"]:
                raise Mismatch("YOLOv10's SPPF pools each pool again; this one pools the same map")
            cfg = {"type": "spp", **cfg}
        backbone.append(cfg)
        state.update(_nest(f"backbone.{i}", p))
        outs.append(x)
    c0, c1, c2 = (outs[i] for i in features)

    scale, up = upsample(walk, c2)
    top_down0, p_t0, p1 = _c2f(walk, concat(walk, 1, up, c1))
    scale1, up1 = upsample(walk, p1)
    if scale1 != scale:
        raise Mismatch(f"the two upsamples scale by {scale} and {scale1}")
    top_down1, p_t1, p0 = _c2f(walk, concat(walk, 1, up1, c0))
    down0, p_d0, d0 = _plain_conv(walk, p0)
    bottom_up0, p_b0, n1 = _c2f(walk, concat(walk, 1, d0, p1))
    down1, p_d1, d1 = _scdown(walk, n1)
    bottom_up1, p_b1, n2 = _c2f(walk, concat(walk, 1, d1, c2))
    neck = {"reduce": [None, None], "top_down": [top_down0, top_down1], "down": [down0, down1],
            "bottom_up": [bottom_up0, bottom_up1], "scale": scale}
    for name, p in (("top_down.0", p_t0), ("top_down.1", p_t1), ("down.0", p_d0), ("down.1", p_d1),
                    ("bottom_up.0", p_b0), ("bottom_up.1", p_b1)):
        state.update(_nest(f"neck.{name}", p))

    head, p, out, anchors, stride_const, anchors_total = _detect(walk, [p0, n1, n2])
    state.update(_nest("head", p))
    if walk.i != len(walk.nodes) or out != graph.outputs[0]:
        raise Mismatch(f"the head ends at node {walk.i} of {len(walk.nodes)}, producing {out} "
                       f"where the graph returns {graph.outputs[0]}")

    # the strides: one value per level, in the order of the levels
    stride_values = stride_const.reshape(-1)
    strides, start = [], 0
    for level in range(3):
        if start >= stride_values.numel():
            raise Mismatch("the stride constant is shorter than the three levels")
        s = float(stride_values[start])
        h, w = input_size[0] // int(s), input_size[1] // int(s)
        if s != int(s) or not torch.all(stride_values[start:start + h * w] == s):
            raise Mismatch(f"level {level}: the stride constant is not {s} over its {h}x{w} anchors")
        strides.append(int(s))
        start += h * w
    if start != stride_values.numel() or start != anchors_total:
        raise Mismatch(f"{start} anchors over the three levels, the export has {stride_values.numel()}")
    head["strides"] = strides
    config = {"input_size": input_size, "backbone": backbone, "features": features, "neck": neck, "head": head}
    return config, state, anchors
