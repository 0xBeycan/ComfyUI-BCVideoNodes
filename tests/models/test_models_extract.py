"""The offline extractors (scripts/onnx_extract.py) on miniature RTMW and YOLOv10 graphs:
the structure they read out and the numbers the module built from it produces.

The graphs are written here node for node the way the upstream exports write them, with
tiny channels and 64x64 / 32x32 inputs, and the reference numbers come from onnx's own reference
evaluator, so the tests run on the CPU in seconds and need neither the model files nor a
GPU. What they can prove without the real weights is exactly what goes wrong silently: that
the extractor reads block counts, kernels, residuals and constants out of the graph rather
than assuming them, that the module is wired the way the graph is, and that a graph which
is not the model is refused instead of half-matched.

Needs onnx (a conversion-time dependency, not a runtime one); skipped without it.

    python -m pytest tests/models/test_models_extract.py
"""
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import onnx_extract  # noqa: E402

from bcvideonodes.models.common import checkpoint  # noqa: E402

KEYPOINTS = 5
SIMCC_X, SIMCC_Y = 12, 16


class Builder:
    """Nodes and initializers for one graph. Weights are small random values: the test is
    about wiring, and both sides see exactly the same numbers either way."""

    def __init__(self, seed=0):
        self.nodes, self.inits, self.count = [], [], 0
        self.rng = np.random.default_rng(seed)

    def _name(self):
        self.count += 1
        return f"v{self.count}"

    def init(self, array):
        name = self._name()
        self.inits.append(numpy_helper.from_array(np.asarray(array, dtype=np.float32), name))
        return name

    def ints(self, values):
        name = self._name()
        self.inits.append(numpy_helper.from_array(np.asarray(values, dtype=np.int64), name))
        return name

    def node(self, op, inputs, outputs=1, **attrs):
        names = [self._name() for _ in range(outputs)]
        self.nodes.append(helper.make_node(op, inputs, names, **attrs))
        return names[0] if outputs == 1 else names

    def random(self, *shape):
        return self.rng.standard_normal(shape) * 0.3

    def model(self, inp, outs, shape, opset=11):
        graph = helper.make_graph(
            self.nodes, "mini",
            [helper.make_tensor_value_info(inp, TensorProto.FLOAT, shape)],
            [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in outs],
            self.inits)
        return helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])


def conv(b, x, cin, cout, kernel=1, stride=1, groups=1, act="silu"):
    y = b.node("Conv", [x, b.init(b.random(cout, cin // groups, kernel, kernel)), b.init(b.random(cout))],
               kernel_shape=[kernel, kernel], strides=[stride, stride],
               pads=[kernel // 2] * 4, dilations=[1, 1], group=groups)
    if act == "silu":
        return b.node("Mul", [y, b.node("Sigmoid", [y])])
    if act == "relu":
        return b.node("Relu", [y])
    return y


def upsample(b, x):
    return b.node("Resize", [x, b.init(np.zeros(0)), b.init([1.0, 1.0, 2.0, 2.0])],
                  mode="nearest", coordinate_transformation_mode="asymmetric", nearest_mode="floor")


def save(model, tmp_path, name):
    path = str(tmp_path / name)
    onnx.save(model, path)
    return onnx_extract.OnnxGraph(path)


def native(architecture, extracted):
    config, state = extracted[:2]
    net = checkpoint.build(architecture, config)
    checkpoint.check_state(net, state, architecture)
    net.load_state_dict(state, strict=True, assign=True)
    return net.eval()


def reference(model, x):
    return ReferenceEvaluator(model).run(None, {"input": x})


# -- RTMW ----------------------------------------------------------------------------------

def csp(b, x, cin, cout, blocks, attention, residual):
    mid = cout // 2
    short = conv(b, x, cin, mid)
    main = conv(b, x, cin, mid)
    for _ in range(blocks):
        y = conv(b, main, mid, mid, kernel=3)
        y = conv(b, y, mid, mid, kernel=5, groups=mid)
        y = conv(b, y, mid, mid)
        main = b.node("Add", [y, main]) if residual else y
    out = b.node("Concat", [main, short], axis=1)
    if attention:
        gate = conv(b, b.node("GlobalAveragePool", [out]), cout, cout, act=None)
        out = b.node("Mul", [out, b.node("HardSigmoid", [gate], alpha=1 / 6, beta=0.5)])
    return conv(b, out, cout, cout)


def spp(b, x, channels, kernels=(3, 5, 7)):
    hidden = conv(b, x, channels, channels // 2)
    pooled = [b.node("MaxPool", [hidden], kernel_shape=[k, k], strides=[1, 1], pads=[k // 2] * 4, ceil_mode=0)
              for k in kernels]
    return conv(b, b.node("Concat", [hidden] + pooled, axis=1), (channels // 2) * (len(kernels) + 1), channels)


def flatten(b, x):
    shape = b.node("Shape", [x])
    sliced = b.node("Slice", [shape, b.ints([0]), b.ints([2]), b.ints([0])])
    return b.node("Reshape", [x, b.node("Concat", [sliced, b.ints([-1])], axis=0)])


def scale_norm(b, x, dim, gain):
    norm = b.node("Sqrt", [b.node("ReduceSum", [b.node("Mul", [x, x])], axes=[2], keepdims=1)])
    scaled = b.node("Mul", [norm, b.init(dim ** -0.5)])
    clipped = b.node("Clip", [scaled, b.init(1e-5), ""])
    return b.node("Mul", [b.node("Div", [x, clipped]), b.init(gain)])


def gau(b, x, dim, expand, key):
    normed = scale_norm(b, x, dim, 0.9)
    uv = b.node("MatMul", [normed, b.init(b.random(dim, 2 * expand + key))])
    activated = b.node("Mul", [uv, b.node("Sigmoid", [uv])])
    u, v, base = b.node("Split", [activated], outputs=3, axis=2, split=[expand, expand, key])
    offset = b.node("Add", [b.node("Mul", [b.node("Unsqueeze", [base], axes=[2]),
                                           b.init(b.random(1, 1, 2, key))]), b.init(b.random(2, key))])
    q, k = b.node("Split", [offset], outputs=2, axis=2, split=[1, 1])
    q = b.node("Squeeze", [q], axes=[2])
    k = b.node("Squeeze", [k], axes=[2])
    qk = b.node("MatMul", [q, b.node("Transpose", [k], perm=[0, 2, 1])])
    attn = b.node("Relu", [b.node("Div", [qk, b.init(float(key) ** 0.5)])])
    gated = b.node("Mul", [u, b.node("MatMul", [b.node("Mul", [attn, attn]), v])])
    out = b.node("MatMul", [gated, b.init(b.random(expand, dim))])
    return b.node("Add", [b.node("Mul", [x, b.init(b.random(dim))]), out])


def mini_rtmw(blocks=(1, 2, 1, 1), simcc=(SIMCC_X, SIMCC_Y)):
    """A 64x64 RTMW: three stem convolutions, four stages, a three level PAFPN and the SimCC
    head, with the block counts the caller asks for."""
    b = Builder()
    x = "input"
    x = conv(b, x, 3, 4, kernel=3, stride=2)
    x = conv(b, x, 4, 4, kernel=3)
    x = conv(b, x, 4, 8, kernel=3)

    feats = []
    for i, (cin, cout) in enumerate(((8, 8), (8, 16), (16, 32), (32, 64))):
        x = conv(b, x, cin, cout, kernel=3, stride=2)
        if i == 3:
            x = spp(b, x, cout)
        x = csp(b, x, cout, cout, blocks[i], attention=True, residual=i < 3)
        feats.append(x)

    r2 = conv(b, feats[3], 64, 32)
    p1 = csp(b, b.node("Concat", [upsample(b, r2), feats[2]], axis=1), 64, 32, 1, False, False)
    r1 = conv(b, p1, 32, 16)
    p0 = csp(b, b.node("Concat", [upsample(b, r1), feats[1]], axis=1), 32, 16, 1, False, False)
    n1 = csp(b, b.node("Concat", [conv(b, p0, 16, 16, kernel=3, stride=2), r1], axis=1), 32, 32, 1, False, False)
    n2 = csp(b, b.node("Concat", [conv(b, n1, 32, 32, kernel=3, stride=2), r2], axis=1), 64, 64, 1, False, False)

    a = scale_norm(b, flatten(b, conv(b, n2, 64, KEYPOINTS, kernel=3, act="relu")), 4, 0.3)
    a = b.node("MatMul", [a, b.init(b.random(4, 8))])
    shuffled = b.node("DepthToSpace", [n2], blocksize=2, mode="CRD")
    mid = conv(b, shuffled, 16, 16, kernel=3, act="relu")
    y = conv(b, b.node("Concat", [mid, n1], axis=1), 48, KEYPOINTS, kernel=3, act="relu")
    c = scale_norm(b, flatten(b, y), 16, 0.4)
    c = b.node("MatMul", [c, b.init(b.random(16, 8))])
    z = gau(b, b.node("Concat", [a, c], axis=2), 16, 8, 4)
    simcc_x = b.node("MatMul", [z, b.init(b.random(16, simcc[0]))])
    simcc_y = b.node("MatMul", [z, b.init(b.random(16, simcc[1]))])
    return b.model("input", [simcc_x, simcc_y], ["batch", 3, 64, 64])


@pytest.fixture(scope="module")
def rtmw_model():
    return mini_rtmw()


@pytest.fixture(scope="module")
def rtmw_graph(rtmw_model, tmp_path_factory):
    return save(rtmw_model, tmp_path_factory.mktemp("rtmw"), "mini.onnx")


def test_rtmw_structure_is_read_from_the_graph(rtmw_graph):
    net = native("rtmw", onnx_extract.extract_rtmw(rtmw_graph))
    assert net.config["input_size"] == [64, 64]
    assert [len(stage.csp.blocks) for stage in net.stages] == [1, 2, 1, 1]
    assert [stage.spp is not None for stage in net.stages] == [False, False, False, True]
    assert net.stages[3].spp.kernels == [3, 5, 7] and not net.stages[3].spp.cascade
    # the backbone gates every stage, the neck gates nothing
    assert all(stage.csp.attention is not None for stage in net.stages)
    assert net.neck.top_down[0].attention is None and net.neck.bottom_up[1].attention is None
    # only the first three stages carry the residual inside their blocks
    assert [stage.csp.blocks[0].residual for stage in net.stages] == [True, True, True, False]
    assert net.neck.scale == 2.0
    assert net.head.blocksize == 2
    assert net.head.gau.sizes == [8, 8, 4]
    assert net.head.gau.sqrt_s == pytest.approx(2.0)
    assert net.head.cls_x.weight.shape == (16, SIMCC_X)
    assert net.head.cls_y.weight.shape == (16, SIMCC_Y)


def test_rtmw_hyper_parameters_come_from_the_node_attributes(rtmw_graph):
    net = native("rtmw", onnx_extract.extract_rtmw(rtmw_graph))
    depthwise = net.stages[0].csp.blocks[0].convs[1]
    assert depthwise.weight.shape == (4, 1, 5, 5) and depthwise.groups == 4 and depthwise.padding == (2, 2)
    assert net.stages[1].downsample.stride == (2, 2) and net.stages[1].downsample.padding == (1, 1)
    attention = net.stages[0].csp.attention
    assert attention.alpha == pytest.approx(1 / 6) and attention.beta == pytest.approx(0.5)
    assert net.head.mlp_norm.scale == pytest.approx(0.5) and net.head.mlp_norm.gain == pytest.approx(0.3)
    assert net.head.mlp2_norm.scale == pytest.approx(0.25) and net.head.mlp2_norm.gain == pytest.approx(0.4)
    assert net.head.mlp_norm.eps == pytest.approx(1e-5)


def test_rtmw_same_numbers_as_the_onnx_reference(rtmw_model, rtmw_graph):
    net = native("rtmw", onnx_extract.extract_rtmw(rtmw_graph))
    x = np.random.default_rng(1).standard_normal((2, 3, 64, 64)).astype(np.float32)
    with torch.inference_mode():
        got = net(torch.from_numpy(x))
    want = reference(rtmw_model, x)
    assert len(got) == len(want) == 2
    for a, b in zip(got, want):
        assert tuple(a.shape) == b.shape
        assert np.abs(a.numpy() - b).max() < 1e-4


def test_a_graph_that_is_not_rtmw_is_refused(rtmw_graph, tmp_path):
    with pytest.raises(onnx_extract.Mismatch):
        onnx_extract.extract_vitpose(rtmw_graph)
    with pytest.raises(onnx_extract.Mismatch):
        onnx_extract.extract_yolov10(rtmw_graph)

    b = Builder()
    out = b.node("Relu", [conv(b, "input", 3, 4, kernel=3)])
    plain = save(b.model("input", [out, out], ["batch", 3, 64, 64]), tmp_path, "plain.onnx")
    with pytest.raises(onnx_extract.Mismatch):
        onnx_extract.extract_rtmw(plain)


def test_a_graph_missing_a_node_is_refused(tmp_path):
    """Dropping the last node leaves a graph whose shapes are all still right; the walk has
    to notice that it ran out of nodes rather than return a module with one output."""
    model = mini_rtmw()
    del model.graph.node[-1]
    del model.graph.output[-1]
    with pytest.raises(onnx_extract.Mismatch):
        onnx_extract.extract_rtmw(save(model, tmp_path, "short.onnx"))


def test_a_rewired_graph_is_refused(tmp_path):
    """The concat that joins the two halves of a CSP layer takes (main, short). Swapping it
    keeps every shape and changes the result, so the walk has to check the order."""
    model = mini_rtmw()
    node = next(n for n in model.graph.node if n.op_type == "Concat")
    node.input[0], node.input[1] = node.input[1], node.input[0]
    with pytest.raises(onnx_extract.Mismatch):
        onnx_extract.extract_rtmw(save(model, tmp_path, "swapped.onnx"))


# -- YOLOv10 -------------------------------------------------------------------------------

REG_MAX, CLASSES, MAX_DET = 4, 3, 10
STRIDES = (8, 16, 32)


def c2f(b, x, cin, cout, n, shortcut, cib=False):
    c = cout // 2
    a, h = b.node("Split", [conv(b, x, cin, 2 * c), b.ints([c, c])], outputs=2, axis=1)
    outs, cur = [a, h], h
    for _ in range(n):
        if cib:
            t = conv(b, cur, c, c, kernel=3, groups=c)
            t = conv(b, t, c, 2 * c)
            t = conv(b, t, 2 * c, 2 * c, kernel=3, groups=2 * c)
            t = conv(b, t, 2 * c, c)
            t = conv(b, t, c, c, kernel=3, groups=c)
        else:
            t = conv(b, conv(b, cur, c, c, kernel=3), c, c, kernel=3)
        cur = b.node("Add", [cur, t]) if shortcut else t
        outs.append(cur)
    return conv(b, b.node("Concat", outs, axis=1), (2 + n) * c, cout)


def scdown(b, x, cin, cout):
    return conv(b, conv(b, x, cin, cout), cout, cout, kernel=3, stride=2, groups=cout, act=None)


def sppf(b, x, c):
    h = conv(b, x, c, c // 2)
    p1 = b.node("MaxPool", [h], kernel_shape=[5, 5], strides=[1, 1], pads=[2] * 4, ceil_mode=0, dilations=[1, 1])
    p2 = b.node("MaxPool", [p1], kernel_shape=[5, 5], strides=[1, 1], pads=[2] * 4, ceil_mode=0, dilations=[1, 1])
    p3 = b.node("MaxPool", [p2], kernel_shape=[5, 5], strides=[1, 1], pads=[2] * 4, ceil_mode=0, dilations=[1, 1])
    return conv(b, b.node("Concat", [h, p1, p2, p3], axis=1), 2 * c, c)


def psa(b, x, c, heads, size):
    dim = c // 2
    head_dim = dim // heads
    key_dim = head_dim // 2
    a, h = b.node("Split", [conv(b, x, c, c), b.ints([dim, dim])], outputs=2, axis=1)
    qkv = conv(b, h, dim, dim + 2 * key_dim * heads, act=None)
    q, k, v = b.node("Split", [b.node("Reshape", [qkv, b.ints([1, heads, -1, size * size])]),
                               b.ints([key_dim, key_dim, head_dim])], outputs=3, axis=2)
    attn = b.node("Mul", [b.node("MatMul", [b.node("Transpose", [q], perm=[0, 1, 3, 2]), k]), b.init(key_dim ** -0.5)])
    attn = b.node("Softmax", [attn], axis=-1)
    o = b.node("Reshape", [b.node("MatMul", [v, b.node("Transpose", [attn], perm=[0, 1, 3, 2])]),
                           b.ints([1, -1, size, size])])
    pe = conv(b, b.node("Reshape", [v, b.ints([1, -1, size, size])]), dim, dim, kernel=3, groups=dim, act=None)
    h = b.node("Add", [h, conv(b, b.node("Add", [o, pe]), dim, dim, act=None)])
    h = b.node("Add", [h, conv(b, conv(b, h, dim, 2 * dim), 2 * dim, dim, act=None)])
    return conv(b, b.node("Concat", [a, h], axis=1), c, c)


def anchor_constants(size):
    points, strides = [], []
    for s in STRIDES:
        n = size // s
        gy, gx = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
        points.append(np.stack([gx.ravel(), gy.ravel()]))
        strides.append(np.full(n * n, float(s)))
    return np.concatenate(points, 1)[None], np.concatenate(strides)[None]


def mini_yolov10(size=32):
    """A 32x32 YOLOv10: the export's backbone layout, the PAFPN neck without reduce
    convolutions and the one-to-one head with the top-k, at tiny widths. 32x32 keeps the
    anchor count (21) under the 64 rows onnx's reference GatherElements can gather over."""
    b = Builder(seed=3)
    x = conv(b, "input", 3, 8, kernel=3, stride=2)
    x = conv(b, x, 8, 8, kernel=3, stride=2)
    x = c2f(b, x, 8, 8, 1, True)
    x = conv(b, x, 8, 16, kernel=3, stride=2)
    c0 = x = c2f(b, x, 16, 16, 2, True)
    x = scdown(b, x, 16, 16)
    c1 = x = c2f(b, x, 16, 16, 1, True, cib=True)
    x = scdown(b, x, 16, 32)
    x = c2f(b, x, 32, 32, 1, True, cib=True)
    x = sppf(b, x, 32)
    c2 = psa(b, x, 32, heads=2, size=size // 32)

    p1 = c2f(b, b.node("Concat", [upsample(b, c2), c1], axis=1), 48, 16, 1, False)
    p0 = c2f(b, b.node("Concat", [upsample(b, p1), c0], axis=1), 32, 16, 1, False)
    n1 = c2f(b, b.node("Concat", [conv(b, p0, 16, 16, kernel=3, stride=2), p1], axis=1), 32, 16, 1, True, cib=True)
    n2 = c2f(b, b.node("Concat", [scdown(b, n1, 16, 16), c2], axis=1), 48, 32, 1, True, cib=True)

    # the export runs every level's branches first and flattens the three levels after
    levels = []
    for src, ch in ((p0, 16), (n1, 16), (n2, 32)):
        box = conv(b, conv(b, conv(b, src, ch, 8, kernel=3), 8, 8, kernel=3), 8, 4 * REG_MAX, act=None)
        cls = conv(b, src, ch, ch, kernel=3, groups=ch)
        cls = conv(b, cls, ch, 8)
        cls = conv(b, cls, 8, 8, kernel=3, groups=8)
        cls = conv(b, cls, 8, 8)
        cls = conv(b, cls, 8, CLASSES, act=None)
        levels.append(b.node("Concat", [box, cls], axis=1))
    flat = [b.node("Reshape", [level, b.ints([1, 4 * REG_MAX + CLASSES, -1])]) for level in levels]
    anchors_n = sum((size // s) ** 2 for s in STRIDES)
    dist, logits = b.node("Split", [b.node("Concat", flat, axis=2), b.ints([4 * REG_MAX, CLASSES])], outputs=2, axis=1)
    d = b.node("Transpose", [b.node("Reshape", [dist, b.ints([1, 4, REG_MAX, anchors_n])])], perm=[0, 2, 1, 3])
    d = b.node("Conv", [b.node("Softmax", [d], axis=1), b.init(np.arange(REG_MAX).reshape(1, REG_MAX, 1, 1))],
               kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)
    lt, rb = b.node("Split", [b.node("Reshape", [d, b.ints([1, 4, anchors_n])]), b.ints([2, 2])], outputs=2, axis=1)
    points, strides = anchor_constants(size)
    anchors = b.init(points)
    xyxy = b.node("Concat", [b.node("Sub", [anchors, lt]), b.node("Add", [anchors, rb])], axis=1)
    pred = b.node("Concat", [b.node("Mul", [xyxy, b.init(strides)]), b.node("Sigmoid", [logits])], axis=1)
    boxes, scores = b.node("Split", [b.node("Transpose", [pred], perm=[0, 2, 1]), b.ints([4, CLASSES])],
                           outputs=2, axis=-1)
    k = b.ints([MAX_DET])
    _, idx = b.node("TopK", [b.node("ReduceMax", [scores], axes=[-1], keepdims=0), k], outputs=2,
                    axis=-1, largest=1, sorted=1)
    idx = b.node("Unsqueeze", [idx, b.ints([-1])])
    top_boxes = b.node("GatherElements", [boxes, b.node("Tile", [idx, b.ints([1, 1, 4])])], axis=1)
    top_scores = b.node("GatherElements", [scores, b.node("Tile", [idx, b.ints([1, 1, CLASSES])])], axis=1)
    vals, idx2 = b.node("TopK", [b.node("Flatten", [top_scores], axis=1), k], outputs=2, axis=-1, largest=1, sorted=1)
    nc = b.ints(CLASSES)
    labels = b.node("Mod", [idx2, nc], fmod=0)
    which = b.node("Unsqueeze", [b.node("Div", [idx2, nc]), b.ints([-1])])
    final = b.node("GatherElements", [top_boxes, b.node("Tile", [which, b.ints([1, 1, 4])])], axis=1)
    out = b.node("Concat", [final, b.node("Unsqueeze", [vals, b.ints([-1])]),
                            b.node("Cast", [b.node("Unsqueeze", [labels, b.ints([-1])])], to=TensorProto.FLOAT)],
                 axis=-1)
    return b.model("input", [out], [1, 3, size, size], opset=13)


@pytest.fixture(scope="module")
def yolo_model():
    return mini_yolov10()


@pytest.fixture(scope="module")
def yolo_graph(yolo_model, tmp_path_factory):
    return save(yolo_model, tmp_path_factory.mktemp("yolo"), "mini.onnx")


def test_yolov10_structure_is_read_from_the_graph(yolo_graph):
    extracted = onnx_extract.extract_yolov10(yolo_graph)
    net = native("yolov10", extracted)
    cfg = net.config
    assert [layer["type"] for layer in cfg["backbone"]] == ["conv", "conv", "c2f", "conv", "c2f", "chain", "c2f",
                                                            "chain", "c2f", "spp", "psa"]
    assert [len(net.backbone[i].blocks) for i in (2, 4, 6, 8)] == [1, 2, 1, 1]
    # a bottleneck is two convolutions, a CIB five; the neck's first two merges have no residual
    assert [len(net.backbone[i].blocks[0].convs) for i in (2, 6)] == [2, 5]
    assert [m.blocks[0].residual for m in (*net.neck.top_down, *net.neck.bottom_up)] == [False, False, True, True]
    assert net.backbone[9].cascade and net.backbone[9].kernels == [5, 5, 5]
    assert net.backbone[5].convs[1].groups == 16 and net.backbone[5].convs[1].act is None
    attn = net.backbone[10].attn
    assert (attn.heads, attn.sizes) == (2, [4, 4, 8]) and attn.scale == pytest.approx(4 ** -0.5)
    assert all(isinstance(r, torch.nn.Identity) for r in net.neck.reduce)
    head = net.head
    assert (head.reg_max, head.num_classes, head.max_det, head.strides) == (REG_MAX, CLASSES, MAX_DET, [8, 16, 32])


def test_yolov10_anchors_are_the_exported_constants(yolo_graph):
    import convert_models

    extracted = onnx_extract.extract_yolov10(yolo_graph)
    convert_models.check_anchors(native("yolov10", extracted), extracted[2])


def test_yolov10_same_numbers_as_the_onnx_reference(yolo_model, yolo_graph):
    net = native("yolov10", onnx_extract.extract_yolov10(yolo_graph))
    x = np.random.default_rng(2).random((1, 3, 32, 32)).astype(np.float32)
    with torch.inference_mode():
        got = net(torch.from_numpy(x)).numpy()
    want = reference(yolo_model, x)[0]
    assert got.shape == want.shape == (1, MAX_DET, 6)
    assert np.array_equal(got[..., 5], want[..., 5])
    assert np.abs(got[..., :5] - want[..., :5]).max() < 1e-4


def test_a_rewired_yolov10_is_refused(tmp_path):
    """C2f's closing concat takes the split halves first; swapping them keeps the shapes."""
    model = mini_yolov10()
    node = next(n for n in model.graph.node if n.op_type == "Concat")
    node.input[0], node.input[1] = node.input[1], node.input[0]
    with pytest.raises(onnx_extract.Mismatch):
        onnx_extract.extract_yolov10(save(model, tmp_path, "swapped.onnx"))


def test_yolov10_with_moved_anchors_is_refused(tmp_path):
    """The module computes the anchor grid itself, so an export whose constant is anything
    else must not convert."""
    import convert_models

    model = mini_yolov10()
    sub = next(n for n in model.graph.node if n.op_type == "Sub")
    init = next(t for t in model.graph.initializer if t.name == sub.input[0])
    shifted = numpy_helper.to_array(init) + 0.25
    init.CopyFrom(numpy_helper.from_array(shifted.astype(np.float32), init.name))
    extracted = onnx_extract.extract_yolov10(save(model, tmp_path, "moved.onnx"))
    with pytest.raises(ValueError):
        convert_models.check_anchors(native("yolov10", extracted), extracted[2])
