"""Convert the upstream ONNX detection and pose models to the safetensors files the nodes load.

Offline only: this is the one place that reads ONNX (it needs `onnx`, which the nodes do
not). For each model it reads the export, extracts the native module's config and weights
(scripts/onnx_extract.py), builds the module, writes its state dict with the config in the
safetensors metadata, loads the written file back the way the nodes do, and checks the two
agree. Rerunning it is how a model gets updated.

    python scripts/convert_models.py --onnx-dir /path/to/onnx --out /path/to/out \\
        [--verify-video clip.mp4 [clip.mp4 ...]] [--frames 30] [--device cuda]

With --verify-video, frames of those clips go through the module built from the ONNX file
and through the module loaded from the safetensors file, and the deviation is reported on
what the nodes use: decoded keypoints the model draws (confidence >= 0.5) and detector
boxes. Without it only a synthetic input is compared.

The upload is a separate step: scripts/upload_models.py.
"""
import argparse
import os
import sys
import time
import types

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# The pack, bound as the package `bcvideonodes` the way ComfyUI and tests/conftest.py bind it, so
# the relative imports of its modules resolve; its __init__ (the nodes) is not run. A process that
# has bound it already keeps that binding.
if "bcvideonodes" not in sys.modules:
    _pack = types.ModuleType("bcvideonodes")
    _pack.__path__ = [ROOT]
    sys.modules["bcvideonodes"] = _pack

from onnx_extract import OnnxGraph, extract_rtmw, extract_vitpose, extract_yolov10  # noqa: E402

from bcvideonodes.models.common import checkpoint  # noqa: E402
from bcvideonodes.models.rtmw.decode import decode_simcc  # noqa: E402
from bcvideonodes.models.vitpose.decode import decode_heatmaps  # noqa: E402
from bcvideonodes.libs.bbox import whole_frame_box  # noqa: E402
from bcvideonodes.models.common.pose_input import pose_crop  # noqa: E402

CONVERTER_VERSION = "1"

# name -> the upstream export, where it comes from, the file written and its precision.
# fp16 only where it pays: ViTPose-H halves from 2.43 GB to 1.22 GB for a largest move of
# one heatmap cell on drawn keypoints; RTMW-l (219 MB) and YOLOv10x (113 MB) stay fp32.
MODELS = {
    "vitpose": {"onnx": "vitpose_h_wholebody_model.onnx", "source": "Kijai/vitpose_comfy (onnx/vitpose_h_wholebody_model.onnx)",
                "file": "vitpose_h_wholebody_fp16.safetensors", "dtype": torch.float16, "extract": extract_vitpose},
    "rtmw": {"onnx": "rtmw_dw_x_l_wholebody_384x288.onnx", "source": "bukuroo/RTMW-ONNX (rtmw-l-384.onnx)",
             "file": "rtmw_l_wholebody_384x288_fp32.safetensors", "dtype": torch.float32, "extract": extract_rtmw},
    "yolov10": {"onnx": "yolov10x.onnx", "source": "onnx-community/yolov10x (onnx/model.onnx)",
                "file": "yolov10x_fp32.safetensors", "dtype": torch.float32, "extract": extract_yolov10},
}

DRAW_THRESHOLD = 0.5


def build_from_onnx(name, onnx_dir):
    """The native module built from the ONNX export, in fp32 on the CPU, and the export's
    anchor constant for YOLO (None otherwise)."""
    spec = MODELS[name]
    graph = OnnxGraph(os.path.join(onnx_dir, spec["onnx"]))
    extracted = spec["extract"](graph)
    config, state = extracted[:2]
    net = checkpoint.build(name, config)
    checkpoint.check_state(net, state, spec["onnx"])
    net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()},
                        strict=True, assign=True)
    anchors = extracted[2] if len(extracted) > 2 else None
    return net, anchors


def check_anchors(net, anchors):
    """The YOLO module computes its anchor points from the feature map sizes; the export
    stores them as a constant. They have to be the same numbers."""
    h, w = net.config["input_size"]
    feats = [torch.zeros(1, 1, h // s, w // s) for s in net.head.strides]
    points, _ = net.head.anchors(feats)
    if points.shape != anchors.shape[1:] or not torch.equal(points, anchors[0]):
        raise ValueError(f"the anchor points the module computes differ from the export's "
                         f"(shape {list(points.shape)} vs {list(anchors.shape[1:])})")


def read_frames(paths, count):
    """`count` frames spread evenly over each clip, RGB 0..1 float32."""
    clips = []
    for path in paths:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f"{path}: no frames")
        wanted = set(np.linspace(0, total - 1, min(count, total)).round().astype(int).tolist())
        frames, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i in wanted:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            i += 1
        cap.release()
        clips.append(frames)
    return clips


def _detector_input(frame, device):
    """`frame` as the detector module takes it: resized to 640x640, [1, 3, 640, 640] on `device`."""
    return torch.from_numpy(cv2.resize(frame, (640, 640)).transpose(2, 0, 1)[None].copy()).to(device)


def detect_person(yolo, frame, device):
    """The highest scoring person row of the detector on `frame`, in frame pixels, or the
    whole frame when there is none."""
    H, W = frame.shape[:2]
    x = _detector_input(frame, device)
    with torch.inference_mode():
        rows = yolo(x)[0].float().cpu().numpy()
    persons = rows[rows[:, 5] == 0]
    if not len(persons) or persons[0, 4] < 0.05:
        return whole_frame_box(W, H)
    x1, y1, x2, y2, score = persons[0, :5]
    return np.array([x1 * W / 640, y1 * H / 640, x2 * W / 640, y2 * H / 640, score])


def pose_crops(frames, boxes, input_size):
    crops, centers, scales = [], [], []
    for frame, box in zip(frames, boxes):
        img_norm, center, scale = pose_crop(frame, box, input_size)
        crops.append(img_norm)
        centers.append(np.array(center))
        scales.append(np.array(scale))
    return crops, centers, scales


def run_pose(name, net, crops, centers, scales, device):
    dtype = next(net.parameters()).dtype
    out = []
    for img, center, scale in zip(crops, centers, scales):
        x = torch.from_numpy(img[None]).to(device, dtype)
        with torch.inference_mode():
            raw = net(x)
        if name == "rtmw":
            simcc_x, simcc_y = (r.float().cpu().numpy() for r in raw)
            out.append(decode_simcc(simcc_x, simcc_y, center[None], scale[None], net.config["input_size"])[0])
        else:
            out.append(decode_heatmaps(raw.float().cpu().numpy(), center[None], scale[None])[0])
    return np.stack(out)


def compare_keypoints(ref, new):
    """Deviation over the keypoints the reference draws, plus how many change sides of the
    draw threshold."""
    drawn = ref[..., 2] >= DRAW_THRESHOLD
    moved = np.linalg.norm(ref[..., :2] - new[..., :2], axis=-1)[drawn]
    flipped = int(((ref[..., 2] >= DRAW_THRESHOLD) != (new[..., 2] >= DRAW_THRESHOLD)).sum())
    conf = np.abs(ref[..., 2] - new[..., 2])[drawn]
    return (f"{int(drawn.sum())} drawn keypoints, moved {int((moved > 0).sum())}, max {moved.max():.3f} px, "
            f"median {np.median(moved):.3f} px, mean {moved.mean():.4f} px; confidence max diff {conf.max():.2e}; "
            f"{flipped} cross the {DRAW_THRESHOLD} draw threshold")


def compare_boxes(ref_rows, new_rows, sizes):
    """Detector rows above 0.05 in frame pixels, row by row (the export sorts them by
    score), and the selected person box."""
    diffs, count_diff, person = [], 0, []
    for ref, new, (H, W) in zip(ref_rows, new_rows, sizes):
        scale = np.array([W / 640, H / 640, W / 640, H / 640])
        keep_ref, keep_new = ref[:, 4] >= 0.05, new[:, 4] >= 0.05
        count_diff += int(keep_ref.sum() != keep_new.sum())
        n = min(keep_ref.sum(), keep_new.sum())
        if n:
            diffs.append(np.abs(ref[:n, :4] - new[:n, :4]) * scale)
        pr, pn = ref[ref[:, 5] == 0], new[new[:, 5] == 0]
        if len(pr) and len(pn):
            person.append(np.abs(pr[0, :4] - pn[0, :4]) * scale)
    d = np.concatenate(diffs) if diffs else np.zeros((0, 4))
    p = np.stack(person) if person else np.zeros((0, 4))
    return (f"{len(d)} boxes >= 0.05: max {d.max() if d.size else 0:.4f} px, median {np.median(d) if d.size else 0:.4f} px; "
            f"{count_diff} frames with a different box count; top person box max {p.max() if p.size else 0:.4f} px "
            f"over {len(p)} frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--onnx-dir", required=True, help="folder with the upstream .onnx files (and ViTPose's .bin)")
    parser.add_argument("--out", required=True, help="folder the safetensors files are written to")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--verify-video", nargs="*", default=[], help="clips whose frames the verification runs on")
    parser.add_argument("--frames", type=int, default=30, help="frames per clip for the verification")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)
    clips = read_frames(args.verify_video, args.frames) if args.verify_video else []
    frames = [f for clip in clips for f in clip]

    # the verification crops come from the detector, so it is converted first when asked for
    order = sorted(args.models, key=lambda n: n != "yolov10")
    detector_boxes = None
    for name in order:
        spec = MODELS[name]
        t0 = time.time()
        ref, anchors = build_from_onnx(name, args.onnx_dir)
        if anchors is not None:
            check_anchors(ref, anchors)
        path = os.path.join(args.out, spec["file"])
        checkpoint.save(ref, name, path, spec["dtype"], extra={"source": spec["source"],
                                                                "converter_version": CONVERTER_VERSION})
        new = checkpoint.load(path, name).to(device)
        ref = ref.to(device)
        size_mb = os.path.getsize(path) / 1e6
        print(f"{name}: wrote {path} ({size_mb:.1f} MB, {str(spec['dtype']).replace('torch.', '')}) "
              f"in {time.time() - t0:.0f}s", flush=True)

        h, w = ref.config["input_size"]
        x = torch.randn(1, 3, h, w, generator=torch.Generator().manual_seed(0)).to(device)
        with torch.inference_mode():
            a, b = ref(x), new(x.to(next(new.parameters()).dtype))
        a, b = (a if isinstance(a, tuple) else (a,)), (b if isinstance(b, tuple) else (b,))
        raw = max(float((u.float() - v.float()).abs().max()) for u, v in zip(a, b))
        print(f"  synthetic input, raw output max |onnx-built - file-built|: {raw:.3e}", flush=True)

        if not frames:
            continue
        if name == "yolov10":
            ref_rows, new_rows, detector_boxes = [], [], []
            for frame in frames:
                xin = _detector_input(frame, device)
                with torch.inference_mode():
                    ref_rows.append(ref(xin)[0].float().cpu().numpy())
                    new_rows.append(new(xin)[0].float().cpu().numpy())
                detector_boxes.append(detect_person(ref, frame, device))
            print(f"  {len(frames)} frames: {compare_boxes(ref_rows, new_rows, [f.shape[:2] for f in frames])}", flush=True)
        else:
            if detector_boxes is None:
                yolo, _ = build_from_onnx("yolov10", args.onnx_dir)
                yolo = yolo.to(device)
                detector_boxes = [detect_person(yolo, f, device) for f in frames]
                del yolo
            crops, centers, scales = pose_crops(frames, detector_boxes, tuple(ref.config["input_size"]))
            kp_ref = run_pose(name, ref, crops, centers, scales, device)
            kp_new = run_pose(name, new, crops, centers, scales, device)
            print(f"  {len(frames)} crops: {compare_keypoints(kp_ref, kp_new)}", flush=True)
        del ref, new
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
