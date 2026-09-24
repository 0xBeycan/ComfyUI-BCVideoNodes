---
license: other
library_name: safetensors
tags:
  - comfyui
  - pose-estimation
  - object-detection
  - wholebody
pipeline_tag: keypoint-detection
---

# BCVideoNodes models

The detection and pose models used by
[ComfyUI-BCVideoNodes](https://github.com/0xBeycan/ComfyUI-BCVideoNodes).

The nodes download whatever they need from here on first use, so nothing in this repository
has to be fetched by hand. It exists so the weights have a stable home, and so anyone can
see exactly which files a node is pulling.

## What is here

| file | model | source | licence |
|---|---|---|---|
| `vitpose_h_wholebody_fp16.safetensors` | ViTPose-H, 133 COCO-WholeBody keypoints | [Kijai/vitpose_comfy](https://huggingface.co/Kijai/vitpose_comfy) | Apache-2.0 |
| `yolov10x_fp32.safetensors` | YOLOv10-X person detection | [onnx-community/yolov10x](https://huggingface.co/onnx-community/yolov10x) | AGPL-3.0 |

`vitpose_h_wholebody` is fp16; the detector is fp32. See below.

## Why these are not the original files

Upstream ships these as ONNX. The nodes run them as plain torch modules instead — the
architecture is written out by hand and the weights are loaded into it, which removes the
`onnx` dependency, removes a protobuf parse and a graph walk from every model load, and
replaces ViTPose's two-file external-data export (a 420 KB graph plus a 2.5 GB `.bin`) with
one file.

Each file carries its architecture in the safetensors metadata, so the loader needs nothing
but the file.

## Precision

ViTPose-H is fp16, YOLOv10x is fp32, because fp16 is not free and only ViTPose-H is large
enough for it to pay:

| model | fp32 | fp16 | saved |
|---|---|---|---|
| ViTPose-H | 2.43 GB | 1.22 GB | 1.2 GB |
| YOLOv10x | 113 MB | 57 MB | 56 MB |

What it costs, measured on 60 real frames over every keypoint the model draws, against the
same model in fp32 on identical crops: ViTPose-H, 5230 keypoints compared, 20 moved, largest
move **1 heatmap cell**. Worth paying for 1.2 GB.

Keypoints the model does not draw are excluded from those figures, and that matters: on
those the score distribution is flat, so the argmax is noise in either precision, and fp16
appears to move them by up to 215 px. That number is what a flat distribution does, not what
fp16 does.

## Conversion

The converter lives in the node repository,
[`scripts/convert_models.py`](https://github.com/0xBeycan/ComfyUI-BCVideoNodes/blob/main/scripts/convert_models.py). It reads the upstream ONNX file, builds the
native module from it, and writes the module's own state dict plus its architecture. Rerun
it to update a model; do not hand-edit the files here.

## Credit

The pose models are OpenMMLab's and the ViTPose authors' work; YOLOv10 is
[THU-MIG](https://github.com/THU-MIG/yolov10). This repository only re-packages them. The
wholebody pose utilities in the node repository are vendored from the Alibaba Wan team's
WanAnimate preprocess and keep their copyright header.
