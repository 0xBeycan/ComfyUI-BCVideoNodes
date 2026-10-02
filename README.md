# ComfyUI-BCVideoNodes

Video nodes for ComfyUI: a Wan Animate preprocess built from small nodes that
are usable in any video pipeline (wholebody pose with ViTPose-H or Sapiens2, SAM 3.1 person tracking,
face crops, pose and mask checks), a SCAIL-2 preprocess, three samplers
that turn a reference image plus a driving video of any length into a Wan
Animate or SCAIL-2 video of exactly that length, and the video nodes around
them: load a video at the model's generation size, fit a reference image to
it, save the result, compare two videos in the node.

| Node | Id | Category |
|------|----|----------|
| [**Pose Detection**](docs/pose.md#pose-detection) | `BCVPoseDetection` | `BCVideoNodes` |
| [**Pose Config**](docs/pose.md#pose-config) | `BCVPoseConfig` | `BCVideoNodes` |
| [**SAM 3.1 Multiplex Video Track**](docs/sam3-1-multiplex.md#sam-31-multiplex-video-track) | `BCVSAM3VideoTrack` | `BCVideoNodes` |
| [**SAM 3.1 Multiplex Config**](docs/sam3-1-multiplex.md#sam-31-multiplex-config) | `BCVSAM3Config` | `BCVideoNodes` |
| [**Face Crop**](docs/pose.md#face-crop) | `BCVFaceCrop` | `BCVideoNodes` |
| [**Pose Guard**](docs/guards.md#pose-guard-and-mask-guard) | `BCVPoseGuard` | `BCVideoNodes` |
| [**Mask Guard**](docs/guards.md#pose-guard-and-mask-guard) | `BCVMaskGuard` | `BCVideoNodes` |
| [**WanAnimate Preprocess**](docs/wan-animate-preprocess.md#wananimate-preprocess-and-wananimate-preprocess-guard) | `BCVWanAnimatePreprocess` | `BCVideoNodes/Wan/Animate` |
| [**WanAnimate Preprocess Guard**](docs/wan-animate-preprocess.md#wananimate-preprocess-and-wananimate-preprocess-guard) | `BCVWanAnimatePreprocessGuard` | `BCVideoNodes/Wan/Animate` |
| [**Wan Animate Long Video Sampler**](docs/long-video-samplers.md#wan-animate-long-video-sampler-wananimatetovideo) | `BCVWanAnimateLongVideoSampler` | `BCVideoNodes/Wan/Animate` |
| [**Wan Animate 2 Long Video Sampler**](docs/long-video-samplers.md#wan-animate-2-long-video-sampler-wananimate2tovideo) | `BCVWanAnimate2LongVideoSampler` | `BCVideoNodes/Wan/Animate` |
| [**SCAIL-2 Colored Mask**](docs/scail2-preprocess.md#scail-2-colored-mask-and-scail-2-preprocess) | `BCVSCAIL2ColoredMask` | `BCVideoNodes/SCAIL` |
| [**SCAIL-2 Preprocess**](docs/scail2-preprocess.md#scail-2-colored-mask-and-scail-2-preprocess) | `BCVSCAIL2Preprocess` | `BCVideoNodes/SCAIL` |
| [**SCAIL-2 Preprocess Guard**](docs/scail2-preprocess.md#scail-2-preprocess-guard) | `BCVSCAIL2PreprocessGuard` | `BCVideoNodes/SCAIL` |
| [**SCAIL-2 Long Video Sampler**](docs/long-video-samplers.md#scail-2-long-video-sampler-wanscailtovideo) | `BCVSCAIL2LongVideoSampler` | `BCVideoNodes/SCAIL` |
| [**Load Video**](docs/video-nodes.md#load-video) | `BCVLoadVideo` | `BCVideoNodes/Video` |
| [**Get Video Info**](docs/video-nodes.md#get-video-info) | `BCVGetVideoInfo` | `BCVideoNodes/Video` |
| [**Load Reference Image**](docs/video-nodes.md#load-reference-image) | `BCVLoadReferenceImage` | `BCVideoNodes/Video` |
| [**Conform Video**](docs/video-nodes.md#conform-video) | `BCVConformVideo` | `BCVideoNodes/Video` |
| [**Save Video**](docs/video-nodes.md#save-video) | `BCVSaveVideo` | `BCVideoNodes/Video` |
| [**Video Comparer**](docs/video-nodes.md#video-comparer) | `BCVVideoComparer` | `BCVideoNodes/Video` |

## Install

Clone into `ComfyUI/custom_nodes/`, install the requirements into ComfyUI's
Python and restart:

```
pip install -r requirements.txt
```

The only runtime dependency beyond ComfyUI is `opencv-python`; torch, numpy,
scipy, safetensors, tqdm and PyAV (`av`, the video nodes' decoder and encoder)
come with ComfyUI, so the video nodes add no requirement. The previews are
plain JavaScript in `web/js/`, which ComfyUI serves; nothing to build. `onnx` and `huggingface_hub`
are needed only for the offline conversion and upload scripts
(`pip install .[dev]`).

## Documentation

### Video nodes

- [Video nodes](docs/video-nodes.md): Load Video, Get Video Info, Load Reference Image, Conform
  Video, Save Video, Video Comparer, and the player in the node.

### Preprocess nodes

Feed them frames already at the generation size (Load Video loads them so);
the pose images, masks and boxes come out at the size of the frames that went
in, the pose images at Pose Detection's `width` x `height` when both are
connected.

- [Pose and Face Crop](docs/pose.md): Pose Detection, Pose Config, Face Crop.
- [SAM 3.1 Multiplex](docs/sam3-1-multiplex.md): SAM 3.1 Multiplex Video Track and its modes,
  SAM 3.1 Multiplex Config, and the input precedence of the preprocess nodes.
- [Guards](docs/guards.md): Pose Guard and Mask Guard, their fails, warnings and metrics, and the
  reference check.
- [WanAnimate Preprocess](docs/wan-animate-preprocess.md): WanAnimate Preprocess and WanAnimate
  Preprocess Guard.
- [SCAIL-2 Preprocess](docs/scail2-preprocess.md): SCAIL-2 Colored Mask, SCAIL-2 Preprocess,
  SCAIL-2 Preprocess Guard.

### Long video samplers

- [Long video samplers](docs/long-video-samplers.md): the three samplers, their wiring and
  defaults, frames_per_chunk by VRAM, the length math, tail padding and the color anchor.

### Reference

- [Unused outputs](docs/unused-outputs.md): which outputs come out empty when nothing is
  connected to them.
- [Measured against the earlier workflow](docs/measurements.md): time, RAM and VRAM of one Wan
  2.2 Animate replacement workflow, this pack against the earlier packs.
- [Models](docs/models.md): what is downloaded, from where, into which folder.
- [Development](docs/development.md): the tests.

## Roadmap

- **SCAIL-2 pose-driven mode: not planned.** The pack runs SCAIL-2's end-to-end
  mode: the raw driving video is the pose input. The pose-driven mode drives the
  model with a rendered 3D skeleton instead.
  [zai-org/SCAIL-Pose](https://github.com/zai-org/SCAIL-Pose/tree/519c7f54cb972e7f92684213b7ef6c3e05a8f3b2)
  marks end-to-end "(Recommended)", "More accurate and easier than pose-driven
  for most cases", and keeps pose-driven for extremely challenging inputs. It
  would be added only if a need for it appears.
- **Multi-person** (all nodes): multi-person pose and mask are on the roadmap, to be
  implemented and tested later. Today everything is optimised for one person; future
  multi-person changes will take effect in multi-person mode only.

## Licences

The code is MIT (`LICENSE`), except the files vendored from the Alibaba Wan
team's WanAnimate preprocess, which are Apache-2.0 and keep their copyright
header (`libs/pose_utils/LICENSE`):

- `libs/pose_utils/pose2d_utils.py`
- `libs/pose_utils/human_visualization.py`
- `pipelines/face.py`
- `models/common/wrapper.py`
- `models/vitpose/wrapper.py`
- `models/yolo/wrapper.py`
- `models/vitpose/decode.py`

The model weights keep their own licences:

| Model | File | Licence |
|-------|------|---------|
| ViTPose-H wholebody | `vitpose_h_wholebody_fp16.safetensors` | Apache-2.0 |
| YOLOv10x | `yolov10x_fp32.safetensors` | AGPL-3.0 |
| SAM 3.1 | `sam3.1_multiplex_fp16.safetensors` | Meta's SAM License |
| Sapiens2 pose | `sapiens2_pose_*.safetensors` | Sapiens2 License (Meta) |

The person detector's weights (YOLOv10x) are AGPL-3.0. Running them locally
is unaffected; offering a service over a network that runs them (a hosted
workflow, a SaaS) brings AGPL-3.0's network clause into play, which requires
making the corresponding source available to that service's users. With
`bboxes` connected, Pose Detection neither downloads, loads nor runs the
detector.

SAM 3.1 is fetched by ComfyUI from `Comfy-Org/sam3.1` under Meta's SAM
License; this package does not redistribute it.

The Sapiens2 pose weights are Meta's, under the Sapiens2 License; the
converted files are fetched from `beycanai/sapiens2-convrot`, whose
`LICENSE.md` carries it.
