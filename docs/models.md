# Models

Everything is downloaded on first use; nothing has to be fetched by hand.

- Detection and pose models: from
  [huggingface.co/beycanai/BCVideoNodes-models](https://huggingface.co/beycanai/BCVideoNodes-models)
  into `ComfyUI/models/detection/` (`yolov10x_fp32.safetensors`,
  `vitpose_h_wholebody_fp16.safetensors`; a model is fetched when a node
  first needs it). They are native torch modules stored as safetensors and
  loaded and offloaded by ComfyUI's model management; `onnx` is not needed.
  `scripts/convert_models.py` rebuilds them from the upstream ONNX exports.
- SAM 3.1: ComfyUI's own `sam3.1_multiplex_fp16.safetensors`, from
  `Comfy-Org/sam3.1` into `ComfyUI/models/checkpoints/` when it is missing.
- Sapiens2 pose (a Sapiens2 `pose_model`): from
  [huggingface.co/beycanai/sapiens2-convrot](https://huggingface.co/beycanai/sapiens2-convrot)
  into `ComfyUI/models/detection/`, the file of the chosen model only
  (`sapiens2_pose_<size>_<bf16|int8_convrot>.safetensors`).
  `scripts/convert_sapiens2.py` writes the bf16 file from the transformers
  checkpoint; the int8 ConvRot file is made from it with convert_to_quant.
