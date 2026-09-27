"""ViTPose-H: registers architecture "vitpose" and pose estimator "ViTPose-H".

The weights, and so the forward, are fp16; Wan and Kijai run the fp32 ONNX model. Against
onnxruntime fp32 on the 3,073 crops of the base run, 27 draw decisions flip (4 body, 23 hand
keypoints, all near the draw threshold); the same native module in fp32 flips none (largest move
0.065 px), so the flips are fp16's, not the port's. The pose-parity measurement ran fp32 as its own
arm and found no net gain; the owner ruled that fp16 stays."""
from ..common import registry
from .net import ViTPoseNet
from .wrapper import ViTPose

registry.register("architecture", "vitpose", ViTPoseNet)
registry.register("pose_estimator", "ViTPose-H", ViTPose, file="vitpose_h_wholebody_fp16.safetensors")
