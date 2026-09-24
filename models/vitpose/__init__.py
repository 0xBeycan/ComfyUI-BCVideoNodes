"""ViTPose-H: registers architecture "vitpose" and pose estimator "ViTPose-H"."""
from ..common import registry
from .net import ViTPoseNet
from .wrapper import ViTPose

registry.register("architecture", "vitpose", ViTPoseNet)
registry.register("pose_estimator", "ViTPose-H", ViTPose, file="vitpose_h_wholebody_fp16.safetensors")
