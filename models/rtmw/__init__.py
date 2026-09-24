"""RTMW-l: registers architecture "rtmw" and pose estimator "RTMW-l"."""
from ..common import registry
from .net import RTMWNet
from .wrapper import RTMW

registry.register("architecture", "rtmw", RTMWNet)
registry.register("pose_estimator", "RTMW-l", RTMW, file="rtmw_l_wholebody_384x288_fp32.safetensors")
