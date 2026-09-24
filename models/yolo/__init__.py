"""YOLOv10x: registers architecture "yolov10" and person detector "YOLOv10x"."""
from ..common import registry
from .net import YOLOv10Net
from .wrapper import Yolo

registry.register("architecture", "yolov10", YOLOv10Net)
registry.register("person_detector", "YOLOv10x", Yolo, file="yolov10x_fp32.safetensors")
