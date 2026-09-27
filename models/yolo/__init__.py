"""YOLOv10x: registers architecture "yolov10" and person detector "YOLOv10x".

Wan's preprocess (preprocess_data.py) and Kijai's node run YOLOv10m; this pack has run YOLOv10x
since its pose nodes were added, and why is not recorded."""
from ..common import registry
from .net import YOLOv10Net
from .wrapper import Yolo

registry.register("architecture", "yolov10", YOLOv10Net)
registry.register("person_detector", "YOLOv10x", Yolo, file="yolov10x_fp32.safetensors")
