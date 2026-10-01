"""video_info (BCV_VIDEO_INFO), what Load Video loaded, as Get Video Info and Load Reference Image
read it. A plain dict at runtime; VideoInfo is its annotation and fixes its key order. Standard
library only: the node module reads the order at import."""
from typing import TypedDict


class VideoInfo(TypedDict):
    """The model, resolution and resolved orientation (portrait or landscape) Load Video loaded
    with, the source's frame rate, frame count, duration and displayed size, then the loaded
    batch's."""
    model: str
    resolution: str
    orientation: str
    source_fps: float
    source_frame_count: int
    source_duration: float
    source_width: int
    source_height: int
    loaded_fps: float
    loaded_frame_count: int
    loaded_duration: float
    loaded_width: int
    loaded_height: int


# ComfyUI's type of a field's Python type: Get Video Info's output types, in VideoInfo's order
COMFY_TYPES = {str: "STRING", int: "INT", float: "FLOAT"}
