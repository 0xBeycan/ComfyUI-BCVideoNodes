"""Save Video and Video Comparer: IMAGE frames written as a video file one frame at a time
(libs/video_encode.py), shown in the node by the pack's player (web/js). The Video Comparer writes
its two clips side by side into one temporary H.264 file (libs/video_compare.py) through the same
encoder path, so its colour conversion and tags are Save Video's h264-mp4."""

from ..libs.video_encode import CODECS, union, widget_values
from .common import VIDEO

UI_KEY = "bcv_video"  # the ui payload the web player reads
FIRST_CODEC = next(iter(CODECS))
FPS = ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                 "tooltip": "Frames per second of the video. Wire Get Video Info's loaded_fps to keep the source's timing."})
COMPARER_CODEC = "h264-mp4"
COMPARER_CRF = 18


def _preview(file, subfolder, folder_type, fps, frames, audio, **extra):
    """One entry of the ui payload the player reads."""
    return {"filename": file, "subfolder": subfolder, "type": folder_type, "fps": float(fps), "frames": frames,
            "audio": audio, **extra}


def _counted(images):
    """The frames of `images`, advancing ComfyUI's progress bar as each one is written."""
    from comfy.utils import ProgressBar

    bar = ProgressBar(images.shape[0])
    for frame in images:
        yield frame
        bar.update(1)


class BCVSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        crf = [CODECS[name]["crf"] for name in CODECS]
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The frames, written one at a time."}),
                "fps": FPS,
                "filename_prefix": ("STRING", {"default": "video/ComfyUI", "tooltip": "The file name, with subfolders if it has them (video/ComfyUI -> video/ComfyUI_00001_.mp4). A counter is appended; %date:...% and the other ComfyUI name tokens work."}),
                "codec": (list(CODECS), {"default": FIRST_CODEC, "bcv_codecs": {name: widget_values(name) for name in CODECS},
                                         "tooltip": "The video codec and container. h264-mp4 plays everywhere; h265-mp4 is smaller at the same quality; av1-webm smaller still, slower to encode; vp9-webm plays in every browser. Changing it resets crf, preset and pix_fmt to the codec's defaults. The audio track is AAC in mp4, Opus in webm."}),
                "crf": ("INT", {"default": CODECS[FIRST_CODEC]["crf"]["default"], "min": min(c["min"] for c in crf),
                                "max": max(c["max"] for c in crf), "step": 1,
                                "tooltip": "Constant quality: lower is better quality and a larger file. Range and default follow the codec (h264/h265 0-51, av1 1-63, vp9 0-63)."}),
                "preset": (union("preset"), {"default": CODECS[FIRST_CODEC]["preset"]["default"],
                                             "tooltip": "Encode speed against file size at the same crf: slower presets give a smaller file. h264/h265: ultrafast ... placebo; av1: 0 (slowest) ... 13; vp9: cpu-used 0 (slowest) ... 5."}),
                "pix_fmt": (union("pix_fmt"), {"default": CODECS[FIRST_CODEC]["pix_fmt"]["default"],
                                               "tooltip": "yuv420p plays everywhere. yuv420p10le: 10 bits, less banding, not every player. yuv444p (h264): no chroma subsampling, for editing; browsers do not play it."}),
                "save_output": ("BOOLEAN", {"default": True, "tooltip": "On: into the output folder. Off: into the temp folder, which ComfyUI empties when it starts."}),
                "save_metadata": ("BOOLEAN", {"default": True, "tooltip": "Writes the workflow and the prompt into the file, so dropping the file on ComfyUI loads the workflow. ComfyUI's --disable-metadata turns it off for every saver."}),
            },
            "optional": {
                "audio": ("AUDIO", {"tooltip": "Muxed in and cut to the video's length."}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = VIDEO
    DESCRIPTION = "Saves the frames as a video file, encoded one frame at a time: BT.709 colour in tv range with its tags, the audio cut to the video's length, the workflow in the file. Plays in the node."

    def save(self, images, fps, filename_prefix, codec, crf, preset, pix_fmt, save_output, save_metadata, audio=None,
             prompt=None, extra_pnginfo=None):
        import os

        import folder_paths
        from comfy.cli_args import args

        from ..libs.video_encode import check_encoders, check_settings, has_audio, write_video

        # before the folders are made: a setting the codec does not take fails with nothing written
        check_settings(codec, crf, preset, pix_fmt)
        check_encoders(codec)
        root = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        folder, name, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, root, images.shape[2], images.shape[1])
        file = f"{name}_{counter:05}_.{CODECS[codec]['extension']}"
        metadata = None
        if save_metadata and not args.disable_metadata:
            metadata = {**(extra_pnginfo or {}), **({"prompt": prompt} if prompt is not None else {})}
        audio = audio if has_audio(audio) else None
        frames = write_video(os.path.join(folder, file), _counted(images), fps, codec, crf, preset, pix_fmt, audio, metadata)
        entry = _preview(file, subfolder, "output" if save_output else "temp", fps, frames, audio is not None,
                         format=f"video/{CODECS[codec]['extension']}")
        return {"ui": {UI_KEY: [entry]}}


class BCVVideoComparer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": FPS,
            },
            "optional": {
                "video_a": ("IMAGE", {"tooltip": "Fills the node."}),
                "video_b": ("IMAGE", {"tooltip": "Drawn over A from the left edge up to the pointer."}),
                "audio": ("AUDIO", {"tooltip": "Played with the clips, cut to their length."}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = VIDEO
    DESCRIPTION = "Compares two videos in the node with a draggable divider: A fills the node, B is drawn from the left edge up to the pointer. Both are written side by side into one temporary H.264 file, so they play in step; clips of different length are cut to the shorter, a smaller frame is letterboxed into the larger."

    def compare(self, fps, video_a=None, video_b=None, audio=None):
        import os
        import uuid

        import folder_paths

        from ..libs.video_compare import side_by_side_frames, side_by_side_geometry
        from ..libs.video_encode import has_audio, write_video

        sides = [(tag, images) for tag, images in (("A", video_a), ("B", video_b)) if images is not None and images.shape[0] > 0]
        if not sides:
            return {"ui": {UI_KEY: []}}
        frames, h, w = side_by_side_geometry(sides)
        temp = folder_paths.get_temp_directory()
        os.makedirs(temp, exist_ok=True)
        file = f"bcv.compare.{uuid.uuid4().hex[:8]}.{CODECS[COMPARER_CODEC]['extension']}"
        audio = audio if has_audio(audio) else None
        # a keyframe every second keeps the seek bar responsive
        write_video(os.path.join(temp, file), side_by_side_frames(sides, frames, h, w), fps, COMPARER_CODEC, COMPARER_CRF,
                    CODECS[COMPARER_CODEC]["preset"]["default"], CODECS[COMPARER_CODEC]["pix_fmt"]["default"], audio,
                    keyframe_interval=max(1, round(fps)))
        entry = _preview(file, "", "temp", fps, {tag: int(images.shape[0]) for tag, images in sides}, audio is not None,
                         sides=[tag for tag, _ in sides])
        return {"ui": {UI_KEY: [entry]}}
