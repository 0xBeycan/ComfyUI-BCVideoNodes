"""Load Video, Get Video Info, Load Reference Image and Conform Video, and the server route Load
Video's preview asks what the loader will load (PLAN_ROUTE)."""
from functools import lru_cache, partial

from ..libs import log
from ..libs.video_info import COMFY_TYPES, VideoInfo
from .common import VIDEO, prompt_server
from .unused_outputs import LINK_INPUTS, drop_unlinked_heavy

# GET, with Load Video's widget values as the query (PLAN_PARAMS): web/js/load_video.js
PLAN_ROUTE = "/bcvideonodes/load_video/plan"
PLAN_PARAMS = ("video", "model", "resolution", "orientation", "force_fps", "start_frame", "frame_count")


def _input_files(kind):
    """The files of ComfyUI's input folder whose content type is `kind` ("video", "image"), sorted."""
    import os

    import folder_paths

    folder = folder_paths.get_input_directory()
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return sorted(folder_paths.filter_files_content_types(files, [kind]))


def _missing_file(name, kind):
    """None when the input file `name` exists, else the message ComfyUI shows."""
    import folder_paths

    return None if folder_paths.exists_annotated_filepath(name) else f"Invalid {kind} file: {name}"


class BCVLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        from ..libs import video_sizes as sizes

        return {
            "required": {
                "video": (_input_files("video"), {"tooltip": "The video file, from ComfyUI's input folder."}),
                "model": (list(sizes.MODELS), {"default": "Wan", "tooltip": "The model the video is loaded for: its generation sizes and its frame rule (4n+1 frames for both)."}),
                "resolution": (sizes.RESOLUTIONS, {
                    "default": "720p",
                    "bcv_sizes": {name: model["sizes"] for name, model in sizes.MODELS.items()},
                    "tooltip": "The generation size by its short edge; the labels follow model. Wan: 480p (480x832), 720p (720x1280). SCAIL: 512p (512x896), 704p (704x1280). Portrait sizes; landscape swaps them. The video is centre-cropped to that aspect and resized with lanczos."}),
                "orientation": (sizes.ORIENTATIONS, {"default": sizes.AUTO, "tooltip": "auto: portrait when the video is taller than wide, otherwise landscape (a square video is landscape). landscape / portrait: that orientation, reached by a centre crop."}),
                "force_fps": ("STRING", {"default": "", "tooltip": "Empty: the video's own frame rate. A number above 0, at most the video's own rate: real frames kept or dropped on that rate's time grid (never blended or repeated; a rate above the video's is an error, one within 0.01% of it is the video's rate)."}),
                "start_frame": ("INT", {"default": 1, "min": 1, "max": 2 ** 31 - 1, "step": 1, "tooltip": "The first frame loaded, counted from 1, after force_fps."}),
                "frame_count": ("STRING", {"default": "", "tooltip": "Empty: every frame from start_frame on. A whole number of at least 1: that many frames, counted after force_fps. The count is then cut to the model's 4n+1."}),
            },
            "hidden": dict(LINK_INPUTS),
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "BCV_VIDEO_INFO")
    RETURN_NAMES = ("images", "audio", "video_info")
    # dropped at return when nothing links it (nodes/unused_outputs.py): video_info counts the
    # decoded frames, so the decode runs either way
    HEAVY_OUTPUTS = ("images",)
    FUNCTION = "load"
    CATEGORY = VIDEO
    DESCRIPTION = "Loads a video one frame at a time, centre-cropped and resized (lanczos) to the model's generation size straight into the output, so the full-resolution clip never sits in memory. Colours follow the file's own colour tags. Outputs the frames, the audio of the loaded range (None when the file has no audio) and video_info."

    def load(self, video, model, resolution, orientation, force_fps, start_frame, frame_count, prompt_graph=None,
             unique_id=None):
        import folder_paths

        from ..pipelines import video_input

        path = folder_paths.get_annotated_filepath(video)
        return drop_unlinked_heavy(type(self), video_input.load_video(path, model, resolution, orientation, force_fps,
                                                                      start_frame, frame_count), prompt_graph, unique_id)

    @classmethod
    def IS_CHANGED(cls, video, **kwargs):
        import os

        import folder_paths

        return os.path.getmtime(folder_paths.get_annotated_filepath(video))

    @classmethod
    def VALIDATE_INPUTS(cls, video=None, model=None, resolution=None):
        from ..libs import video_sizes as sizes

        if video is not None and (missing := _missing_file(video, "video")):
            return missing
        if model is not None and resolution is not None:
            try:
                sizes.model_size(model, resolution)
            except ValueError as error:
                return str(error)
        return True


def register_plan_route():
    """Adds GET PLAN_ROUTE to ComfyUI's server and returns its handler; None outside ComfyUI,
    where there is no route and Load Video's preview shows no frame count."""
    server = prompt_server()
    if server is None:
        log.info("no ComfyUI server: Load Video's preview gets no frame count")
        return None
    server.routes.get(PLAN_ROUTE)(plan_route)
    return plan_route


async def plan_route(request):
    """PLAN_ROUTE: the LoadPreview (pipelines/video_input.py) of the input file `video` with the
    other widget values, as JSON. 400 for a missing parameter, a start_frame that is not a whole
    number, or a `video` outside ComfyUI's input folder."""
    import asyncio

    from aiohttp import web

    from ..pipelines import video_input

    query = request.rel_url.query
    missing = [name for name in PLAN_PARAMS if name not in query]
    if missing:
        return web.Response(status=400, text=f"missing {', '.join(missing)}")
    try:
        start_frame = int(query["start_frame"])
    except ValueError:
        return web.Response(status=400, text="start_frame is not a whole number")
    path = _input_path(query["video"])
    if path is None:
        return web.Response(status=400, text="video must be a file of ComfyUI's input folder")
    if error := _missing_file(query["video"], "video"):
        return web.json_response(video_input.LoadPreview(source=None, info=None, available=None, error=error))
    # the probe reads the whole container: off the event loop
    answer = await asyncio.get_running_loop().run_in_executor(None, partial(
        video_input.preview, path, query["model"], query["resolution"], query["orientation"], query["force_fps"],
        start_frame, query["frame_count"], probe=_probe))
    return web.json_response(answer)


def _input_path(name):
    """The path Load Video loads `name` from, or None when `name` leaves ComfyUI's input folder:
    another folder's annotation, an absolute path, a `..` (refused as core's /view refuses them),
    or a path that resolves outside the folder."""
    import os

    import folder_paths

    bare, folder = folder_paths.annotated_filepath(name)
    if not bare or folder not in (None, folder_paths.get_input_directory()):
        return None
    if bare[0] in "/\\" or os.path.isabs(bare) or ".." in bare:
        return None
    try:
        return folder_paths.get_annotated_filepath(name)  # refuses a path that resolves outside the folder
    except ValueError:
        return None


def _probe(path):
    """video_decode.probe of `path`, kept while the file's size and modification time stay."""
    import os

    stat = os.stat(path)
    return dict(_probe_file(path, stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=16)
def _probe_file(path, mtime, size):
    from ..libs import video_decode

    return video_decode.probe(path)


class BCVGetVideoInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_info": ("BCV_VIDEO_INFO", {"tooltip": "From Load Video."})}}

    RETURN_TYPES = tuple(COMFY_TYPES[kind] for kind in VideoInfo.__annotations__.values())
    RETURN_NAMES = tuple(VideoInfo.__annotations__)
    FUNCTION = "get"
    CATEGORY = VIDEO
    DESCRIPTION = "Splits Load Video's video_info: the model, resolution and orientation it loaded with, then the source's frame rate, frame count, duration and size, then the loaded frames'."

    def get(self, video_info):
        return tuple(video_info[name] for name in self.RETURN_NAMES)


class BCVLoadReferenceImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": (_input_files("image"), {"image_upload": True, "tooltip": "The reference image, from ComfyUI's input folder."}),
                "video_info": ("BCV_VIDEO_INFO", {"tooltip": "From Load Video: the image is fitted to its loaded_width x loaded_height, so the reference matches the video's model, resolution and orientation."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load"
    CATEGORY = VIDEO
    DESCRIPTION = "Loads an image as core's Load Image does (EXIF orientation, alpha as the mask) and fits it to the video's loaded size by a centre crop and lanczos, the resize Load Video applies to the frames."

    def load(self, image, video_info):
        import folder_paths
        from comfy_api.latest import ui

        from ..pipelines import video_input

        path = folder_paths.get_annotated_filepath(image)
        fitted, mask = video_input.load_reference_image(path, video_info["loaded_width"], video_info["loaded_height"])
        return {"ui": ui.PreviewImage(fitted).as_dict(), "result": (fitted, mask)}

    @classmethod
    def IS_CHANGED(cls, image, **kwargs):
        import hashlib

        import folder_paths

        with open(folder_paths.get_annotated_filepath(image), "rb") as file:
            return hashlib.sha256(file.read()).hexdigest()

    @classmethod
    def VALIDATE_INPUTS(cls, image=None):
        if image is not None and (missing := _missing_file(image, "image")):
            return missing
        return True


class BCVConformVideo:
    @classmethod
    def INPUT_TYPES(cls):
        from ..libs import resize

        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The video frames."}),
                "fit": (resize.FITS, {"default": resize.CROP, "tooltip": "crop: resized to cover the standard size, the overflow cut off centred. pad: resized to fit inside it, centred between black bars."}),
                "method": (resize.METHODS, {"default": resize.LANCZOS, "tooltip": "The resampling method (comfy.utils.common_upscale's)."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "conform"
    CATEGORY = VIDEO
    DESCRIPTION = "Fits a video to the nearest standard size, 480p (480x854), 720p (720x1280) or 1080p (1080x1920), landscape swapped: the one whose scale on the short edge is closest to 1, in the frames' own orientation. A pixel resize, not diffusion. The input is returned untouched when it already has that size."

    def conform(self, images, fit, method):
        from ..pipelines import video_input

        return (video_input.conform_video(images, fit, method),)
