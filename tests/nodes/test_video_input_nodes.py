"""The video input nodes' ComfyUI surface: Load Video's widgets as the contract fixes them (order,
defaults, the resolution sizes for the frontend, no core video upload, precision the last widget,
fp16 by default), its validation of a
resolution of another model, its file of the output or temp folder the value names (a video
dragged from the queue) and never one outside it, Get Video Info's outputs in video_info's order
(Load Video's audio itself first), Load Reference
Image's preview payload, Conform Video. ComfyUI's input, output and temp folders are a test's tmp
dirs."""
import os

import pytest

torch = pytest.importorskip("torch")
folder_paths = pytest.importorskip("folder_paths")
pytest.importorskip("av")

from video_input_fakes import grey_clip, ramp_audio, video  # noqa: E402


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """ComfyUI's input, output and temp folders moved into tmp_path."""
    for name in ("input", "output", "temp"):
        (tmp_path / name).mkdir()
        monkeypatch.setattr(folder_paths, f"{name}_directory", str(tmp_path / name))
    return tmp_path


def test_load_video_widgets(folders):
    grey_clip(folders / "input" / "clip.mkv", 3)
    (folders / "input" / "notes.txt").write_text("not a video")
    required = video.BCVLoadVideo.INPUT_TYPES()["required"]
    assert list(required) == ["video", "model", "resolution", "orientation", "force_fps", "start_frame", "frame_count"]
    files, options = required["video"]
    assert files == ["clip.mkv"] and "video_upload" not in options
    assert required["model"][0] == ["Wan", "SCAIL", "None"] and required["model"][1]["default"] == "Wan"
    labels, options = required["resolution"]
    assert labels == ["480p", "720p", "512p", "704p", "1080p", "source"] and options["default"] == "720p"
    assert options["bcv_sizes"] == {
        "Wan": {"480p": [480, 832], "720p": [720, 1280], "source": None},
        "SCAIL": {"512p": [512, 896], "704p": [704, 1280], "source": None},
        "None": {"480p": [480, 854], "720p": [720, 1280], "1080p": [1080, 1920], "source": None}}
    assert set(options) == {"default", "bcv_sizes", "tooltip"}
    assert required["orientation"][0] == ["auto", "landscape", "portrait"] and required["orientation"][1]["default"] == "auto"
    assert required["force_fps"][0] == "STRING" and required["force_fps"][1]["default"] == ""
    assert required["start_frame"][0] == "INT" and required["start_frame"][1]["default"] == 1
    assert required["start_frame"][1]["min"] == 1
    assert required["frame_count"][0] == "STRING" and required["frame_count"][1]["default"] == ""
    # the last widget, optional: a workflow saved before it keeps its widget values (and gets fp16)
    optional = video.BCVLoadVideo.INPUT_TYPES()["optional"]
    assert list(optional) == ["precision"]
    assert optional["precision"][0] == ["fp32", "fp16"] and optional["precision"][1]["default"] == "fp16"
    assert set(optional["precision"][1]) == {"default", "tooltip"}
    assert video.BCVLoadVideo.RETURN_TYPES == ("IMAGE", "AUDIO", "BCV_VIDEO_INFO")


def test_load_video_validation(folders):
    grey_clip(folders / "input" / "clip.mkv", 3)
    validate = video.BCVLoadVideo.VALIDATE_INPUTS
    assert validate(video="clip.mkv", model="Wan", resolution="720p") is True
    assert validate(video="clip.mkv", model="Wan", resolution="512p") == (
        "resolution '512p' does not belong to model Wan; pick one of 480p, 720p, source.")
    assert validate(video="clip.mkv", model="None", resolution="1080p") is True
    assert validate(video="clip.mkv", model="SCAIL", resolution="source") is True
    assert validate(video="clip.mkv", model="Wan", resolution="1080p") == (
        "resolution '1080p' does not belong to model Wan; pick one of 480p, 720p, source.")
    assert validate(video="gone.mp4", model="SCAIL", resolution="704p") == "Invalid video file: gone.mp4"
    assert validate(video="clip.mkv") is True  # model and resolution linked: checked when it runs


def test_load_video_loads_from_the_input_folder(folders):
    grey_clip(folders / "input" / "clip.mkv", 9)
    images, audio, info = video.BCVLoadVideo().load("clip.mkv", "SCAIL", "512p", "auto", "", 1, "")
    assert images.shape == (9, 512, 896, 3) and audio is None and info["resolution"] == "512p"
    images, _, info = video.BCVLoadVideo().load("clip.mkv", "None", "source", "auto", "", 1, "8", "fp32")
    assert images.shape == (8, 32, 64, 3) and (info["model"], info["resolution"]) == ("None", "source")
    assert images.dtype == torch.float32
    half, _, half_info = video.BCVLoadVideo().load("clip.mkv", "None", "source", "auto", "", 1, "8")  # fp16, the default
    assert half.dtype == torch.float16 and torch.equal(half, images.half()) and half_info == info


@pytest.mark.parametrize("folder", ["output", "temp"])
def test_load_video_loads_a_file_of_the_folder_its_value_names(folders, folder):
    # the value a video dragged from the queue gets: core's annotated file path, as Load Image takes it
    grey_clip(folders / "input" / "clip.mkv", 5)
    path = grey_clip(folders / folder / "clip.mkv", 9)
    name = f"clip.mkv [{folder}]"
    assert video.BCVLoadVideo.VALIDATE_INPUTS(video=name) is True
    assert video.BCVLoadVideo.IS_CHANGED(name) == os.path.getmtime(path)
    images, _, info = video.BCVLoadVideo().load(name, "Wan", "480p", "auto", "", 1, "")
    assert images.shape == (9, 480, 832, 3) and info["source_frame_count"] == 9


@pytest.mark.parametrize("name", ["../outside.mkv [output]", "{root}/outside.mkv [temp]", "link.mkv [output]"])
def test_load_video_never_leaves_the_folder_its_value_names(folders, name):
    grey_clip(folders / "outside.mkv", 5)
    os.symlink(folders / "outside.mkv", folders / "output" / "link.mkv")
    name = name.format(root=folders)
    assert video.BCVLoadVideo.VALIDATE_INPUTS(video=name) == f"Invalid video file: {name}"
    with pytest.raises(ValueError, match="Invalid file path"):
        video.BCVLoadVideo().load(name, "Wan", "480p", "auto", "", 1, "")


def test_get_video_info_outputs_every_field_in_order():
    audio = {"waveform": torch.zeros(1, 2, 48000), "sample_rate": 48000}
    info = {"audio": audio, "model": "Wan", "resolution": "720p", "orientation": "portrait", "source_fps": 30.0,
            "source_frame_count": 612, "source_duration": 20.4, "source_width": 1080, "source_height": 1920,
            "loaded_fps": 30.0, "loaded_frame_count": 609, "loaded_duration": 20.3, "loaded_width": 720,
            "loaded_height": 1280}
    node = video.BCVGetVideoInfo
    assert node.RETURN_NAMES == tuple(info)
    assert node.RETURN_TYPES == ("AUDIO", "STRING", "STRING", "STRING", "FLOAT", "INT", "FLOAT", "INT", "INT", "FLOAT",
                                 "INT", "FLOAT", "INT", "INT")
    out = node().get(info)
    assert out == tuple(info.values()) and out[0] is audio  # Load Video's audio itself, first


def test_a_typed_force_fps_is_the_rate_save_video_writes(folders):
    # Load Video -> Get Video Info's loaded_fps -> Save Video: the file's rate is force_fps as typed,
    # and with force_fps empty the video's own
    from fractions import Fraction

    from video_output_fakes import decode
    from video_output_fakes import video as output

    grey_clip(folders / "input" / "clip.mkv", 9, Fraction(30000, 1001))
    for force, rate in (("30", Fraction(30)), ("", Fraction(30000, 1001))):
        images, _, info = video.BCVLoadVideo().load("clip.mkv", "None", "source", "auto", force, 1, "", "fp32")
        fps = video.BCVGetVideoInfo().get(info)[list(info).index("loaded_fps")]
        saved = output.BCVSaveVideo().save(images, fps, "clip", "h264-mp4", 19, "medium", "yuv420p", True, False)
        assert decode(str(folders / "output" / saved["ui"][output.UI_KEY][0]["filename"]))["rate"] == rate


def test_get_video_info_hands_on_load_videos_audio(folders):
    # the audio Save Video and the Video Comparer can take from Get Video Info instead of Load Video
    grey_clip(folders / "input" / "clip.mkv", 9, audio=ramp_audio(0.3))
    _, audio, info = video.BCVLoadVideo().load("clip.mkv", "Wan", "480p", "auto", "", 1, "")
    assert audio is not None and video.BCVGetVideoInfo().get(info)[0] is audio


def test_load_reference_image_fits_and_previews(folders):
    from PIL import Image

    Image.new("RGBA", (30, 40), (255, 0, 0, 128)).save(folders / "input" / "ref.png")
    required = video.BCVLoadReferenceImage.INPUT_TYPES()["required"]
    assert list(required) == ["image", "video_info"] and required["image"][1]["image_upload"] is True
    assert required["image"][0] == ["ref.png"]
    out = video.BCVLoadReferenceImage().load("ref.png", {"loaded_width": 12, "loaded_height": 16})
    image, mask = out["result"]
    assert image.shape == (1, 16, 12, 3) and mask.shape == (1, 16, 12)
    assert torch.allclose(mask, torch.full((1, 16, 12), 1 - 128 / 255))
    (preview,) = out["ui"]["images"]
    assert preview["type"] == "temp" and (folders / "temp" / preview["subfolder"] / preview["filename"]).is_file()
    assert video.BCVLoadReferenceImage.VALIDATE_INPUTS(image="gone.png") == "Invalid image file: gone.png"


def test_conform_video_node():
    required = video.BCVConformVideo.INPUT_TYPES()["required"]
    assert list(required) == ["images", "fit", "method"]
    assert required["fit"] == (["crop", "pad"], required["fit"][1]) and required["fit"][1]["default"] == "crop"
    assert required["method"][0][0] == "lanczos" and required["method"][1]["default"] == "lanczos"
    (out,) = video.BCVConformVideo().conform(torch.rand((1, 896, 512, 3)), "crop", "lanczos")
    assert out.shape == (1, 854, 480, 3)
