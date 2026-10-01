"""The video input nodes' ComfyUI surface: Load Video's widgets as the contract fixes them (order,
defaults, the resolution sizes for the frontend, no core video upload), its validation of a
resolution of another model, Get Video Info's outputs in video_info's order, Load Reference
Image's preview payload, Conform Video. ComfyUI's input and temp folders are a test's tmp dirs."""
import pytest

torch = pytest.importorskip("torch")
folder_paths = pytest.importorskip("folder_paths")
pytest.importorskip("av")

from video_input_fakes import grey_clip, video  # noqa: E402


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """ComfyUI's input and temp folders moved into tmp_path."""
    (tmp_path / "input").mkdir()
    (tmp_path / "temp").mkdir()
    monkeypatch.setattr(folder_paths, "input_directory", str(tmp_path / "input"))
    monkeypatch.setattr(folder_paths, "temp_directory", str(tmp_path / "temp"))
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
    images, _, info = video.BCVLoadVideo().load("clip.mkv", "None", "source", "auto", "", 1, "8")
    assert images.shape == (8, 32, 64, 3) and (info["model"], info["resolution"]) == ("None", "source")


def test_get_video_info_outputs_every_field_in_order():
    info = {"model": "Wan", "resolution": "720p", "orientation": "portrait", "source_fps": 30.0,
            "source_frame_count": 612, "source_duration": 20.4, "source_width": 1080, "source_height": 1920,
            "loaded_fps": 30.0, "loaded_frame_count": 609, "loaded_duration": 20.3, "loaded_width": 720,
            "loaded_height": 1280}
    node = video.BCVGetVideoInfo
    assert node.RETURN_NAMES == tuple(info)
    assert node.RETURN_TYPES == ("STRING", "STRING", "STRING", "FLOAT", "INT", "FLOAT", "INT", "INT", "FLOAT", "INT",
                                 "FLOAT", "INT", "INT")
    assert node().get(info) == tuple(info.values())


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
