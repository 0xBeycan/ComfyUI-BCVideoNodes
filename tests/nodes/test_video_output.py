"""The Save Video and Video Comparer nodes: their inputs, the file names (prefix, subfolder,
counter), output vs temp, the workflow in the file, the ui payload the player reads; the
comparer's clip as the BCNodes node wrote it (A | B in one H.264 file, cut to the shorter clip,
the smaller side letterboxed, AAC cut to the clip). ComfyUI's folders are pointed at tmp_path.

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_video_output.py
"""
import json

import pytest

torch = pytest.importorskip("torch")
av = pytest.importorskip("av")
folder_paths = pytest.importorskip("folder_paths")
cli_args = pytest.importorskip("comfy.cli_args")

from names import spec  # noqa: E402
from video_output_fakes import clip, decode, tone, video  # noqa: E402

REQUIRED = ["images", "fps", "filename_prefix", "codec", "crf", "preset", "pix_fmt", "save_output", "save_metadata"]


@pytest.fixture
def folders(tmp_path, monkeypatch):
    output, temp = tmp_path / "output", tmp_path / "temp"
    output.mkdir()
    temp.mkdir()
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(output))
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(temp))
    monkeypatch.setattr(cli_args.args, "disable_metadata", False)
    return output, temp


def save(**values):
    defaults = {name: options[1]["default"] for name, options in spec("BCVSaveVideo")["required"].items() if name != "images"}
    return video.BCVSaveVideo().save(**{"images": clip(), **defaults, **values})


def test_save_video_inputs():
    types = spec("BCVSaveVideo")
    assert list(types["required"]) == REQUIRED
    assert list(types["optional"]) == ["audio"] and types["optional"]["audio"][0] == "AUDIO"
    assert types["hidden"] == {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
    codec, options = types["required"]["codec"]
    assert codec == list(video.CODECS) and options["default"] == "h264-mp4"
    assert options["bcv_codecs"] == {name: video.widget_values(name) for name in video.CODECS}
    crf = types["required"]["crf"][1]
    assert (crf["default"], crf["min"], crf["max"]) == (19, 0, 63)
    assert types["required"]["preset"][0] == video.union("preset") and types["required"]["preset"][1]["default"] == "medium"
    assert types["required"]["pix_fmt"][0] == video.union("pix_fmt") and types["required"]["pix_fmt"][1]["default"] == "yuv420p"
    cls = video.BCVSaveVideo
    assert cls.RETURN_TYPES == () and cls.OUTPUT_NODE and cls.CATEGORY == video.VIDEO == "BCVideoNodes/Video"


def test_names_counter_and_folders(folders):
    output, temp = folders
    first = save(filename_prefix="clips/run")["ui"][video.UI_KEY][0]
    second = save(filename_prefix="clips/run")["ui"][video.UI_KEY][0]
    assert (first["filename"], first["subfolder"], first["type"]) == ("run_00001_.mp4", "clips", "output")
    assert second["filename"] == "run_00002_.mp4"
    assert (output / "clips" / "run_00001_.mp4").is_file() and (output / "clips" / "run_00002_.mp4").is_file()
    assert first["frames"] == 9 and first["fps"] == 24.0 and first["audio"] is False and first["format"] == "video/mp4"

    entry = save(filename_prefix="run", save_output=False, codec="vp9-webm", crf=20, preset="1")["ui"][video.UI_KEY][0]
    assert (entry["filename"], entry["type"]) == ("run_00001_.webm", "temp")
    assert (temp / "run_00001_.webm").is_file()


def test_audio_is_muxed_and_none_is_accepted(folders):
    output, _ = folders
    entry = save(filename_prefix="a", fps=30.0, audio=tone(3.0))["ui"][video.UI_KEY][0]
    got = decode(str(output / entry["filename"]))
    assert entry["audio"] is True and got["audio_codec"] == "aac"
    assert got["audio_samples"] <= video.audio_samples(9, 30.0, 44100) + 1024
    for audio in (None, {"waveform": torch.zeros(1, 2, 0), "sample_rate": 44100}):
        entry = save(filename_prefix="b", audio=audio)["ui"][video.UI_KEY][0]
        assert entry["audio"] is False and "audio_codec" not in decode(str(output / entry["filename"]))


def test_the_workflow_in_the_file(folders, monkeypatch):
    output, _ = folders
    workflow = {"nodes": [{"id": 7, "type": "BCVSaveVideo"}]}
    prompt = {"7": {"class_type": "BCVSaveVideo", "inputs": {}}}

    def tags(**values):
        entry = save(filename_prefix="m", prompt=prompt, extra_pnginfo={"workflow": workflow}, **values)["ui"][video.UI_KEY][0]
        return {k.lower(): v for k, v in decode(str(output / entry["filename"]))["metadata"].items()}

    written = tags()
    assert json.loads(written["workflow"]) == workflow and json.loads(written["prompt"]) == prompt
    assert not {"workflow", "prompt"} & set(tags(save_metadata=False))
    monkeypatch.setattr(cli_args.args, "disable_metadata", True)
    assert not {"workflow", "prompt"} & set(tags())


def test_a_value_the_codec_does_not_take(folders):
    output, _ = folders
    with pytest.raises(ValueError, match="preset 'medium' is not one of av1-webm's"):
        save(codec="av1-webm", crf=23, preset="medium")
    assert not any(output.iterdir())


# ---- Video Comparer ----------------------------------------------------------------------------

def compare(**inputs):
    return video.BCVVideoComparer().compare(**{"fps": 12.0, **inputs})["ui"][video.UI_KEY]


def test_comparer_inputs():
    types = spec("BCVVideoComparer")
    assert list(types["required"]) == ["fps"]
    fps = types["required"]["fps"]
    assert fps[0] == "FLOAT" and {k: fps[1][k] for k in ("default", "min", "max", "step")} == {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01}
    assert {name: t[0] for name, t in types["optional"].items()} == {"video_a": "IMAGE", "video_b": "IMAGE", "audio": "AUDIO"}
    cls = video.BCVVideoComparer
    assert cls.RETURN_TYPES == () and cls.OUTPUT_NODE and cls.CATEGORY == "BCVideoNodes/Video"


def test_comparer_writes_a_then_b_into_one_file(folders):
    _, temp = folders
    black = torch.zeros(6, 33, 65, 3)  # odd sizes are cut to even
    white = torch.ones(4, 32, 64, 3)
    (info,) = compare(video_a=black, video_b=white)
    assert info["sides"] == ["A", "B"] and info["frames"] == {"A": 6, "B": 4} and info["fps"] == 12.0 and info["audio"] is False
    assert info["type"] == "temp" and info["subfolder"] == ""
    got = decode(str(temp / info["filename"]))
    assert (got["width"], got["height"]) == (128, 32) and got["frames"].shape[0] == 4 and "audio_codec" not in got
    assert got["codec"] == "h264" and got["pix_fmt"] == "yuv420p"
    assert (got["colorspace"], got["primaries"], got["trc"], got["color_range"]) == (1, 1, 1, 1)  # Save Video's h264
    assert got["frames"][:, :, :60].max() < 0.02 and got["frames"][:, :, 68:].min() > 0.98

    assert compare(video_b=white)[0]["sides"] == ["B"]
    assert compare() == []


def test_comparer_audio_and_letterbox(folders):
    _, temp = folders
    audio = {"waveform": torch.zeros((1, 2, 48000)), "sample_rate": 16000}
    (info,) = compare(video_a=torch.zeros((12, 8, 16, 3)), video_b=torch.ones((12, 16, 32, 3)), audio=audio)
    got = decode(str(temp / info["filename"]))
    assert info["audio"] is True and (got["width"], got["height"]) == (64, 16) and got["frames"].shape[0] == 12
    assert got["audio_codec"] == "aac" and got["audio_rate"] == 16000
    assert abs(got["audio_samples"] / 16000 - 1.0) < 0.15  # 12 frames at 12 fps: cut to 1 s
