"""Load Video's plan route (nodes/video_input.py, PLAN_ROUTE): what the loader will load, from the
loader's own checks and frame selection, without loading a frame.

- the answer on a clip whose audio runs past its video: the loader's own count (the container's
  duration would give more), the frames an empty frame_count stands for, the source; its info is
  the loader's video_info without the audio (no sample goes into the JSON);
- its frame_range (the seconds slider's span): the loaded rate, the model's frame step and the
  count an empty frame_count loads, the loader's own; it comes with frame_count's errors too;
- every error is the loader's message, word for word, and the source and the empty frame_count's
  count still come with an error they do not depend on;
- the probe is read once per file version;
- the route serves only files of ComfyUI's input folder, and of the output or temp folder a
  value's annotation names ("clip.mkv [output]", what a video dragged from the queue gets),
  each inside its folder (400 otherwise);
- without a ComfyUI server there is no route.

The handler is called with aiohttp's mocked request; ComfyUI's input folder is a test's tmp dir."""
import asyncio
import json
import logging
import os
import sys
import types
from urllib.parse import urlencode

import pytest

torch = pytest.importorskip("torch")
folder_paths = pytest.importorskip("folder_paths")
av = pytest.importorskip("av")
test_utils = pytest.importorskip("aiohttp.test_utils")
web = pytest.importorskip("aiohttp.web")

from video_input_fakes import grey_clip, ramp_audio, video  # noqa: E402

WIDGETS = {"model": "Wan", "resolution": "480p", "orientation": "auto", "force_fps": "", "start_frame": 1,
           "frame_count": "", "precision": "fp32"}


@pytest.fixture
def folders(tmp_path, monkeypatch):
    """ComfyUI's input, output and temp folders moved into tmp_path; the probe cache emptied."""
    for name in ("input", "output", "temp"):
        (tmp_path / name).mkdir()
        monkeypatch.setattr(folder_paths, f"{name}_directory", str(tmp_path / name))
    video._probe_file.cache_clear()
    return tmp_path


def ask(video_name, **widgets):
    """(status, the JSON answer or the text) of GET PLAN_ROUTE for `video_name` and the widgets."""
    query = urlencode({"video": video_name, **WIDGETS, **widgets})
    response = asyncio.run(video.plan_route(test_utils.make_mocked_request("GET", f"{video.PLAN_ROUTE}?{query}")))
    return response.status, (json.loads(response.text) if response.content_type == "application/json" else response.text)


def without_audio(info):
    """The loader's video_info without its audio, what the route answers."""
    return {name: value for name, value in info.items() if name != "audio"}


def loader_error(path, **widgets):
    """The message load_video raises for these widgets."""
    values = {**WIDGETS, **widgets}
    with pytest.raises(ValueError) as error:
        video.load_video(str(path), values["model"], values["resolution"], values["orientation"], values["force_fps"],
                         values["start_frame"], values["frame_count"], values["precision"])
    return str(error.value)


# --- the answer ------------------------------------------------------------------------------------

def test_audio_longer_than_the_video_gives_the_loaders_count(folders):
    # 60 frames at 30 fps (2 s) with 2.2 s of sound: the container's duration is the audio's, so a
    # count from it would be 66 (65 after 4n+1); the loader loads 57
    path = grey_clip(folders / "input" / "clip.mkv", 60, audio=ramp_audio(2.2))
    with av.open(path) as container:
        assert container.duration / 1e6 == pytest.approx(2.2, abs=0.01)
    status, answer = ask("clip.mkv")
    assert status == 200 and answer["error"] is None
    assert answer["source"] == {"fps": 30.0, "frames": 60, "width": 64, "height": 32, "start": 0.0, "audio": True}
    assert answer["available"] == 60
    assert answer["info"]["loaded_frame_count"] == 57 and answer["info"]["loaded_fps"] == 30.0
    assert (answer["info"]["loaded_width"], answer["info"]["loaded_height"]) == (832, 480)
    _, audio, info = video.load_video(path, **WIDGETS)
    assert audio is not None and "audio" not in answer["info"]
    assert answer["info"] == without_audio(info)


@pytest.mark.parametrize("widgets, info_frames, available", [
    ({"start_frame": 5, "frame_count": "11"}, 9, 56),
    ({"force_fps": "24", "start_frame": 3}, 45, 46),  # 60 frames at 30 fps keep 48 at 24 fps
    ({"force_fps": "30"}, 57, 60),  # the video's own rate: every frame
    ({"force_fps": "60"}, 117, 119),  # above it: real frames repeated on the 60 fps grid
    ({"force_fps": "31", "start_frame": 3}, 57, 59),  # 61 frames on the 31 fps grid
    ({"model": "SCAIL", "resolution": "704p", "orientation": "portrait", "frame_count": "21"}, 21, 60),
    # model None: no 4n+1; resolution source: the video's own size, or its crop to the other orientation
    ({"model": "None", "resolution": "1080p"}, 60, 60),
    ({"model": "None", "resolution": "source", "orientation": "portrait", "start_frame": 3, "frame_count": "10"}, 10, 58),
    ({"resolution": "source"}, 57, 60),
    ({"resolution": "source", "orientation": "portrait", "frame_count": "10"}, 9, 60),  # 16x32, on Wan's grid
    ({"model": "SCAIL", "resolution": "source", "start_frame": 50}, 9, 11),
    ({"precision": "fp16"}, 57, 60),  # the same frames, stored in float16
])
def test_the_answer_is_the_loaders_video_info(folders, widgets, info_frames, available):
    path = grey_clip(folders / "input" / "clip.mkv", 60)
    status, answer = ask("clip.mkv", **widgets)
    values = {**WIDGETS, **widgets}
    _, _, info = video.load_video(path, values["model"], values["resolution"], values["orientation"],
                                  values["force_fps"], values["start_frame"], values["frame_count"], values["precision"])
    assert status == 200 and answer["error"] is None and answer["info"] == without_audio(info)
    assert answer["info"]["loaded_frame_count"] == info_frames and answer["available"] == available
    assert answer["source"]["audio"] is False


@pytest.mark.parametrize("widgets, has_source, available", [
    ({"force_fps": "fast"}, True, None),
    ({"force_fps": "0"}, True, None),
    ({"frame_count": "0"}, True, 30),
    ({"frame_count": "2.5"}, True, 30),
    ({"start_frame": 31}, True, None),
    ({"start_frame": 25, "frame_count": "10"}, True, 6),
    ({"force_fps": "24", "start_frame": 25}, True, None),
    ({"start_frame": 0}, True, None),
    ({"resolution": "704p"}, True, 30),
    ({"resolution": "1080p"}, True, 30),
    ({"model": "None", "resolution": "512p"}, True, 30),
    ({"model": "SCAIL", "resolution": "source", "orientation": "portrait"}, True, 30),  # 16x32: under the grid
    ({"orientation": "sideways"}, True, 30),
    ({"precision": "fp8"}, True, 30),
])
def test_errors_are_the_loaders_messages(folders, widgets, has_source, available):
    path = grey_clip(folders / "input" / "clip.mkv", 30)
    status, answer = ask("clip.mkv", **widgets)
    assert status == 200 and answer["info"] is None
    assert answer["error"] == loader_error(path, **widgets)
    assert (answer["source"] is not None) == has_source and answer["available"] == available


@pytest.mark.parametrize("widgets, frame_range", [
    ({}, {"fps": 30.0, "step": 4, "maximum": 57}),  # 60 frames: 57 on 4n+1
    ({"frame_count": "11"}, {"fps": 30.0, "step": 4, "maximum": 57}),  # frame_count does not change it
    ({"start_frame": 5}, {"fps": 30.0, "step": 4, "maximum": 53}),  # 56 left
    ({"force_fps": "24", "start_frame": 3}, {"fps": 24.0, "step": 4, "maximum": 45}),  # 46 left at 24 fps
    ({"model": "SCAIL", "resolution": "704p"}, {"fps": 30.0, "step": 4, "maximum": 57}),
    ({"model": "None", "resolution": "720p", "start_frame": 3}, {"fps": 30.0, "step": 1, "maximum": 58}),
    # with frame_count's own errors and a resolution of another model: the range still comes
    ({"frame_count": "999"}, {"fps": 30.0, "step": 4, "maximum": 57}),
    ({"frame_count": "0"}, {"fps": 30.0, "step": 4, "maximum": 57}),
    ({"resolution": "704p"}, {"fps": 30.0, "step": 4, "maximum": 57}),
    # what it depends on is wrong: none
    ({"start_frame": 61}, None),
    ({"force_fps": "fast"}, None),
    ({"model": "Other"}, None),
])
def test_the_frame_range_is_what_an_empty_frame_count_loads(folders, widgets, frame_range):
    path = grey_clip(folders / "input" / "clip.mkv", 60)
    status, answer = ask("clip.mkv", **widgets)
    assert status == 200 and answer["frame_range"] == frame_range
    if frame_range is not None and "resolution" not in widgets:
        values = {**WIDGETS, **widgets, "frame_count": ""}
        _, _, info = video.load_video(path, values["model"], values["resolution"], values["orientation"],
                                      values["force_fps"], values["start_frame"], "", values["precision"])
        assert (info["loaded_frame_count"], info["loaded_fps"]) == (frame_range["maximum"], frame_range["fps"])


def test_a_file_the_loader_cannot_read(folders):
    (folders / "input" / "notes.mp4").write_text("not a video")
    status, answer = ask("notes.mp4")
    assert status == 200 and answer["source"] is None and answer["info"] is None and answer["available"] is None
    assert answer["frame_range"] is None
    assert answer["error"] == loader_error(folders / "input" / "notes.mp4")


@pytest.mark.parametrize("name", ["gone.mp4", "gone.mp4 [output]", "gone.mp4 [temp]"])
def test_a_missing_file_gets_the_validation_message(folders, name):
    status, answer = ask(name)
    assert status == 200
    assert answer == {"source": None, "info": None, "available": None, "frame_range": None,
                      "error": f"Invalid video file: {name}"}
    assert answer["error"] == video.BCVLoadVideo.VALIDATE_INPUTS(video=name)


def test_the_probe_is_read_once_per_file_version(folders, monkeypatch):
    path = grey_clip(folders / "input" / "clip.mkv", 9)
    probe, calls = video.probe, []
    monkeypatch.setattr(video, "probe", lambda p: calls.append(p) or probe(p))
    ask("clip.mkv")
    ask("clip.mkv", start_frame=3)
    ask("clip.mkv", force_fps="15")
    assert calls == [path]
    grey_clip(folders / "input" / "clip.mkv", 13)
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10 ** 9))
    _, answer = ask("clip.mkv")
    assert calls == [path, path] and answer["source"]["frames"] == 13


# --- the input, output and temp folders only ------------------------------------------------------

@pytest.mark.parametrize("name, frames", [
    ("clip.mkv", 5), ("sub/clip.mkv", 5), ("clip.mkv [input]", 5),
    ("clip.mkv [output]", 9), ("sub/clip.mkv [output]", 9), ("clip.mkv [temp]", 13), ("sub/clip.mkv [temp]", 13),
])
def test_files_of_the_three_folders_and_their_subfolders(folders, name, frames):
    # a clip of another length in each folder: the answer comes from the folder the value names
    for folder, count in (("input", 5), ("output", 9), ("temp", 13)):
        (folders / folder / "sub").mkdir()
        grey_clip(folders / folder / "clip.mkv", count)
        grey_clip(folders / folder / "sub" / "clip.mkv", count)
    status, answer = ask(name)
    assert status == 200 and answer["error"] is None and answer["info"]["loaded_frame_count"] == frames
    _, _, info = video.BCVLoadVideo().load(name, **WIDGETS)
    assert answer["info"] == without_audio(info)


@pytest.mark.parametrize("name", [
    "../outside.mkv", "sub/../../outside.mkv", "{root}/outside.mkv", "/etc/hosts", "link.mkv", "",
    "../outside.mkv [output]", "../input/clip.mkv [output]", "sub/../../outside.mkv [temp]",
    "{root}/outside.mkv [output]", "/etc/hosts [temp]", "link.mkv [output]", "link.mkv [temp]", " [output]",
])
def test_anything_outside_the_three_folders_is_refused(folders, name):
    grey_clip(folders / "outside.mkv", 5)
    for folder in ("input", "output", "temp"):
        grey_clip(folders / folder / "clip.mkv", 5)
        os.symlink(folders / "outside.mkv", folders / folder / "link.mkv")
    status, text = ask(name.format(root=folders))
    assert status == 400 and text == "video must be a file of ComfyUI's input, output or temp folder"


@pytest.mark.parametrize("query, text", [
    ({"video": "clip.mkv", "model": "Wan"}, "missing resolution, orientation, force_fps, start_frame, frame_count, precision"),
    ({**WIDGETS, "video": "clip.mkv", "start_frame": "two"}, "start_frame is not a whole number"),
])
def test_a_malformed_query_is_refused(folders, query, text):
    request = test_utils.make_mocked_request("GET", f"{video.PLAN_ROUTE}?{urlencode(query)}")
    response = asyncio.run(video.plan_route(request))
    assert response.status == 400 and response.text == text


# --- registration ----------------------------------------------------------------------------------

def test_the_route_is_registered_only_on_a_server(folders, monkeypatch, caplog):
    monkeypatch.delitem(sys.modules, "server", raising=False)
    caplog.set_level(logging.INFO, logger="BCVideoNodes")
    assert video.register_plan_route() is None
    assert "no ComfyUI server: Load Video's preview gets no frame count" in caplog.text
    routes = web.RouteTableDef()
    server = types.SimpleNamespace(routes=routes)
    monkeypatch.setitem(sys.modules, "server", types.SimpleNamespace(PromptServer=types.SimpleNamespace(instance=server)))
    assert video.register_plan_route() is video.plan_route
    assert [(route.method, route.path, route.handler) for route in routes] == [("GET", video.PLAN_ROUTE, video.plan_route)]
    assert video.PLAN_ROUTE == "/bcvideonodes/load_video/plan"
    # every widget the loader reads; not the seconds slider, which it ignores
    widgets = video.BCVLoadVideo.INPUT_TYPES()
    assert video.PLAN_PARAMS == (*widgets["required"], *(name for name in widgets["optional"] if name != video.SECONDS))
    assert list(widgets["optional"])[-1] == video.SECONDS == "seconds"
