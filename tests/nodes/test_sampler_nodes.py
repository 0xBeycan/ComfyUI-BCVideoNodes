"""The long-video sampler nodes' surface: the widgets all three share, and each Wan Animate node's own
inputs and defaults (SCAIL-2's are in test_nodes_scail2.py), in the order saved workflows store them. ComfyUI itself is stubbed; skipped when
torch is not installed.
"""

import pytest

torch = pytest.importorskip("torch")

from sampler_fakes import ANIMATE1, ANIMATE2, SCAIL2, node_module  # noqa: E402,F401


# widget order the released 0.1.0 node had: saved workflows store widget
# values positionally, so this must never change
ANIMATE2_REQUIRED_ORDER = [
    "model", "positive", "negative", "vae", "reference_image", "pose_video",
    "width", "height", "frames_per_chunk", "total_frames",
    "shift", "sampler_name", "scheduler", "steps", "denoise", "cfg", "seed", "seed_mode",
    "reference_image_strength", "pose_strength", "pose_start_percent", "pose_end_percent",
]


# --- shared behaviour, all three nodes ---

@pytest.mark.parametrize("node", [ANIMATE1, ANIMATE2, SCAIL2])
def test_input_types_shared_widgets(node_module, node):
    spec = getattr(node_module, node).INPUT_TYPES()
    required = list(spec["required"])
    assert required[:len(ANIMATE2_REQUIRED_ORDER) - 4] == ANIMATE2_REQUIRED_ORDER[:-4]
    assert spec["required"]["seed"][1]["control_after_generate"] is True
    assert list(spec["optional"])[-2:] == ["sigmas_override", "color_anchor_strength"]
    # the last widget, off by default: saved workflows store widget values by position
    assert spec["optional"]["color_anchor_strength"] == ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                                                   "tooltip": spec["optional"]["color_anchor_strength"][1]["tooltip"]})
    # ComfyUI's samplers, then wan_dpmpp, as ComfyUI's schedulers are followed by wan_beta
    assert spec["required"]["sampler_name"][0] == ["euler", "lcm", "wan_dpmpp"]
    assert spec["required"]["scheduler"][0] == ["normal", "simple", "beta", "wan_beta"]
    # every widget says what it does
    for section in ("required", "optional"):
        for name, (kind, *options) in spec[section].items():
            if isinstance(kind, list) or kind in ("INT", "FLOAT", "BOOLEAN"):
                assert options and options[0].get("tooltip"), name
    assert spec["required"]["frames_per_chunk"][1]["default"] == 81
    assert spec["required"]["total_frames"][1]["default"] == 81
    assert spec["required"]["total_frames"][1]["min"] == 0  # 0 still means "pose video length"
    assert node_module._combo_default(["a", "b"], "lcm") == "a"


# --- Wan Animate 2 node ---

def test_animate2_input_types(node_module):
    spec = node_module.BCVWanAnimate2LongVideoSampler.INPUT_TYPES()
    assert list(spec["required"]) == ANIMATE2_REQUIRED_ORDER + ["attn_log_scale", "last_chunk", "tail_padding"]
    assert list(spec["optional"]) == ["positive_pose", "clip_vision_output", "clip_vision_output_pose", "clip_vision", "sigmas_override",
                                      "color_anchor_strength"]
    assert spec["required"]["frames_per_chunk"][1]["default"] == 81
    assert spec["required"]["shift"][1]["default"] == 5.0
    assert spec["required"]["sampler_name"][1]["default"] == "euler"
    # 10 steps as the official distilled config; wan_beta because it won the user's A/B against beta and simple is one click away
    assert spec["required"]["scheduler"][1]["default"] == "wan_beta"
    assert spec["required"]["scheduler"][0][-1] == "wan_beta"
    assert spec["required"]["steps"][1]["default"] == 10
    assert spec["required"]["attn_log_scale"][1]["default"] == -1.3
    assert spec["required"]["last_chunk"][0] == ["fit", "full", "min29"]
    assert spec["required"]["last_chunk"][1]["default"] == "fit"
    assert spec["required"]["tail_padding"][0] == ["last_frame", "ping_pong"]
    assert spec["required"]["tail_padding"][1]["default"] == "last_frame"


# --- Wan Animate (1) node ---

def test_animate1_input_types(node_module):
    spec = node_module.BCVWanAnimateLongVideoSampler.INPUT_TYPES()
    assert list(spec["required"]) == ANIMATE2_REQUIRED_ORDER[:-4] + ["continue_motion_max_frames", "last_chunk", "tail_padding"]
    assert list(spec["optional"]) == ["clip_vision_output", "face_video", "background_video", "character_mask", "sigmas_override",
                                      "color_anchor_strength"]
    # core node + official template defaults
    assert spec["required"]["frames_per_chunk"][1]["default"] == 81
    assert spec["required"]["continue_motion_max_frames"][1] == {"default": 5, "min": 1, "max": 16384, "step": 4, "tooltip": spec["required"]["continue_motion_max_frames"][1]["tooltip"]}
    assert spec["required"]["shift"][1]["default"] == 8.0
    assert spec["required"]["sampler_name"][1]["default"] == "euler"
    assert spec["required"]["scheduler"][1]["default"] == "wan_beta"
    assert spec["required"]["steps"][1]["default"] == 6
    assert spec["optional"]["character_mask"][0] == "MASK"
    assert spec["required"]["last_chunk"][0] == ["fit", "full", "min29"]
    assert spec["required"]["last_chunk"][1]["default"] == "fit"
    assert spec["required"]["tail_padding"][0] == ["last_frame", "ping_pong"]
    assert spec["required"]["tail_padding"][1]["default"] == "last_frame"
