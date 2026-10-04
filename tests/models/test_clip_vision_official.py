"""clip_vision_encode_official on core's own CLIP vision model (a tiny random-weight config of the
ViT-H architecture): its outputs are the ones core's CLIPVisionEncode gives where the two
preprocessings agree (an image already at CLIP's size, at 8-bit levels), and its
penultimate_hidden_states, which core's Wan models read as clip_fea, are official's use_31_block
(wan/modules/clip.py VisionTransformer.forward: every transformer block but the last), written
out on core's modules. Runs where ComfyUI is importable (with the ComfyUI root on PYTHONPATH).
"""

import pytest

torch = pytest.importorskip("torch")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
clip_vision_module = pytest.importorskip("comfy.clip_vision")

from comfy.ldm.modules.attention import optimized_attention_for_device  # noqa: E402

from scail2_fakes import scail2  # noqa: E402

SIZE = 28
CONFIG = {"hidden_size": 32, "image_size": SIZE, "intermediate_size": 64, "num_attention_heads": 2, "num_hidden_layers": 4,
          "patch_size": 14, "projection_dim": 16, "hidden_act": "gelu", "layer_norm_eps": 1e-5,
          "model_type": "clip_vision_model", "num_channels": 3}


@pytest.fixture(scope="module")
def clip_vision():
    model = clip_vision_module.ClipVisionModel(CONFIG)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.model.parameters():
            parameter.copy_((torch.randn(parameter.shape, generator=generator) * 0.1).to(parameter.dtype))
    return model


def image_8bit(seed=1):
    return torch.randint(0, 256, (1, SIZE, SIZE, 3), generator=torch.Generator().manual_seed(seed)) / 255.0


def test_the_outputs_are_core_s_clip_vision_encode_outputs(clip_vision):
    image = image_8bit()
    core = clip_vision.encode_image(image, crop=False)
    ours = scail2.clip_vision_encode_official(clip_vision, image)
    for key in ("last_hidden_state", "image_embeds", "penultimate_hidden_states"):
        assert torch.equal(ours[key], core[key]), key
    assert ours.image_sizes == core.image_sizes and ours.mm_projected is None and core.mm_projected is None


def test_penultimate_hidden_states_are_official_use_31_block(clip_vision):
    image = image_8bit(2)
    mean = torch.tensor(clip_vision.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(clip_vision.image_std).view(1, 3, 1, 1)
    pixels = ((image.movedim(-1, 1) - mean) / std).to(clip_vision.load_device)
    vision = clip_vision.model.vision_model
    with torch.no_grad():
        x = vision.pre_layrnorm(vision.embeddings(pixels))
        attention = optimized_attention_for_device(x.device, mask=False, small_input=True)
        for block in vision.encoder.layers[:-1]:
            x = block(x, None, attention)
    ours = scail2.clip_vision_encode_official(clip_vision, image)
    assert torch.equal(ours.penultimate_hidden_states, x.to(ours.penultimate_hidden_states.device))
