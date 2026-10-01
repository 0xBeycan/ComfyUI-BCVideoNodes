"""Sapiens2 pose (Meta, facebookresearch/sapiens2): registers architecture "sapiens2_pose" and a pose
estimator per model file, "Sapiens2 <size> <precision>", the Sapiens2 names of the `pose_model`
widget of Pose Detection and the preprocess wrappers (nodes/pose.py). The files are downloaded from
their own repository, not the pack's: bf16 made by scripts/convert_sapiens2.py from
the transformers checkpoints, int8_convrot made from the bf16 file by convert_to_quant.

Sapiens2 gives the body, the feet and the hands; the pose pipeline takes the face from ViTPose-H
(pipelines/sapiens2_pose.py)."""
from ..common import registry
from .net import Sapiens2PoseNet
from .wrapper import ARCHITECTURE, Sapiens2Pose

REPO = "beycanai/sapiens2-convrot"
# (size, precision) in the order the widgets list them: the largest model first, int8 ConvRot before bf16
MODELS = [(size, precision) for size in ("5b", "1b", "0.8b", "0.4b") for precision in ("int8_convrot", "bf16")]

registry.register("architecture", ARCHITECTURE, Sapiens2PoseNet)
for size, precision in MODELS:
    registry.register("pose_estimator", f"Sapiens2 {size} {precision.replace('_', ' ')}", Sapiens2Pose,
                      file=f"sapiens2_pose_{size}_{precision}.safetensors", repo=REPO)
