"""The `sapiens2` Names the Sapiens2 test bodies read pack names through (tests/names.py), and what
the Sapiens2 test files share: a tiny module config and FakeSapiens2, a stand-in for the wrapper.

Standard library and numpy only at module level.
"""
import numpy as np

from names import Names, Ref, refs, seams

sapiens2 = Names("sapiens2", {
    **refs("models.sapiens2", "REPO", "MODELS"),
    **refs("models.sapiens2.net", "Sapiens2PoseNet"),
    **refs("models.sapiens2.wrapper", "ARCHITECTURE", "CONVERTER", "Sapiens2Pose", "build", "quant_compute_dtype"),
    **refs("models.sapiens2.decode", "INPUT_SIZE", "HEATMAP_SIZE", "crop_input", "udp_crop", "udp_crop_params",
           "udp_decode", "udp_warp_matrix"),
    **refs("models.sapiens2.keypoints", "COCO_FROM_SAPIENS2", "COCO_WHOLEBODY_NAMES", "FACE", "SAPIENS2_70_NAMES",
           "to_coco133"),
    **refs("pipelines.sapiens2_pose", "BATCH_SIZE", "hybrid_keypoints", "sapiens2_pose"),
    **refs("nodes.pose", "POSE_MODELS"),
    **refs("models.common.checkpoint", "FORMAT_VERSION"),
    **seams("models.common.loader", "load_pose_estimator"),
    "registry": Ref("models.common.registry"),
    # the offline converter: the key remap and the config it writes
    "convert": Ref("scripts.convert_sapiens2"),
})

# a tiny module config: 4 layers, two of them grouped-query (2 key/value heads for 4 query heads),
# a 64x48 input, 5 keypoints
TINY = {"input_size": [64, 48], "patch_size": 16, "num_channels": 3, "hidden_size": 64, "num_layers": 4,
        "num_heads": 4, "kv_heads": [4, 2, 2, 4], "intermediate_size": 128, "num_register_tokens": 2,
        "rope_theta": 100.0, "rms_norm_eps": 1e-6, "qkv_bias": [True, True, True], "proj_bias": True,
        "mlp_bias": True, "num_keypoints": 5,
        "head": {"upsample_channels": [32, 16], "upsample_kernels": [4, 4], "conv_channels": [16], "conv_kernels": [1]}}


class FakeSapiens2:
    """The Sapiens2 wrapper's call: every body, foot and hand keypoint at its crop's centre plus a
    fixed offset per keypoint, confidence 0.7; the face rows 23-90 are 0 as keypoints.to_coco133 leaves
    them. Records the batch size of every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, crops, centers, scales):
        assert crops.shape[1:] == (3, 1024, 768) and crops.dtype == np.float32
        self.calls.append(len(crops))
        kp = np.zeros((len(crops), 133, 3), dtype=np.float32)
        kp[:, :, 0] = centers[:, :1] + np.linspace(-30, 30, 133)
        kp[:, :, 1] = centers[:, 1:] + np.linspace(-50, 50, 133)
        kp[:, :, 2] = 0.7
        kp[:, 23:91] = 0.0
        return kp
