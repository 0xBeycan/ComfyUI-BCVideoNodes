# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""ViTPose-H's raw output, heatmaps, to keypoints in frame coordinates, [N, K, 3] rows of x, y and
a 0..1-ish confidence. Kept apart from the wrappers so the offline conversion can decode
exactly the way the nodes do without importing ComfyUI."""
import numpy as np


def decode_heatmaps(heatmaps, center, scale):
    """ViTPose: DARK-refined heatmap maxima mapped back through the crop. The maxima are
    the confidences as they are. A maximum at or below 0 puts its keypoint at heatmap cell -1,
    just off the crop's top-left, with that confidence (pose2d_utils._get_max_preds), as in Wan
    and Kijai; it is drawn only at draw threshold 0 and a confidence of exactly 0."""
    from ...libs.pose_utils.pose2d_utils import keypoints_from_heatmaps
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                        center=center,
                                        scale=scale*200,
                                        unbiased=True,
                                        use_udp=False)
    return np.concatenate([points, prob], axis=2)
