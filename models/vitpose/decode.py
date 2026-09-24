# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""ViTPose-H's raw output, heatmaps, to keypoints in frame coordinates, [N, K, 3] rows of x, y and
a 0..1-ish confidence. Kept apart from the wrappers so the offline conversion can decode
exactly the way the nodes do without importing ComfyUI."""
import numpy as np


def decode_heatmaps(heatmaps, center, scale, conf_scale=1.0):
    """ViTPose: DARK-refined heatmap maxima mapped back through the crop. The maxima are
    the confidences as they are; a conf_scale other than 1 divides them."""
    from ...libs.pose_utils.pose2d_utils import keypoints_from_heatmaps
    points, prob = keypoints_from_heatmaps(heatmaps=heatmaps,
                                        center=center,
                                        scale=scale*200,
                                        unbiased=True,
                                        use_udp=False)
    if conf_scale != 1.0:
        prob = prob / conf_scale
    return np.concatenate([points, prob], axis=2)
