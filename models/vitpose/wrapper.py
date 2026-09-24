# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""ViTPose-H as the pose pipeline calls it: the heatmap keypoints of a crop."""
from ..common.wrapper import NativeModel
from .decode import decode_heatmaps


class ViTPose(NativeModel):
    architecture = "vitpose"
    # ViTPose's confidences are its heatmap maxima as they are: nothing divides them, and
    # PoseConfig.confidence_scale (RTMW only) is ignored for this model.
    conf_scale = None

    def forward(self, img, center, scale, **kwargs):
        return decode_heatmaps(self.run(img), center, scale)
