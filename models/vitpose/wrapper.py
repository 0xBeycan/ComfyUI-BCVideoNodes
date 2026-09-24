# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
"""ViTPose-H as the pose pipeline calls it: the heatmap keypoints of a crop."""
from ..common.wrapper import NativeModel
from .decode import decode_heatmaps


class ViTPose(NativeModel):
    architecture = "vitpose"

    def forward(self, img, center, scale, **kwargs):
        return decode_heatmaps(self.run(img), center, scale)
