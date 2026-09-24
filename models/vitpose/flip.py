"""ViTPose's test-time flip: the COCO-WholeBody mirror index and the flip-back of the mirrored
crop's heatmaps (ViTPose.flip_keypoints runs it)."""
import numpy as np


# COCO-WholeBody's mirror: FLIP_INDEX[k] is the keypoint that k is in the mirrored image (left
# and right swapped; nose, the face centre line and the chin stay). Derived from the `swap`
# names of mmpose's configs/_base_/datasets/coco_wholebody.py (open-mmlab/mmpose @ 2a0a2d2),
# the dataset meta ViTPose's wholebody configs read: body 0-16, feet 17-22 (left big toe,
# small toe, heel <-> right), face 23-90 (68 points, mirrored across the centre line), left
# hand 91-111 <-> right hand 112-132.
FLIP_INDEX = (
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 20, 21, 22, 17, 18, 19, 39, 38, 37, 36,
    35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 50, 51,
    52, 53, 58, 57, 56, 55, 54, 68, 67, 66, 65, 70, 69, 62, 61, 60, 59, 64, 63, 77, 76, 75, 74, 73, 72,
    71, 82, 81, 80, 79, 78, 87, 86, 85, 84, 83, 90, 89, 88, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111)


def flip_back(heatmaps):
    """The heatmaps [N, 133, h, w] of a mirrored crop, back in the crop's own frame: each
    keypoint takes its mirror partner's map, flipped left-right, then shifted one column right.

    This is ViTPose's own test-time flip (ViTAE-Transformer/ViTPose @ c050ed2,
    mmpose/core/post_processing/post_transforms.py flip_back and
    topdown_heatmap_simple_head.py inference_model), with shift_heatmap=True as its
    ViTPose_huge_wholebody_256x192 config sets it. The shift belongs to the non-UDP decode
    that config and decode_heatmaps use: mirroring a heatmap maps column x to w - 1 - x while
    the image mirror maps it to about w - x (less a quarter column at stride 4), so the
    flipped-back map sits about one column to the left."""
    out = np.ascontiguousarray(heatmaps[:, list(FLIP_INDEX), :, ::-1])
    out[..., 1:] = out[..., :-1].copy()
    return out
