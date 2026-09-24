"""The AAPose body layout: each index name the pipelines address a keypoint by is that keypoint's
position in BODY_NAMES. The names and the index constants live side by side in libs/keypoints.py
as literals; this ties the two. Standard library only.
"""
import pytest

from bcvideonodes.libs.keypoints import (BODY_NAMES, L_ANKLE, L_FOOT, L_HIP, L_SHOULDER, R_ANKLE, R_FOOT, R_HIP,
                                         R_SHOULDER)


@pytest.mark.parametrize("index, name", [
    (R_SHOULDER, "r_shoulder"),
    (L_SHOULDER, "l_shoulder"),
    (R_HIP, "r_hip"),
    (L_HIP, "l_hip"),
    (R_ANKLE, "r_ankle"),
    (L_ANKLE, "l_ankle"),
    (R_FOOT, "r_foot"),
    (L_FOOT, "l_foot"),
], ids=["R_SHOULDER", "L_SHOULDER", "R_HIP", "L_HIP", "R_ANKLE", "L_ANKLE", "R_FOOT", "L_FOOT"])
def test_index_name_is_the_position_of_its_keypoint(index, name):
    assert BODY_NAMES[index] == name
