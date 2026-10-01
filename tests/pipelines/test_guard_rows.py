"""The guard row TypedDicts of pipelines/guard/common.py against the row tuples beside them: the
same keys in the same order, so the annotations cannot drift from POSE_ROW, MASK_ROW,
PREPROCESS_ROW, SCAIL2_ROW, SCAIL2_REFERENCE and MASK_REFERENCE (tests/pipelines/test_guard_split.py and
test_guard_reference.py tie the tuples to the metrics JSON):

    python -m pytest tests/pipelines/test_guard_rows.py
"""
import pytest

from bcvideonodes.pipelines.guard import common


@pytest.mark.parametrize("row, keys", [("PoseRow", "POSE_ROW"), ("MaskRow", "MASK_ROW"),
                                       ("PreprocessRow", "PREPROCESS_ROW"), ("Scail2Row", "SCAIL2_ROW"),
                                       ("Scail2Reference", "SCAIL2_REFERENCE"), ("MaskReference", "MASK_REFERENCE")])
def test_the_row_typeddict_has_the_keys_of_its_tuple_in_order(row, keys):
    assert tuple(getattr(common, row).__annotations__) == getattr(common, keys)
