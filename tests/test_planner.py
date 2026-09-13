import pytest

from chunk_planner import (
    MIN_CHUNK,
    format_plan,
    next_chunk_length,
    overlap_for_motion_frames,
    plan_chunks,
    produced_frames,
    snap_down,
    snap_up,
)

TOTALS = [3, 5, 81, 82, 100, 161, 240, 360, 457, 800]
CHUNKS = [81, 65, 49, 33]
OVERLAPS = [1, 5]


def test_planner_imports_nothing():
    import types

    import chunk_planner

    assert not [name for name, value in vars(chunk_planner).items() if isinstance(value, types.ModuleType)]


@pytest.mark.parametrize("value,expected", [(5, 5), (8, 5), (9, 9), (81, 81), (83, 81), (84, 81), (1, 5), (0, 5)])
def test_snap_down(value, expected):
    assert snap_down(value) == expected


@pytest.mark.parametrize("value,expected", [(5, 5), (6, 9), (9, 9), (81, 81), (82, 85), (2, 5), (0, 5)])
def test_snap_up(value, expected):
    assert snap_up(value) == expected


@pytest.mark.parametrize("motion,expected", [(0, 0), (1, 1), (2, 1), (4, 1), (5, 5), (9, 9)])
def test_overlap_for_motion_frames(motion, expected):
    assert overlap_for_motion_frames(motion) == expected


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("chunk", CHUNKS)
@pytest.mark.parametrize("total", TOTALS)
def test_plan_covers_total_exactly(total, chunk, overlap):
    plan = plan_chunks(total, chunk, overlap)
    assert plan
    for length in plan:
        assert length >= MIN_CHUNK
        assert (length - 1) % 4 == 0
        assert length <= chunk
    produced = produced_frames(plan, overlap)
    assert produced >= total
    assert produced - total <= 3
    # every chunk but the last is a full one
    assert all(length == snap_down(chunk) for length in plan[:-1])


def test_spec_example_360_at_81():
    plan = plan_chunks(360, 81, 1)
    assert plan == [81, 81, 81, 81, 41]
    assert produced_frames(plan, 1) == 361


def test_single_chunk_when_total_fits():
    assert plan_chunks(81, 81, 1) == [81]
    assert plan_chunks(5, 81, 1) == [5]
    assert plan_chunks(3, 33, 1) == [5]


def test_frames_per_chunk_rounds_down():
    assert plan_chunks(200, 84, 1)[0] == 81
    assert plan_chunks(200, 83, 1)[0] == 81


def test_chunk_must_exceed_overlap():
    with pytest.raises(ValueError):
        next_chunk_length(0, 100, 5, 5)
    with pytest.raises(ValueError):
        plan_chunks(100, 7, 5)  # 7 snaps down to 5


def test_total_must_be_positive():
    with pytest.raises(ValueError):
        plan_chunks(0, 81, 1)


@pytest.mark.parametrize("total", TOTALS)
@pytest.mark.parametrize("actual_trim", [0, 1, 5])
def test_loop_self_corrects_when_node_trims_differently(total, actual_trim):
    # models the node loop: the next length comes from the REAL produced
    # count, and after a chained chunk the overlap is replaced by the
    # trim_image the node actually returned. A wrong initial assumption
    # changes the chunk count, not the final total.
    assumed = 1
    chunk = 49
    overlap = assumed
    produced = 0
    lengths = []
    while produced < total:
        length = next_chunk_length(produced, total, chunk, overlap)
        assert length > 0
        assert (length - 1) % 4 == 0
        kept = length if not lengths else length - actual_trim
        assert kept > 0
        lengths.append(length)
        produced += kept
        if len(lengths) > 1:
            overlap = actual_trim
        assert len(lengths) < 1000
    assert produced >= total
    assert produced - total <= 3 + max(0, assumed - actual_trim)


def test_next_chunk_length_returns_zero_when_done():
    assert next_chunk_length(100, 100, 81, 1) == 0
    assert next_chunk_length(103, 100, 81, 1) == 0


def test_format_plan():
    plan = plan_chunks(360, 81, 1)
    text = format_plan(plan, produced_frames(plan, 1), 360, 360, 1)
    assert text == "81 + 81 + 81 + 81 + 41 -> 361 produced -> 360 frames (pose 360, overlap 1)"
