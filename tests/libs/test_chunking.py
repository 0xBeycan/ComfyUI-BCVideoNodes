import pytest

from bcvideonodes.libs.chunking import (
    LAST_CHUNK,
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

    from bcvideonodes.libs import chunking as chunk_planner

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


@pytest.mark.parametrize("policy, plan", [("fit", [81, 81, 81, 13]), ("full", [81, 81, 81, 81])])
def test_last_chunk_policies_240_at_81_overlap_5(policy, plan):
    # the README example: fit -> 241 produced, full -> 309 produced, both cut to 240
    chunk_length, _ = LAST_CHUNK[policy]
    assert plan_chunks(240, 81, 5, chunk_length) == plan
    assert produced_frames(plan, 5) == {"fit": 241, "full": 309}[policy]


def test_last_chunk_values_are_the_widget_options():
    assert list(LAST_CHUNK) == ["fit", "full", "min29"]


def test_min29_888_at_81_overlap_1():
    # fit ends in a 9-frame chunk; min29 runs it at 29, as the official Wan-Animate-2 pads its last clip
    fit, _ = LAST_CHUNK["fit"]
    min29, _ = LAST_CHUNK["min29"]
    assert plan_chunks(888, 81, 1, fit) == [81] * 11 + [9]  # 81 + 80 * 10 = 881, then 7 more
    assert plan_chunks(888, 81, 1, min29) == [81] * 11 + [29]
    assert produced_frames([81] * 11 + [29], 1) == 909


def test_min29_is_fit_when_the_last_chunk_is_long_enough():
    fit, _ = LAST_CHUNK["fit"]
    min29, _ = LAST_CHUNK["min29"]
    assert plan_chunks(1110, 81, 1, min29) == plan_chunks(1110, 81, 1, fit) == [81] * 13 + [73]  # 1041, then 69 more


def test_min29_single_chunk_shorter_than_29():
    min29, _ = LAST_CHUNK["min29"]
    assert plan_chunks(10, 81, 1, min29) == [29]
    assert plan_chunks(3, 81, 5, min29) == [29]


def test_min29_is_capped_at_frames_per_chunk():
    fit, _ = LAST_CHUNK["fit"]
    min29, _ = LAST_CHUNK["min29"]
    # 100 frames at 17, overlap 1: 17 + 16 * 5 = 97, then 3 more: fit runs 5 frames, min29 the full 17
    assert plan_chunks(100, 17, 1, fit) == [17] * 6 + [5]
    assert plan_chunks(100, 17, 1, min29) == [17] * 7
    assert plan_chunks(10, 17, 1, min29) == [17]


@pytest.mark.parametrize("overlap", OVERLAPS)
@pytest.mark.parametrize("chunk", CHUNKS)
@pytest.mark.parametrize("total", TOTALS)
def test_min29_plan_covers_total(total, chunk, overlap):
    min29, _ = LAST_CHUNK["min29"]
    plan = plan_chunks(total, chunk, overlap, min29)
    assert all((length - 1) % 4 == 0 and length <= chunk for length in plan)
    assert all(length == snap_down(chunk) for length in plan[:-1])
    assert plan[-1] >= min(29, snap_down(chunk))
    assert produced_frames(plan, overlap) >= total
