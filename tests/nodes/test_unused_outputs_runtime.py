"""Unused heavy outputs through ComfyUI itself: the pack's on_prompt handler registered on ComfyUI's
PromptServer by the root __init__, then ComfyUI's validate_prompt and PromptExecutor, as POST
/prompt runs a prompt. The ComfyUI side runs in a process of its own (tests/unused_outputs_runtime.py:
under `python -m pytest` the pack's nodes/ hides ComfyUI's nodes.py); these tests read its facts.

- P1-P5 under the classic, LRU and RAM-pressure caches: pose_video_mask unlinked (empty in the
  cache), linked later (the node runs again, the consumer gets it full), the same prompt again (a
  cache hit, nothing runs), unlinked again (empty), and again (nothing runs);
- no stamp (the handler did not run): full;
- a stale stamp in a resubmitted prompt: overwritten;
- a link the stamp missed: full, with a warning;
- a pack loaded after this one registers its handler after ours: once the server starts, ours
  runs after it and the saving stays on;
- another pack's handler added after ours once the server runs: off for that prompt (full), a
  console line, nothing sent to the browser;
- a stamping handler after ours (ComfyUI-BCNodes'): still on.

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_unused_outputs_runtime.py
"""
import importlib.util
import json
import os
import subprocess
import sys

import pytest

HARNESS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unused_outputs_runtime.py")
FULL, EMPTY = [4, 16, 8, 3], [0, 16, 8, 3]
OFF = "RAM saving of unused outputs is off for this run: unused_outputs_runtime changes the prompt after it."


@pytest.fixture(scope="module")
def facts(tmp_path_factory):
    for module in ("torch", "execution", "server"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not importable: put the ComfyUI root on PYTHONPATH")
    done = subprocess.run([sys.executable, HARNESS], cwd=tmp_path_factory.mktemp("comfy"), capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 0, done.stderr[-4000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_root_init_registers_the_stamp_on_comfyuis_server(facts):
    assert facts["handlers_loaded"] == ["LinkStamp"]


def test_a_handler_registered_after_ours_runs_before_it_once_the_server_starts(facts):
    assert facts["handlers_later_pack"] == ["LinkStamp", "later_pack"]
    assert facts["handlers_started"] == ["later_pack", "LinkStamp"]
    on = facts["after_startup"]
    assert on["runs"] == [{"stamp": "", "runs": 1, "seen": None, "cached": EMPTY}]
    assert on["sent"] == [] and not any("off for this run" in line for line in on["lines"])


@pytest.mark.parametrize("kind", ["CLASSIC", "LRU", "RAM_PRESSURE"])
def test_linking_the_output_later_runs_the_node_again(facts, kind):
    p1, p2, p3, p4, p5 = facts[f"p1_p5_{kind}"]["runs"]
    assert p1 == {"stamp": "", "runs": 1, "seen": None, "cached": EMPTY}
    assert p2 == {"stamp": "pose_video_mask", "runs": 1, "seen": FULL, "cached": FULL}
    # a cache hit: nothing runs, and the history holds the output node's cached shape
    assert p3 == {"stamp": "pose_video_mask", "runs": 0, "seen": FULL, "cached": FULL}
    # unlinked again: empty, run again or the P1 entry still in the LRU cache
    assert p4["cached"] == EMPTY and p4["runs"] == (0 if kind == "LRU" else 1)
    assert p5["runs"] == 0 and p5["cached"] == EMPTY


def test_without_the_handler_every_output_is_full(facts):
    assert facts["no_stamp"]["runs"] == [{"stamp": None, "runs": 1, "seen": None, "cached": FULL}]


def test_a_stale_stamp_is_overwritten(facts):
    assert facts["stale_stamp"]["runs"] == [{"stamp": "", "runs": 1, "seen": None, "cached": EMPTY}]


def test_a_link_the_stamp_missed_gets_the_full_output_and_a_warning(facts):
    missed = facts["missed_link"]
    assert missed["runs"] == [{"stamp": "", "runs": 1, "seen": FULL, "cached": FULL}]
    assert any("the prompt links pose_video_mask, which its link stamp does not list" in line for line in missed["lines"])


def test_another_packs_handler_after_ours_turns_it_off_for_the_run(facts):
    off = facts["another_pack_after"]
    assert off["runs"] == [{"stamp": None, "runs": 1, "seen": None, "cached": FULL}]
    assert any(OFF in line for line in off["lines"]) and off["sent"] == []


def test_a_stamping_handler_after_ours_keeps_it_on(facts):
    on = facts["bcnodes_after"]
    assert on["runs"] == [{"stamp": "", "runs": 1, "seen": None, "cached": EMPTY}] and on["sent"] == []
