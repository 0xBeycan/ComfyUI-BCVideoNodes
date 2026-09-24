"""The SAM 3.1 Multiplex counts TypedDicts against the code that fills them: each TypedDict's keys,
in order, are the labels its function writes into `counts` - the keys of the dict it starts
from, then every `counts["..."]` it stores, in the order they first appear - and those are the
literal lists below. A label added to the code without its key, or a key reordered, fails here.

The two modules import without ComfyUI, but this file imports comfy.cli_args first, so it runs
where ComfyUI is importable:

    python -m pytest tests/pipelines/test_sam3_1_multiplex_counts.py
"""
import ast
import inspect

import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True

from bcvideonodes.pipelines.sam3_1_multiplex import pose, prompt  # noqa: E402

COUNTS = {
    "PromptCounts": (prompt, "segment_by_prompt",
                     ("false starts", "reconditioned", "tracked backwards", "frames segmented", "tracked from frame")),
    "MultiCounts": (prompt, "segment_by_prompt_multi",
                    ("false starts", "duplicates", "reconditioned", "suppressed", "objects tracked",
                     "frames segmented")),
    "PoseCounts": (pose, "segment_by_pose",
                   ("prompted", "propagated", "re-seeded early", "kept at low recall", "no prompt", "empty prompt")),
}


def labels_written(function):
    """The labels `function` writes into its local `counts`, in the order they first appear in
    its source: the keys of the dict literal it binds, then the keys of every store."""
    found = []
    for node in ast.walk(ast.parse(inspect.getsource(function))):
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "counts" for t in targets):
                found += [(node.lineno, node.col_offset, i, key.value) for i, key in enumerate(node.value.keys)]
        elif (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
              and isinstance(node.value, ast.Name) and node.value.id == "counts"):
            assert isinstance(node.slice, ast.Constant), f"counts[{ast.unparse(node.slice)}] is not a literal label"
            found.append((node.lineno, node.col_offset, 0, node.slice.value))
    labels = []
    for *_, label in sorted(found):
        if label not in labels:
            labels.append(label)
    return tuple(labels)


@pytest.mark.parametrize("name", list(COUNTS))
def test_the_counts_typeddict_has_the_labels_the_code_writes_in_order(name):
    module, function, literal = COUNTS[name]
    keys = tuple(getattr(module, name).__annotations__)
    assert keys == literal
    assert labels_written(getattr(module, function)) == literal
