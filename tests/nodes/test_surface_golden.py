"""G2, the surface of the 11 nodes: everything ComfyUI reads from the pack and a saved workflow
depends on. Per node key, the md5 of the repr of its full INPUT_TYPES (sections, names, order,
types and every option: defaults, ranges, tooltips, round, forceInput, control_after_generate),
RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY, DESCRIPTION, OUTPUT_NODE (absent) and display
name; and the repr of the key list of both mappings, which keeps their order. Recorded from the
code after the SAM naming step, so the two SAM 3.1 Multiplex display names and texts are pinned
in that form.

The mappings are read from the root package, as ComfyUI reads them. The two samplers are read
under the sampler_fakes stubs, so the lists and ranges they take from ComfyUI (SAMPLER_NAMES,
SCHEDULER_NAMES, MAX_RESOLUTION) are the stubs' fixed values. The preprocess nodes read the
real modules, two of which import ComfyUI at their top (models/common/download.py and
pipelines/sam3_1_multiplex/track.py), so this runs where ComfyUI is importable (with the ComfyUI
root on PYTHONPATH) and is skipped elsewhere:

    PYTHONPATH=/path/to/ComfyUI python -m pytest tests/nodes/test_surface_golden.py
"""
import pytest

pytest.importorskip("numpy")
pytest.importorskip("torch")
pytest.importorskip("cv2")
cli_args = pytest.importorskip("comfy.cli_args")
cli_args.args.disable_xformers = True
pytest.importorskip("folder_paths")

import bcvideonodes as pack  # noqa: E402

import golden  # noqa: E402
from sampler_fakes import ANIMATE1, ANIMATE2, node_module  # noqa: E402,F401

SAMPLERS = [ANIMATE1, ANIMATE2]
PREPROCESS = ["BCVPoseDetection", "BCVPoseConfig", "BCVSAM3VideoTrack", "BCVSAM3Config", "BCVFaceCrop",
              "BCVPoseGuard", "BCVMaskGuard", "BCVWanAnimatePreprocess", "BCVWanAnimatePreprocessGuard"]

ABSENT = "<absent>"


def surface(cls, display_name):
    return (
        ("INPUT_TYPES", cls.INPUT_TYPES()),
        ("RETURN_TYPES", cls.RETURN_TYPES),
        ("RETURN_NAMES", cls.RETURN_NAMES),
        ("FUNCTION", cls.FUNCTION),
        ("CATEGORY", cls.CATEGORY),
        ("DESCRIPTION", cls.DESCRIPTION),
        ("OUTPUT_NODE", getattr(cls, "OUTPUT_NODE", ABSENT)),
        ("display name", display_name),
    )


def test_mapping_keys_in_order():
    golden.check(__file__, "NODE_CLASS_MAPPINGS keys", repr(list(pack.NODE_CLASS_MAPPINGS)))
    golden.check(__file__, "NODE_DISPLAY_NAME_MAPPINGS keys", repr(list(pack.NODE_DISPLAY_NAME_MAPPINGS)))


@pytest.mark.parametrize("node_id", SAMPLERS)
def test_sampler_surface(node_module, node_id):
    cls = getattr(node_module, node_id)
    golden.check(__file__, node_id, golden.digest(repr(surface(cls, pack.NODE_DISPLAY_NAME_MAPPINGS[node_id]))))


@pytest.mark.parametrize("node_id", PREPROCESS)
def test_preprocess_surface(node_id):
    cls = pack.NODE_CLASS_MAPPINGS[node_id]
    golden.check(__file__, node_id, golden.digest(repr(surface(cls, pack.NODE_DISPLAY_NAME_MAPPINGS[node_id]))))
