"""Import-time gate for ComfyUI-BCVideoNodes.

Run with a Python that has torch and numpy, and the ComfyUI root on PYTHONPATH:

    PYTHONPATH=/path/to/ComfyUI python tests/test_import_time.py

A standalone script: tests/conftest.py keeps pytest from collecting it. Every measurement runs in
a fresh subprocess with torch and numpy already loaded (ComfyUI has them loaded long before
custom nodes are imported), the repo bound as a package under PKG_NAME the way ComfyUI binds a
custom node, and a temporary working directory, so a bare `nodes` is ComfyUI's and is counted.
A module is heavy when its top-level name is in HEAVY.

  1. the package, its __init__ executed: the median of RUNS runs under BUDGET_S, no heavy
     module, and exactly the NODE_KEYS, in their order;
  2. each node module alone, on a bare package: the nodes package and every module of nodes/:
     no heavy module;
  3. each module of CHECK3_MODULES alone, on a bare package: the heavy modules it pulls in are a
     subset of what ALLOWED gives it (E1_SET, E1_SET | E3_SET, or cv2 and its submodules), and
     nothing for every other module;
  4. each module of CHECK4_MODULES alone, in a subprocess without PYTHONPATH, so ComfyUI is not
     importable: it imports, and afterwards comfy, folder_paths and nodes are not loaded;
  5. the E1 and E3 trigger points: for each node key, in its own subprocess (sys.modules
     persists within a process, so a node run after another would inherit its imports), the
     package bound, the core samplers stubbed and that node's INPUT_TYPES called. E1 is whether
     the detection-folder download module is then imported, E3 whether the SAM 3.1 Multiplex
     module holding the tracker import is. The pairs are compared with the ones recorded in
     tests/goldens/import_gate.json.

Every module of CHECK3_MODULES and CHECK4_MODULES must exist in the tree; a missing one fails.
Every module under libs/, pipelines/ and models/ must be in CHECK3_MODULES; an unlisted one fails.

It also measures E1_SET and E3_SET, the heavy modules `import folder_paths` and the SAM 3.1
tracker import add on this ComfyUI, and prints them. The golden file keeps them, with the
package time and heavy modules, as information: they are not compared, since a core update may
change them. With BCV_GOLDEN_RECORD=1 a value missing from the golden file is recorded
(tests/golden.py); a recorded one is never overwritten.

Exits non-zero and lists what broke.
"""
import importlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
import types

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(TESTS_DIR)
PKG_NAME = "bcvideonodes_under_test"
BUDGET_S = 0.1
RUNS = 3
HEAVY = [
    # ComfyUI-BCNodes' list
    "transformers", "timm", "scipy", "cv2", "PIL", "huggingface_hub",
    "safetensors", "kornia", "einops", "torchvision", "folder_paths",
    "yaml", "postfx", "caption_audit",
    # ComfyUI (a bare `nodes` is its nodes.py), its progress bar, onnx
    "comfy", "comfy_extras", "tqdm", "onnx", "nodes",
]
NODE_KEYS = [
    "BCVWanAnimateLongVideoSampler", "BCVWanAnimate2LongVideoSampler", "BCVPoseDetection", "BCVPoseConfig",
    "BCVSAM3VideoTrack", "BCVSAM3Config", "BCVFaceCrop", "BCVPoseGuard", "BCVMaskGuard",
    "BCVWanAnimatePreprocess", "BCVWanAnimatePreprocessGuard",
]
# each exception's trigger module
E1_MODULE = "models.common.download"
E3_MODULE = "pipelines.sam3_1_multiplex.track"

# check 3: every module under libs/, pipelines/ and models/
CHECK3_MODULES = [
    "libs",
    "libs.log",
    "libs.bbox",
    "libs.keypoints",
    "libs.temporal",
    "libs.mask",
    "libs.chunking",
    "libs.sigmas",
    "libs.video",
    "libs.config_widgets",
    "libs.pose_data",
    "libs.pose_utils",
    "libs.pose_utils.pose2d_utils",
    "libs.pose_utils.human_visualization",
    "pipelines",
    "pipelines.long_video",
    "pipelines.pose",
    "pipelines.face",
    "pipelines.guard",
    "pipelines.guard.config",
    "pipelines.guard.common",
    "pipelines.guard.pose",
    "pipelines.guard.mask",
    "pipelines.guard.report",
    "pipelines.guard.timeline",
    "pipelines.guard.combine",
    "pipelines.sam3_1_multiplex",
    "pipelines.sam3_1_multiplex.config",
    "pipelines.sam3_1_multiplex.prompt",
    "pipelines.sam3_1_multiplex.pose",
    "pipelines.sam3_1_multiplex.track",
    "models",
    "models.common",
    "models.common.registry",
    "models.common.interfaces",
    "models.common.checkpoint",
    "models.common.download",
    "models.common.loader",
    "models.common.wrapper",
    "models.common.blocks",
    "models.common.pose_input",
    "models.common.core_nodes",
    "models.common.animate",
    "models.vitpose",
    "models.vitpose.net",
    "models.vitpose.decode",
    "models.vitpose.wrapper",
    "models.rtmw",
    "models.rtmw.net",
    "models.rtmw.decode",
    "models.rtmw.wrapper",
    "models.yolo",
    "models.yolo.net",
    "models.yolo.wrapper",
    "models.sam3_1_multiplex",
    "models.sam3_1_multiplex.loader",
    "models.sam3_1_multiplex.adapter",
    "models.sam3_1_multiplex.postprocess",
    "models.wan_animate",
    "models.wan_animate.adapter",
    "models.wan_animate.mask_repair",
    "models.wan_animate2",
    "models.wan_animate2.adapter",
    "models.wan_animate2.attention",
]
# check 3: the heavy modules a checked module may pull in; every other module may pull in none.
# "cv2" is the vendored pose_utils' cv2 and its submodules (E2).
ALLOWED = {
    "models.common.download": "E1",
    "models.common.loader": "E1",
    "models.sam3_1_multiplex.loader": "E1",
    "pipelines.sam3_1_multiplex.track": "E1+E3",
    "libs.pose_utils.pose2d_utils": "cv2",
    "libs.pose_utils.human_visualization": "cv2",
}
# check 4: the modules that import without ComfyUI. scripts/ needs the first seven,
# test-scripts the guard and the three SAM 3.1 Multiplex helpers, scripts/convert_models.py
# libs.bbox.
CHECK4_MODULES = [
    "models",
    "models.common.registry",
    "models.common.checkpoint",
    "models.common.pose_input",
    "models.vitpose.decode",
    "models.rtmw.decode",
    "libs.pose_utils.pose2d_utils",
    "pipelines.guard",
    "pipelines.sam3_1_multiplex.config",
    "models.sam3_1_multiplex.postprocess",
    "libs.mask",
    "libs.chunking",
    "libs.bbox",
]
COMFYUI = ("comfy", "folder_paths", "nodes")

GOLDEN = "import_gate"
PAIRS = "E1_E3_trigger_pairs"
INFORMATION = "information_not_compared"


# --- run inside the subprocesses ---------------------------------------------------------------

def bind_package(execute_init):
    """Registers the repo directory as a package under PKG_NAME, the way ComfyUI does for
    custom_nodes. With execute_init=False only the package object is created, so a single
    module can be imported alone."""
    if execute_init:
        spec = importlib.util.spec_from_file_location(
            PKG_NAME, os.path.join(PKG_DIR, "__init__.py"), submodule_search_locations=[PKG_DIR]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[PKG_NAME] = module
        spec.loader.exec_module(module)
        return module
    module = types.ModuleType(PKG_NAME)
    module.__path__ = [PKG_DIR]
    sys.modules[PKG_NAME] = module
    return module


def heavy_since(before):
    return sorted(m for m in set(sys.modules) - before if m.split(".")[0] in HEAVY)


def stub_core_samplers():
    """ComfyUI's sampler and scheduler lists and MAX_RESOLUTION as fixed stubs, so a sampler
    node's INPUT_TYPES does not import ComfyUI's nodes.py, and every core node with it."""
    import comfy

    samplers = types.ModuleType("comfy.samplers")
    samplers.SAMPLER_NAMES = ["euler", "lcm"]
    samplers.SCHEDULER_NAMES = ["normal", "simple", "beta"]
    core_nodes = types.ModuleType("nodes")
    core_nodes.MAX_RESOLUTION = 16384
    sys.modules["comfy.samplers"] = samplers
    sys.modules["nodes"] = core_nodes
    comfy.samplers = samplers


def probe_package():
    before = set(sys.modules)
    t0 = time.perf_counter()
    package = bind_package(execute_init=True)
    seconds = time.perf_counter() - t0
    return {"seconds": seconds, "heavy": heavy_since(before), "keys": list(package.NODE_CLASS_MAPPINGS)}


def probe_module(name):
    bind_package(execute_init=False)
    before = set(sys.modules)
    t0 = time.perf_counter()
    importlib.import_module(PKG_NAME + "." + name)
    seconds = time.perf_counter() - t0
    return {"seconds": seconds, "heavy": heavy_since(before)}


def probe_comfyui_free(name):
    """Imports `name` alone where ComfyUI must not be importable; which of COMFYUI it loaded."""
    reachable = [m for m in COMFYUI if importlib.util.find_spec(m) is not None]
    if reachable:
        return {"reachable": reachable, "loaded": []}
    bind_package(execute_init=False)
    importlib.import_module(PKG_NAME + "." + name)
    return {"reachable": [], "loaded": sorted(m for m in sys.modules if m.split(".")[0] in COMFYUI)}


def probe_exception_set(exception):
    before = set(sys.modules)
    if exception == "E1":
        import folder_paths  # noqa: F401
    else:
        from comfy.ldm.sam3.tracker import MultiplexState, _prep_frame, fill_holes_in_mask_scores  # noqa: F401
    return heavy_since(before)


def probe_triggers(node_key):
    stub_core_samplers()
    package = bind_package(execute_init=True)
    package.NODE_CLASS_MAPPINGS[node_key].INPUT_TYPES()

    return {"E1": PKG_NAME + "." + E1_MODULE in sys.modules, "E3": PKG_NAME + "." + E3_MODULE in sys.modules}


# --- run by main --------------------------------------------------------------------------------

class ProbeFailed(Exception):
    pass


def run_probe(cwd, probe, *args, env=None):
    """`probe(*args)` of this file in a fresh subprocess, with torch and numpy loaded first and
    `cwd` as its working directory (and `env` as its environment, when given); returns what it
    returned."""
    code = (
        "import importlib.util, json\n"
        "import torch, numpy\n"
        f"spec = importlib.util.spec_from_file_location('bcv_import_gate', {json.dumps(os.path.abspath(__file__))})\n"
        "gate = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(gate)\n"
        f"print(json.dumps(gate.{probe}(*{json.dumps(list(args))})))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=cwd, env=env)
    if out.returncode != 0:
        raise ProbeFailed(f"{probe}{tuple(args)!r} exited with {out.returncode}:\n{out.stderr}{out.stdout}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def node_modules():
    """The nodes package and every module of nodes/."""
    folder = os.path.join(PKG_DIR, "nodes")
    return ["nodes"] + sorted("nodes." + f[:-3] for f in os.listdir(folder) if f.endswith(".py") and f != "__init__.py")


def exists(name):
    """Whether the module `name` (relative to the pack) is a file or a package in the tree."""
    path = os.path.join(PKG_DIR, *name.split("."))
    return os.path.isfile(path + ".py") or os.path.isfile(os.path.join(path, "__init__.py"))


def layer_modules():
    """Every module under libs/, pipelines/ and models/, named as CHECK3_MODULES names them."""
    names = []
    for top in ("libs", "pipelines", "models"):
        for folder, _, files in os.walk(os.path.join(PKG_DIR, top)):
            for f in files:
                if f.endswith(".py"):
                    parts = os.path.relpath(os.path.join(folder, f[:-3]), PKG_DIR).split(os.sep)
                    names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return sorted(names)


def resolve(names):
    """(the modules of `names` that exist, in order; the ones that do not)."""
    return [n for n in names if exists(n)], [n for n in names if not exists(n)]


def allowed(name, sets):
    """Check 3's test for the module `name`: whether it may pull in a given heavy module."""
    kind = ALLOWED.get(name)
    if kind == "E1":
        return lambda heavy: heavy in sets["E1"]
    if kind == "E1+E3":
        return lambda heavy: heavy in sets["E1"] or heavy in sets["E3"]
    if kind == "cv2":
        return lambda heavy: heavy.split(".")[0] == "cv2"
    return lambda heavy: False


def repeated(cwd, probe, *args):
    """The median time of RUNS runs, the heavy modules any of them pulled in, and every run's result."""
    runs = [run_probe(cwd, probe, *args) for _ in range(RUNS)]
    heavy = sorted(set().union(*(r["heavy"] for r in runs)))
    return statistics.median(r["seconds"] for r in runs), heavy, runs


def main():
    sys.path.insert(0, TESTS_DIR)
    import golden

    failures = []
    probes_failed = False
    with tempfile.TemporaryDirectory() as cwd:
        # 1. the package
        print(f"{'module':<36}{'import (s)':>12}  heavy imports")
        package_s, package_heavy = None, None
        try:
            package_s, package_heavy, runs = repeated(cwd, "probe_package")
            print(f"{'package':<36}{package_s:>12.5f}  {', '.join(package_heavy) or '-'}")
            if package_s > BUDGET_S:
                failures.append(f"package import took {package_s:.3f}s (median of {RUNS}; budget {BUDGET_S}s)")
            if package_heavy:
                failures.append(f"package import pulled in heavy modules: {', '.join(package_heavy)}")
            for r in runs:
                if r["keys"] != NODE_KEYS:
                    failures.append(f"NODE_CLASS_MAPPINGS keys differ: expected {NODE_KEYS}, found {r['keys']}")
                    break
        except ProbeFailed as error:
            probes_failed = True
            failures.append(f"package: {error}")

        # 2. each node module alone
        for name in node_modules():
            try:
                seconds, heavy, _ = repeated(cwd, "probe_module", name)
            except ProbeFailed as error:
                probes_failed = True
                failures.append(f"{name}: {error}")
                continue
            print(f"{name:<36}{seconds:>12.5f}  {', '.join(heavy) or '-'}")
            if heavy:
                failures.append(f"{name} pulled in heavy modules: {', '.join(heavy)}")

        # the exceptions' sets on this ComfyUI, information only
        sets = {}
        for exception, what in (("E1", "import folder_paths"), ("E3", "the SAM 3.1 tracker import")):
            try:
                sets[exception] = run_probe(cwd, "probe_exception_set", exception)
            except ProbeFailed as error:
                probes_failed = True
                failures.append(f"{exception}_SET: {error}")
                continue
            print(f"\n{exception}_SET ({what}): {', '.join(sets[exception]) or '-'}")

        # 3. every module of the table alone, within its ALLOWED set
        checked, missing = resolve(CHECK3_MODULES)
        for name in missing:
            failures.append(f"check 3: {name} does not exist")
        for name in sorted(set(layer_modules()) - set(CHECK3_MODULES)):
            failures.append(f"check 3: {name} is not in CHECK3_MODULES; add it (and to ALLOWED if it may pull in "
                            f"a heavy module)")
        print(f"\n{'module (check 3)':<44}{'allowed':>8}{'heavy':>7}  heavy imports not allowed")
        if len(sets) < 2:
            failures.append("check 3 needs E1_SET and E3_SET, and a probe measuring them failed")
        else:
            for name in checked:
                try:
                    heavy = run_probe(cwd, "probe_module", name)["heavy"]
                except ProbeFailed as error:
                    failures.append(f"check 3, {name}: {error}")
                    continue
                permitted = allowed(name, sets)
                extra = [m for m in heavy if not permitted(m)]
                print(f"{name:<44}{ALLOWED.get(name, '-'):>8}{len(heavy):>7}  {', '.join(extra) or '-'}")
                if extra:
                    failures.append(f"check 3: {name} pulled in heavy modules it is not allowed: {', '.join(extra)}")

        # 4. the ComfyUI-free set, with no PYTHONPATH
        checked, missing = resolve(CHECK4_MODULES)
        for name in missing:
            failures.append(f"check 4: {name} does not exist")
        print(f"\n{'module (check 4, no ComfyUI)':<44}  ComfyUI modules loaded")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        for name in checked:
            try:
                found = run_probe(cwd, "probe_comfyui_free", name, env=env)
            except ProbeFailed as error:
                failures.append(f"check 4: {name} does not import without ComfyUI: {error}")
                continue
            if found["reachable"]:
                failures.append(f"check 4 cannot run: {', '.join(found['reachable'])} importable without PYTHONPATH")
                break
            print(f"{name:<44}  {', '.join(found['loaded']) or '-'}")
            if found["loaded"]:
                failures.append(f"check 4: {name} loaded {', '.join(found['loaded'])} without ComfyUI on the path")

        # 5. the E1 and E3 trigger points
        print(f"\n{'node key':<36}{'E1':>6}{'E3':>6}")
        pairs = {}
        for key in NODE_KEYS:
            try:
                pairs[key] = run_probe(cwd, "probe_triggers", key)
            except ProbeFailed as error:
                probes_failed = True
                failures.append(f"{key} INPUT_TYPES: {error}")
                continue
            print(f"{key:<36}{'yes' if pairs[key]['E1'] else '-':>6}{'yes' if pairs[key]['E3'] else '-':>6}")

    if probes_failed:
        failures.append("a probe failed, so nothing was compared with or recorded in the golden file")
    else:
        try:
            golden.check(GOLDEN, PAIRS, pairs)
        except AssertionError as error:
            path = os.path.join(golden.GOLDENS, GOLDEN + ".json")
            recorded = {}
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    recorded = json.load(f).get(PAIRS, {})
            for key in NODE_KEYS:
                if key in recorded and recorded[key] != pairs[key]:
                    failures.append(f"{key}: E1/E3 trigger pair recorded {recorded[key]}, found {pairs[key]}")
            failures.append(str(error))
        record_information(golden, {
            "package_seconds": round(package_s, 6),
            "package_heavy": package_heavy,
            "E1_SET": sets["E1"],
            "E3_SET": sets["E3"],
        })

    if failures:
        print("\nFAIL")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nOK")


def record_information(golden, information):
    """Written once, under BCV_GOLDEN_RECORD=1 when the golden file has none, and never compared."""
    path = os.path.join(golden.GOLDENS, GOLDEN + ".json")
    recorded = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            recorded = json.load(f)
    if golden.RECORD and INFORMATION not in recorded:
        golden.check(GOLDEN, INFORMATION, information)


if __name__ == "__main__":
    main()
