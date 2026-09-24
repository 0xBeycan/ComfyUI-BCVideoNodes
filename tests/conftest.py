"""Binds the pack the way ComfyUI loads it: the repo root imported from its __init__.py as the
package `bcvideonodes`, so the tests import pack code as `bcvideonodes.<module>` and its relative
imports resolve. This is the root __init__'s only execution in a test process: tests/pytest.ini
anchors pytest's rootdir at tests/, so pytest itself never imports the repo root.

Standard library only at module level: tests/test_package.py and tests/libs/test_chunking.py run
with nothing but pytest installed. pytest puts this directory on sys.path, which is how the tests
import names.py, golden.py and the *_fakes.py modules.
"""
import importlib.util
import os
import sys

# a standalone script, run as python tests/test_import_time.py
collect_ignore = ["test_import_time.py"]

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if "bcvideonodes" not in sys.modules:
    _spec = importlib.util.spec_from_file_location("bcvideonodes", os.path.join(PACK, "__init__.py"),
                                                   submodule_search_locations=[PACK])
    _package = importlib.util.module_from_spec(_spec)
    sys.modules["bcvideonodes"] = _package
    _spec.loader.exec_module(_package)
