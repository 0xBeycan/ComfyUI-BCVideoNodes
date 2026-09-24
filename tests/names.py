"""The test-side view of pack names: where a test body reads, and patches, a pack name.

Test bodies read module attributes through a few module-level names (`sam3`, `pose`, `nodes`,
...). When the module holding a name changes, only a row of that name's table changes, never the
body that reads it. A table maps an attribute name to one of:

- `Ref(module, attr)`: the attribute, or the module itself when `attr` is None, looked up at
  access time, so a value a test has just patched on the module is what the body reads;
- `Seam(Ref, ...)`: the same read, and `setattr` / `delattr` on the `Names` object are forwarded
  to every listed module (the ones the production code reads the name from at call time), so
  `monkeypatch.setattr(sam3, "track", fake)` patches them all and its undo restores them all;
- `Value(fn)`: `fn()` evaluated at access time, for a value no module holds any more.

A `Ref` module is relative to the pack alias (`bcvideonodes`, or `walong` for the sampler
fixture; "" is the root package) when its first component is a file or directory of the pack,
and an absolute module name otherwise (`comfy.model_management`). A name missing from a table,
or a `setattr` on a row that is not a `Seam`, raises AttributeError naming the table, so a seam
nobody retargeted fails loudly instead of patching nothing.

The cross-domain `nodes` table is here; each domain's tables are in its `*_fakes.py`. Standard
library only at module level: tests/test_package.py runs with nothing but pytest installed.
"""
import importlib
import os
import sys


def _module_name(module, alias):
    """The importable name of `module`: under the alias when the pack holds it."""
    if not module:
        return alias
    if alias not in sys.modules:
        raise LookupError(f"the pack alias {alias!r} is not bound; tests/conftest.py binds bcvideonodes, "
                          f"the node_module fixture binds walong")
    root = sys.modules[alias].__path__[0]
    first = module.split(".")[0]
    if os.path.isdir(os.path.join(root, first)) or os.path.isfile(os.path.join(root, first + ".py")):
        return f"{alias}.{module}"
    return module


class Ref:
    """`module.attr`, or `module` itself when `attr` is None, read at access time."""

    def __init__(self, module, attr=None):
        self.module = module
        self.attr = attr

    def target(self, alias):
        return importlib.import_module(_module_name(self.module, alias))

    def get(self, alias):
        module = self.target(alias)
        return module if self.attr is None else getattr(module, self.attr)

    def __repr__(self):
        return f"Ref({self.module!r}, {self.attr!r})"


class Seam:
    """A patchable name: read from the first `Ref`, set on every `Ref`'s module."""

    def __init__(self, *refs):
        if not refs or any(ref.attr is None for ref in refs):
            raise ValueError(f"a Seam needs one or more Refs to module attributes, found {refs!r}")
        self.refs = refs

    def get(self, alias):
        return self.refs[0].get(alias)

    def _same_everywhere(self, alias):
        held = [ref.get(alias) for ref in self.refs]
        if any(value is not held[0] for value in held):
            raise AssertionError(f"{self!r}: the targets hold different objects, so an undo could not "
                                 f"restore each of them: {held!r}")

    def set(self, alias, value):
        self._same_everywhere(alias)
        for ref in self.refs:
            setattr(ref.target(alias), ref.attr, value)

    def delete(self, alias):
        self._same_everywhere(alias)
        for ref in self.refs:
            delattr(ref.target(alias), ref.attr)

    def __repr__(self):
        return "Seam({})".format(", ".join(map(repr, self.refs)))


class Value:
    """`fn()` evaluated at access time."""

    def __init__(self, fn):
        self.fn = fn

    def get(self, alias):
        return self.fn()

    def __repr__(self):
        return f"Value({self.fn!r})"


def refs(module, *names):
    """Rows reading each name from `module`."""
    return {name: Ref(module, name) for name in names}


def seams(module, *names):
    """Rows reading and patching each name on `module`."""
    return {name: Seam(Ref(module, name)) for name in names}


class Names:
    """The names a test body reads, by the rows of one table."""

    def __init__(self, table, rows, alias="bcvideonodes"):
        self.__dict__["_Names__table"] = table
        self.__dict__["_Names__rows"] = dict(rows)
        self.__dict__["_Names__alias"] = alias

    def __row(self, name):
        rows = self.__dict__["_Names__rows"]
        if name not in rows:
            raise AttributeError(f"{name!r} is not in the {self.__dict__['_Names__table']!r} Names table: "
                                 f"add a row for it")
        return rows[name]

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        return self.__row(name).get(self.__alias)

    def __setattr__(self, name, value):
        row = self.__row(name)
        if not isinstance(row, Seam):
            raise AttributeError(f"{name!r} in the {self.__table!r} Names table is {row!r}, not a Seam: "
                                 f"setting it would patch nothing the code reads")
        row.set(self.__alias, value)

    def __delattr__(self, name):
        row = self.__row(name)
        if not isinstance(row, Seam):
            raise AttributeError(f"{name!r} in the {self.__table!r} Names table is {row!r}, not a Seam")
        row.delete(self.__alias)

    def __repr__(self):
        return f"<Names {self.__table!r} over {self.__alias!r}>"


# --- the cross-domain table: the pack's nodes -------------------------------------------------

nodes = Names("nodes", {
    "NODE_CLASS_MAPPINGS": Ref("", "NODE_CLASS_MAPPINGS"),
    **refs("nodes.pose", "BCVPoseDetection"),
    **refs("nodes.sam3_1_multiplex", "BCVSAM3VideoTrack"),
    **refs("nodes.face", "BCVFaceCrop"),
    **refs("nodes.guard", "BCVPoseGuard", "BCVMaskGuard"),
    **refs("nodes.preprocess", "BCVWanAnimatePreprocess", "BCVWanAnimatePreprocessGuard"),
    **refs("nodes.scail2", "BCVSCAIL2ColoredMask", "BCVSCAIL2Preprocess"),
    "_config_inputs": Ref("libs.config_widgets", "config_inputs"),
})


def spec(node_id):
    return nodes.NODE_CLASS_MAPPINGS[node_id].INPUT_TYPES()
