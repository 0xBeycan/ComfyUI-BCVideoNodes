"""The dependency rule of the four layers: nodes/ -> pipelines/ -> models/ -> libs/.

Every import of every .py file under the four layers, the root __init__.py and scripts/ is read
with `ast`, at module level and inside functions (the lazy imports), and resolved to the module
it targets: a relative import against the file's package, `from X import name` to `X.name` when
that is a module of the tree and to `X` otherwise, the scripts' `bcvideonodes.<module>` without
the alias. A target's layer is its first component (the root __init__ is `root`); any other
absolute import is ComfyUI core or third-party and has no layer. A violation is reported as
`source:line -> target (rule)`.

1. an edge goes right (or stays in its layer): root -> nodes; nodes -> nodes, pipelines,
   models, libs; pipelines -> pipelines, models, libs; models -> models, libs; libs -> libs;
   scripts -> models, libs. Nothing imports the root.
2. models/<pkg>/ imports only models/<pkg>/, models/common/ and libs/; models/common/ only
   models/common/ and libs/; models/__init__.py imports the model packages (the registration list).
3. no absolute import of the pack's own modules (OWN); an absolute `nodes` is ComfyUI core, and
   only the CORE_SITES may import it. Every core site must exist. No pack code imports through
   the names tests and scripts bind the pack under (ALIASES): inside ComfyUI the pack is bound
   under another name, so such an import would fail there (scripts/ may use `bcvideonodes`).
4. no import by string (importlib.import_module, __import__, spec_from_file_location).
5. the module-level imports (lazy ones excluded) form no cycle.

Standard library only.
"""
import ast
import glob
import os

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALIAS = "bcvideonodes"
LAYERS = ("nodes", "pipelines", "models", "libs")
OWN = ("pipelines", "models", "libs", "preprocess", "chunk_planner")  # rule 3's names, `nodes` apart
ALIASES = (ALIAS, "walong")  # the pack as tests/conftest.py and the sampler tests bind it
CORE_SITES = ("nodes/sampler.py", "models/common/core_nodes.py")
ALLOWED = {
    "root": {"nodes"},
    "nodes": {"nodes", "pipelines", "models", "libs"},
    "pipelines": {"pipelines", "models", "libs"},
    "models": {"models", "libs"},
    "libs": {"libs"},
    "scripts": {"models", "libs"},
}
STRING_IMPORTS = {"import_module", "__import__", "spec_from_file_location"}


def is_module(name):
    """Whether the dotted `name` is a module or a package of the tree."""
    path = os.path.join(PACK, *name.split("."))
    return os.path.isfile(path + ".py") or os.path.isfile(os.path.join(path, "__init__.py"))


def module_of(path):
    """(module name, is a package) of the file `path`, relative to the pack root."""
    parts = path[:-3].split("/")
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]), True
    return ".".join(parts), False


def layer_of(module):
    return module.split(".")[0] if module else "root"


def sources():
    """The files the rules read: the four layers, the root __init__ and scripts/, relative paths."""
    found = ["__init__.py"]
    for top in LAYERS:
        for root, dirs, files in os.walk(os.path.join(PACK, top)):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__")
            found += sorted(os.path.relpath(os.path.join(root, f), PACK).replace(os.sep, "/")
                            for f in files if f.endswith(".py"))
    found += sorted(os.path.relpath(p, PACK).replace(os.sep, "/") for p in glob.glob(os.path.join(PACK, "scripts", "*.py")))
    return found


class Imports(ast.NodeVisitor):
    """Every import of one file as (line, target module or None when it has no layer, lazy,
    absolute top name), and every call that imports by string."""

    def __init__(self, path):
        self.path = path
        self.module, self.is_package = module_of(path)
        self.script = path.startswith("scripts/")
        self.depth = 0
        self.found = []
        self.string_imports = []

    def visit_FunctionDef(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_Lambda = visit_FunctionDef

    def visit_Import(self, node):
        for alias in node.names:
            self.add(node, alias.name, None, 0)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.add(node, node.module, alias.name, node.level)

    def visit_Call(self, node):
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
        if name in STRING_IMPORTS:
            self.string_imports.append(node.lineno)
        self.generic_visit(node)

    def add(self, node, module, name, level):
        lazy = self.depth > 0
        if level:
            package = self.module if self.is_package else self.module.rpartition(".")[0]
            parts = package.split(".") if package else []
            if level - 1 > len(parts):
                self.found.append((node.lineno, "<beyond the pack root>", lazy, None))
                return
            base = ".".join(parts[:len(parts) - (level - 1)])
            target = ".".join(p for p in (base, module) if p)
            top = None
        else:
            top = module.split(".")[0]
            if not (self.script and top == ALIAS):
                # ComfyUI core, third-party, or an absolute self-import (rule 3 reports those)
                self.found.append((node.lineno, None, lazy, top))
                return
            target, top = module.partition(".")[2], None
        if name is not None and is_module(f"{target}.{name}" if target else name):
            target = f"{target}.{name}" if target else name
        self.found.append((node.lineno, target, lazy, top))


def read(path):
    with open(os.path.join(PACK, path), encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    imports = Imports(path)
    imports.visit(tree)
    return imports


def all_imports():
    return {path: read(path) for path in sources()}


def violations(rule):
    found = []
    for path, imports in all_imports().items():
        for line, target, lazy, top in imports.found:
            reason = rule(path, imports.module, target, top)
            if reason and f"{path}:{line} -> {target or top} ({reason})" not in found:
                found.append(f"{path}:{line} -> {target or top} ({reason})")
    return found


def test_every_layer_file_is_walked():
    walked = set(sources())
    on_disk = {os.path.relpath(p, PACK).replace(os.sep, "/")
               for top in LAYERS for p in glob.glob(os.path.join(PACK, top, "**", "*.py"), recursive=True)}
    assert on_disk <= walked, sorted(on_disk - walked)


def test_edges_point_right():
    def rule(path, source, target, top):
        if target is None:
            return None
        if target == "<beyond the pack root>" or layer_of(target) not in ALLOWED[layer_of(source)]:
            return f"rule 1: {layer_of(source)} -> {layer_of(target)}"
        return None

    assert not violations(rule), "\n".join(violations(rule))


def test_model_packages_do_not_import_each_other():
    def rule(path, source, target, top):
        if target is None or layer_of(source) != "models" or layer_of(target) != "models":
            return None
        parts, into = source.split("."), target.split(".")
        if source == "models":
            ok = len(into) >= 2 and into[1] != "common"
        elif parts[1] == "common":
            ok = len(into) >= 2 and into[1] == "common"
        else:
            ok = len(into) >= 2 and into[1] in (parts[1], "common")
        return None if ok else "rule 2: models sideways"

    assert not violations(rule), "\n".join(violations(rule))


def test_no_absolute_self_import():
    def rule(path, source, target, top):
        if top == "nodes" and path not in CORE_SITES or top in OWN:
            return "rule 3: absolute import of the pack's own module"
        if top in ALIASES:
            return "rule 3: an import through a test alias of the pack"
        return None

    found = violations(rule)
    missing = [site for site in CORE_SITES if not os.path.isfile(os.path.join(PACK, site))]
    assert not found and not missing, "\n".join(found + [f"{site}: a core site the rules name does not exist" for site in missing])


def test_no_string_import():
    found = [f"{path}:{line} (rule 4: an import by string)"
             for path, imports in all_imports().items() for line in imports.string_imports]
    assert not found, "\n".join(found)


def test_no_module_level_cycle():
    graph = {}
    for path, imports in all_imports().items():
        for line, target, lazy, top in imports.found:
            if target and not lazy and target != "<beyond the pack root>":
                graph.setdefault(imports.module, set()).add(target)
    state, cycles = {}, []

    def visit(module, stack):
        state[module] = "open"
        for target in sorted(graph.get(module, ())):
            if state.get(target) == "open":
                cycles.append(" -> ".join(stack[stack.index(target):] + [target]))
            elif target not in state:
                visit(target, stack + [target])
        state[module] = "done"

    for module in sorted(graph):
        if module not in state:
            visit(module, [module])
    assert not cycles, "rule 5: module-level import cycles:\n" + "\n".join(cycles)
