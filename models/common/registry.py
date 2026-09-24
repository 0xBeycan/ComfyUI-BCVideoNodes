"""The model registry: family -> name -> Entry(implementation, file).

Each model package registers its names in its __init__.py, after all of its imports, so an import
that fails leaves nothing half-registered; models/__init__.py imports the packages in registration
order. A pipeline picks a model by name from here, and a node's combo list is names() of a family.

Standard library only, and it imports no model package, so every module of models/ can read it.
"""
from typing import NamedTuple, Optional

FAMILIES = ("architecture", "person_detector", "pose_estimator", "animate")


class Entry(NamedTuple):
    implementation: type
    file: Optional[str] = None


_entries = {family: {} for family in FAMILIES}


def _family(family):
    if family not in _entries:
        raise ValueError(f"unknown model family {family!r}; expected one of {', '.join(FAMILIES)}")
    return _entries[family]


def register(family: str, name: str, implementation: type, file: Optional[str] = None) -> None:
    """Adds `name` to `family`, after the names registered before it."""
    entries = _family(family)
    if name in entries:
        raise ValueError(f"{family} {name!r} is registered twice; a model package registers its names once, "
                         f"in its __init__.py")
    entries[name] = Entry(implementation, file)


def names(family: str) -> list[str]:
    """The names of `family` in registration order, as a new list."""
    return list(_family(family))


def get(family: str, name: str) -> Entry:
    """The entry of `name` in `family`; KeyError(name) when it is not registered. A caller that
    must say something else checks names() first and raises its own message."""
    return _family(family)[name]
