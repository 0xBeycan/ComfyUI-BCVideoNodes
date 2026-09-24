"""The widgets of a config dataclass: one INPUT_TYPES entry per field, with its default, range and
description, so a config node shows the dataclass's own fields."""

WIDGET_TYPES = {bool: "BOOLEAN", int: "INT", float: "FLOAT", str: "STRING"}


def config_inputs(config_cls):
    """Widgets for every field of the config dataclass `config_cls`, in field order: the
    field's default, and its range and description from the field metadata ("min", "max",
    "step", "tooltip" or "doc"). A field of any other type than bool / int / float / str
    raises; a str field with "choices" in its metadata is a combo."""
    import dataclasses

    inputs = {}
    for f in dataclasses.fields(config_cls):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            raise TypeError("{}.{} has no default; a config node needs one for every field".format(config_cls.__name__, f.name))
        kind = f.type if isinstance(f.type, type) else {"bool": bool, "int": int, "float": float, "str": str}.get(f.type, type(default))
        if kind not in WIDGET_TYPES:
            raise TypeError("{}.{} is a {}; a config node can only show bool, int, float and str fields".format(config_cls.__name__, f.name, kind))
        meta = f.metadata
        options = {"default": kind(default)}
        for key in ("min", "max", "step"):
            if key in meta and kind in (int, float):
                options[key] = kind(meta[key])
        tooltip = meta.get("tooltip") or meta.get("doc")
        if tooltip:
            options["tooltip"] = tooltip
        if kind is str and "choices" in meta:
            inputs[f.name] = (list(meta["choices"]), options)
        else:
            inputs[f.name] = (WIDGET_TYPES[kind], options)
    return inputs
