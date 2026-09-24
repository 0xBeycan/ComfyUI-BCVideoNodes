"""What the preprocess nodes share: their category, the config built from a node's widget values,
and the config node base.

The preprocess nodes (pose, SAM 3.1 mask, face crop, guards) each call one function of
pipelines/ (the pose node also models/common/loader), imported inside the method on first use;
the two WanAnimate wrappers call the individual nodes, so a wrapper computes exactly what the
chained nodes compute.
"""

from ..libs.config_widgets import config_inputs

PREPROCESS = "BCVideoNodes"


def _config(config_cls, values):
    """`config_cls` built from the widget values that are its fields; the rest are left out."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(config_cls)}
    return config_cls(**{k: v for k, v in values.items() if k in names})


class _ConfigNode:
    """A node that builds a config dataclass from its widgets; the widgets are generated from
    the dataclass, so a new field shows up here without touching this file."""

    FUNCTION = "build"
    CATEGORY = PREPROCESS

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": config_inputs(cls._config_class())}

    def build(self, **values):
        return (self._config_class()(**values),)
