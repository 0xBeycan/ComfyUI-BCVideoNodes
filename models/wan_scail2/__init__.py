"""Wan 2.1 SCAIL-2: the animate adapter of the core node WanSCAILToVideo."""
from ..common import registry
from .adapter import WanSCAIL2Adapter

registry.register("animate", WanSCAIL2Adapter.ANIMATE_NODE, WanSCAIL2Adapter)
