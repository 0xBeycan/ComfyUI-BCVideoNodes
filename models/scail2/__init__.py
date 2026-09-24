"""SCAIL-2: the animate adapter of the core node WanSCAILToVideo."""
from ..common import registry
from .adapter import SCAIL2Adapter

registry.register("animate", SCAIL2Adapter.ANIMATE_NODE, SCAIL2Adapter)
