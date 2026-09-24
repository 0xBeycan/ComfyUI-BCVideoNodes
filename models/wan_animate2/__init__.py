"""Wan Animate 2: the animate adapter of the core node WanAnimate2ToVideo."""
from ..common import registry
from .adapter import WanAnimate2Adapter

registry.register("animate", WanAnimate2Adapter.ANIMATE_NODE, WanAnimate2Adapter)
