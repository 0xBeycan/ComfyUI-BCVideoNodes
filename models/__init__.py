"""The model packages. Importing this package imports every registering package, in registration
order, so the registry (common/registry.py) is complete before any caller reads it: Python
initialises `models` before any `models.*` module. That order is the order registry.names()
returns."""
from . import vitpose, yolo, wan_animate, wan_animate2, scail2  # noqa: F401
