"""The model packages. Importing this package imports every registering package, in registration
order, so the registry (common/registry.py) is complete before any caller reads it: Python
initialises `models` before any `models.*` module. That order is the order of the combo lists
and of the architecture list."""
from . import vitpose, rtmw, yolo, wan_animate, wan_animate2, wan_scail2  # noqa: F401
