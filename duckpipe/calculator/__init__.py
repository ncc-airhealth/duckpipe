"""
Calculator package utilities.

This module provides auto-discovery of calculator mixin classes so callers don't
need to manually import and add them to the main `Calculator` class. Any class
in this package whose name ends with 'Calculator' will be treated as a mixin and
included automatically.

Notes
- Only classes defined in their module (cls.__module__ == module.__name__) are selected.
- Classes are returned in a deterministic order (alphabetical by class name) to
  keep the MRO stable across runs.
"""
from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from typing import Tuple, Type


def discover_calculator_classes() -> Tuple[Type]:
    """
    Discover all classes in `duckpipe.calculator/` whose class name matches the
    module (file) name. This supports both support classes like `Worker`,
    `Clustering` and all `*Calculator` mixins, assuming the convention that the
    file name equals the class name (e.g., `LanduseCalculator.py` contains
    `LanduseCalculator`).

    Ordering
    - `Worker`, then `Clustering`, then all others sorted alphabetically by class name.

    Returns
    - List[type]: Deterministically ordered list of classes.
    """
    # prepare
    classes: list[type] = []
    pkg_path = os.path.dirname(__file__)
    # find modules
    for m in pkgutil.iter_modules([pkg_path]):
        if not m.name.startswith("_"):
            module = importlib.import_module(f"duckpipe.calculator.{m.name}")
            class_name = module.__name__.rsplit('.', 1)[-1]
            obj = getattr(module, class_name, None)
            if inspect.isclass(obj) and getattr(obj, "__module__", None) == module.__name__:
                classes.append(obj)
    # return
    classes_tuple = tuple(classes)
    return classes_tuple