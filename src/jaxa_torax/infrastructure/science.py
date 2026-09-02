"""Stable lazy facade for the scientific infrastructure.

Implementation lives in focused modules.  Attribute lookup is delegated lazily so
the desktop and cache CLI retain one stable import path without wildcard imports
or eagerly importing every optional science dependency.
"""

from importlib import import_module

_MODULE_NAMES = (
    "cache_repository", "catalog_index", "catalog_service", "event_sources",
    "images", "observation_loader", "previews", "science_core", "selections",
    "workspace_repository",
)


def __getattr__(name: str):
    for module_name in _MODULE_NAMES:
        module = import_module(f"{__package__}.{module_name}")
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals())
    for module_name in _MODULE_NAMES:
        module = import_module(f"{__package__}.{module_name}")
        names.update(name for name in vars(module) if not name.startswith("_"))
    return sorted(names)
