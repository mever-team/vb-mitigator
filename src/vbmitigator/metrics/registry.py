"""Metric registry — the single extension point for metrics.

Adding a metric takes one auto-discovered decorator on a function
``data_dict -> {key: value, ...}`` in any module under ``vbmitigator/metrics/``::

    from vbmitigator.metrics.registry import register_metric

    @register_metric("my_metric", performance="score", best="high")
    def my_metric(data):
        ...
        return {"score": value, ...}

``performance`` names the output key that drives best-checkpoint selection and
``best`` is "high" or "low". Select it from configs with ``METRIC: my_metric``.
"""

import importlib
import pkgutil
import warnings

# name -> {"fn": callable, "best": "high"|"low", "performance": str}
_METRIC_REGISTRY = {}
_DISCOVERED = False

_NON_METRIC_MODULES = {"registry", "utils"}


def register_metric(name, performance, best="high"):
    """Register ``fn`` as metric ``name`` with its best-selection metadata."""
    if best not in ("high", "low"):
        raise ValueError(f"metric '{name}': best must be 'high' or 'low', got {best!r}")

    def decorator(fn):
        existing = _METRIC_REGISTRY.get(name)
        if existing is not None and existing["fn"] is not fn:
            raise ValueError(f"metric '{name}' is already registered")
        _METRIC_REGISTRY[name] = {"fn": fn, "best": best, "performance": performance}
        return fn

    return decorator


def _discover():
    """Import every metric module once so its ``@register_metric`` runs."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    import vbmitigator.metrics as pkg

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name in _NON_METRIC_MODULES:
            continue
        try:
            importlib.import_module(f"vbmitigator.metrics.{info.name}")
        except Exception as e:  # a broken module shouldn't kill discovery
            warnings.warn(f"metric module '{info.name}' could not be imported: {e}")


def available_metrics():
    """Sorted list of registered metric names."""
    _discover()
    return sorted(_METRIC_REGISTRY)


def _entry(name):
    _discover()
    if name not in _METRIC_REGISTRY:
        raise KeyError(
            f"Unknown metric '{name}'. Available: {', '.join(available_metrics())}"
        )
    return _METRIC_REGISTRY[name]


def get_metric(name):
    """Return the metric function registered under ``name``."""
    return _entry(name)["fn"]


def get_metric_meta(name):
    """Return ``{"best", "performance"}`` for metric ``name``."""
    entry = _entry(name)
    return {"best": entry["best"], "performance": entry["performance"]}
