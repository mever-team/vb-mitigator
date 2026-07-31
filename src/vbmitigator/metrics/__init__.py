"""Metrics package.

Public API:
    get_metric(name)         the metric function for cfg.METRIC
    get_metric_meta(name)    {"best", "performance"} selection metadata
    register_metric(name, …) decorator to add a metric
    available_metrics()      list registered metric names
"""

from .registry import (
    available_metrics,
    get_metric,
    get_metric_meta,
    register_metric,
)

__all__ = [
    "get_metric",
    "get_metric_meta",
    "register_metric",
    "available_metrics",
]
