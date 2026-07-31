"""VB-Mitigator: a framework for visual bias mitigation research."""

__version__ = "0.1.0"

from vbmitigator.mitigators import AVAILABLE_METHODS, get_trainer

__all__ = ["get_trainer", "AVAILABLE_METHODS", "__version__"]
