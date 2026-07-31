"""Models package.

Public API:
    get_model(name, num_classes, pretrained)   build a registered model
    register_model(name)                        decorator to add a model
    available_models()                          list registered model names
"""

from .builder import get_bcc, get_local_bccs, get_model
from .registry import available_models, register_model

__all__ = [
    "get_model",
    "get_bcc",
    "get_local_bccs",
    "register_model",
    "available_models",
]
