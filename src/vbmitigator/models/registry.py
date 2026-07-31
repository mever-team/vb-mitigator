"""Model registry — the single extension point for models.

Adding a model takes one decorator on a builder ``(num_classes, pretrained) ->
nn.Module`` in any module under ``vbmitigator/models/``::

    from vbmitigator.models.registry import register_model

    @register_model("my_net")
    def build_my_net(num_classes, pretrained=False):
        return MyNet(num_classes, pretrained)

The module is auto-discovered — no central file to edit. Select it from configs
with ``MODEL.TYPE: my_net``.
"""

import importlib
import pkgutil
import warnings

# Registered builders: name -> callable(num_classes, pretrained) -> nn.Module.
_MODEL_REGISTRY = {}
_DISCOVERED = False

# Package modules that are infrastructure, not model definitions.
_NON_MODEL_MODULES = {"registry", "builder", "utils"}


def register_model(name):
    """Decorator registering ``fn`` (a builder or an nn.Module class) as ``name``."""

    def decorator(fn):
        existing = _MODEL_REGISTRY.get(name)
        if existing is not None and existing is not fn:
            raise ValueError(f"model '{name}' is already registered")
        _MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def _discover():
    """Import every model module once so its ``@register_model`` runs."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    import vbmitigator.models as pkg

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name in _NON_MODEL_MODULES:
            continue
        try:
            importlib.import_module(f"vbmitigator.models.{info.name}")
        except Exception as e:  # a broken/optional module shouldn't kill discovery
            warnings.warn(f"model module '{info.name}' could not be imported: {e}")


def available_models():
    """Sorted list of registered model names."""
    _discover()
    return sorted(_MODEL_REGISTRY)


def get_model(model_name, num_class, pretrained=False):
    """Build the model registered under ``model_name``."""
    _discover()
    if model_name not in _MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_name}'. Available: {', '.join(available_models())}"
        )
    return _MODEL_REGISTRY[model_name](num_class, pretrained)


def tv_weights(pretrained):
    """Map a legacy ``pretrained`` bool to the torchvision ``weights`` argument.

    torchvision deprecated ``pretrained=`` in 0.13 in favor of ``weights=``;
    ``"DEFAULT"`` selects the recommended pretrained weights, ``None`` means
    random init.
    """
    return "DEFAULT" if pretrained else None
