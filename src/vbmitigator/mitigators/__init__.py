"""Registry of bias-mitigation trainers.

Trainers are resolved lazily: selecting a method only imports that method's
module. This keeps the heavy, optional dependencies of the tag-based methods
(``mavias``/``maviasb``/``erm_tags`` need ``recognize-anything``, ``ollama``,
``transformers``) out of the import path for everyone else.

Usage::

    from vbmitigator.mitigators import get_trainer
    trainer = get_trainer(cfg.MITIGATOR.TYPE)(cfg)

``method_to_trainer[name]`` is also supported and returns the trainer class.
"""

import importlib

# name -> (module suffix, class name)
_REGISTRY = {
    "erm": ("erm", "ERMTrainer"),
    "erm_tags": ("erm_tags", "ERMTagsTrainer"),
    "flac": ("flac", "FLACTrainer"),
    "flacb": ("flacb", "FLACBTrainer"),
    "badd": ("badd", "BAddTrainer"),
    "mavias": ("mavias", "MAVIASTrainer"),
    "maviasb": ("maviasb", "MAVIASBTrainer"),
    "groupdro": ("groupdro", "GroupDROTrainer"),
    "debian": ("debian", "DebiANTrainer"),
    "di": ("domain_independent", "DomainIndependentTrainer"),
    "sd": ("spectral_decouple", "SpectralDecoupleTrainer"),
    "lff": ("lff", "LfFTrainer"),
    "bb": ("bb", "BBTrainer"),
    "end": ("end", "EndTrainer"),
    "jtt": ("jtt", "JTTTrainer"),
    "softcon": ("softcon", "SoftConTrainer"),
}

#: Human-readable list of registered method names.
AVAILABLE_METHODS = sorted(_REGISTRY)


def get_trainer(name):
    """Return the trainer *class* registered under ``name`` (imported lazily)."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown mitigator '{name}'. Available: {', '.join(AVAILABLE_METHODS)}"
        )
    module_suffix, class_name = _REGISTRY[name]
    module = importlib.import_module(f"vbmitigator.mitigators.{module_suffix}")
    return getattr(module, class_name)


class _LazyTrainerMap:
    """Mapping-like view returning trainer classes on demand."""

    def __getitem__(self, name):
        return get_trainer(name)

    def __contains__(self, name):
        return name in _REGISTRY

    def __iter__(self):
        return iter(AVAILABLE_METHODS)

    def keys(self):
        return list(AVAILABLE_METHODS)


#: Backwards-compatible lazy mapping: ``method_to_trainer[name] -> class``.
method_to_trainer = _LazyTrainerMap()
