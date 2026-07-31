"""Model construction helpers.

``get_model`` is re-exported from the registry (the single source of truth for
model names). ``get_bcc`` / ``get_local_bccs`` build the bias-capturing
classifiers used by FLAC / BAdd / MAVias-B.
"""

from .registry import get_model
from .utils import get_local_model_dict, get_model_dict

__all__ = ["get_model", "get_bcc", "get_local_bccs"]

# (dataset, bias) -> architecture name used for its bias-capturing classifier.
_BCC_ARCH = {
    ("biased_mnist", "color"): "simple_conv",
    ("fb_biased_mnist", "fgcolor"): "simple_conv",
    ("fb_biased_mnist", "bgcolor"): "simple_conv",
    ("utkface", "race"): "resnet18",
    ("waterbirds", "background"): "resnet50_def",
    ("celeba", "gender"): "resnet18",
    ("urbancars", "background"): "resnet50",
    ("urbancars", "object"): "resnet50",
}


def get_bcc(cfg, num_class):
    """Build the dataset-specific bias-capturing classifiers from downloaded checkpoints.

    Returns ``{bias_name: model}``. Raises ``ValueError`` for an unsupported
    dataset/bias combination.
    """
    dataset_name = cfg.DATASET.TYPE
    nets = {}
    for bias_name in cfg.DATASET.BIASES:
        arch = _BCC_ARCH.get((dataset_name, bias_name))
        if arch is None:
            raise ValueError(
                f"Unsupported dataset ({dataset_name}) or bias ({bias_name}) type."
            )
        state = get_model_dict(dataset_name, bias_name)
        model = get_model(arch, num_class)
        # waterbirds' checkpoint is a bare state_dict; the others wrap it under "model".
        model.load_state_dict(state["model"] if "model" in state else state)
        nets[bias_name] = model
    return nets


def get_local_bccs(cfg, bcc_paths, num_class, device, biases):
    """Load one or more bias-capturing classifiers from local checkpoints.

    Args:
        cfg: config (uses ``cfg.MODEL.TYPE`` for the architecture, shared by all BCCs).
        bcc_paths (list[str]): checkpoint paths, each a dict with a "model" state_dict.
        num_class (int): number of classes of each BCC.
        device: device to place the models on.
        biases (list[str]): bias names, used to key the returned dict (falls back to
            "bcc_{i}" when there are more paths than bias names).

    Returns:
        dict[str, nn.Module]: {name: bcc_model} in eval mode on ``device``.
    """
    bcc_nets = {}
    for i, path in enumerate(bcc_paths):
        state = get_local_model_dict(path)
        net = get_model(cfg.MODEL.TYPE, num_class)
        net.load_state_dict(state["model"])
        net.to(device).eval()
        key = biases[i] if i < len(biases) else f"bcc_{i}"
        bcc_nets[key] = net
    return bcc_nets
