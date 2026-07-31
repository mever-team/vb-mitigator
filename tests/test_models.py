"""Model zoo + registry: every registered model builds and runs a forward pass,
and the unified ResNet backbone keeps its badd/mavias forward variants."""

import pytest
import torch

from vbmitigator.models import available_models, get_model, register_model

# Minimum spatial size each model needs for a forward pass.
INPUT_SIZE = {
    "simple_conv": 32,
    "resnet8": 32,
    "resnet20": 32,
    "resnet32": 32,
    "resnet18": 224,
    "resnet34": 224,
    "resnet50": 224,
    "resnet50_def": 224,
    "efficientnet_b0": 224,
    "vit_b_16": 224,
}


def test_registry_matches_expected():
    assert set(available_models()) == set(INPUT_SIZE)


@pytest.mark.parametrize("name", sorted(INPUT_SIZE))
def test_model_forward(name):
    size = INPUT_SIZE[name]
    model = get_model(name, 4, pretrained=False).eval()
    x = torch.randn(2, 3, size, size)
    with torch.no_grad():
        out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (2, 4)


@pytest.mark.parametrize("name,embed", [("resnet18", 512), ("resnet34", 512), ("resnet50", 2048)])
def test_resnet_badd_and_mavias_forward(name, embed):
    model = get_model(name, 4, pretrained=False).eval()
    x = torch.randn(2, 3, 224, 224)
    f = torch.randn(2, embed)
    with torch.no_grad():
        badd_logits = model.badd_forward(x, [f, f], m=1.0)
        mav_logits, mav_on_f = model.mavias_forward(x, f)
    assert badd_logits.shape == (2, 4)
    assert mav_logits.shape == (2, 4)
    assert mav_on_f.shape == (2, 4)


def test_checkpoint_state_dict_keys_stable():
    """The unified backbone keeps extractor.*/fc.* keys so old checkpoints load."""
    model = get_model("resnet18", 4, pretrained=False)
    keys = set(model.state_dict())
    assert any(k.startswith("extractor.") for k in keys)
    assert {"fc.weight", "fc.bias"} <= keys


def test_add_a_model_in_one_call():
    @register_model("toy_unit_model")
    def _build(num_classes, pretrained=False):
        return torch.nn.Linear(4, num_classes)

    assert "toy_unit_model" in available_models()
    model = get_model("toy_unit_model", 3)
    assert model(torch.randn(2, 4)).shape == (2, 3)


def test_unknown_model_raises_with_hint():
    with pytest.raises(KeyError) as exc:
        get_model("does_not_exist", 2)
    assert "Available" in str(exc.value)
