import torch.nn as nn
from torchvision.models import efficientnet_b0 as _efficientnet_b0

from .registry import register_model, tv_weights


@register_model("efficientnet_b0")
class EfficientNetB0(nn.Module):
    """torchvision EfficientNet-B0 feature extractor + a linear head.

    ``forward(x) -> (logits, feat)``, matching the other backbones.
    """

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        model = _efficientnet_b0(weights=tv_weights(pretrained))
        self.extractor = nn.Sequential(*list(model.children())[:-1])
        self.embed_size = 1280
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)

    def forward(self, x, norm=False):
        out = self.extractor(x)
        out = out.squeeze(-1).squeeze(-1)
        logits = self.fc(out)
        return logits, out
