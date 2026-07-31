import torch.nn as nn
from torchvision.models import vit_b_16 as _vit_b_16

from .registry import register_model, tv_weights


@register_model("vit_b_16")
class VisionTransformer(nn.Module):
    """torchvision ViT-B/16 as a feature extractor + a linear head.

    ``forward(x) -> (logits, feat)``, matching the other backbones.
    """

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        model = _vit_b_16(weights=tv_weights(pretrained))
        self.embed_size = model.heads.head.in_features
        model.heads.head = nn.Identity()  # strip the classification head
        self.extractor = model
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)

    def forward(self, x, norm=False):
        feat = self.extractor(x)
        logits = self.fc(feat)
        return logits, feat
