import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import resnet18, resnet34, resnet50
from torchvision.models.resnet import Bottleneck

from .registry import register_model, tv_weights


def set_resnet_fc(model, num_classes):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# arch name -> (torchvision constructor, feature dimension)
_BACKBONES = {
    "resnet18": (resnet18, 512),
    "resnet34": (resnet34, 512),
    "resnet50": (resnet50, 2048),
}


class ResNetBackbone(nn.Module):
    """A torchvision ResNet feature extractor + a linear classification head.

    Shared by the resnet18/34/50 factories below (they differ only in the
    backbone and its feature dimension). Exposes the forward variants the
    mitigators rely on:

    * ``forward(x)``            -> ``(logits, feat)``
    * ``badd_forward(x, f, m)`` -> ``logits`` (BAdd: add bias features to feat)
    * ``mavias_forward(x, f)``  -> ``(logits, logits_on_f)`` (MAVias)

    Submodule names (``extractor``, ``fc``) match the previous per-arch classes,
    so existing checkpoints load unchanged.
    """

    def __init__(self, arch, num_classes=2, pretrained=False):
        super().__init__()
        constructor, embed_size = _BACKBONES[arch]
        model = constructor(weights=tv_weights(pretrained))
        self.extractor = nn.Sequential(*list(model.children())[:-1])
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x, norm=False):
        feat = self.extractor(x)
        feat = feat.squeeze(-1).squeeze(-1)
        if norm:
            feat = F.normalize(feat, dim=1)
        return self.fc(feat), feat

    def badd_forward(self, x, f, m, norm=False):
        feat = torch.flatten(self.extractor(x), 1)
        if norm:
            feat = F.normalize(feat, dim=1)
        feat = feat + torch.sum(torch.stack(f), dim=0) * m
        return self.fc(feat)

    def mavias_forward(self, x, f, norm=False):
        feat = torch.flatten(self.extractor(x), 1)
        if norm:
            feat = F.normalize(feat, dim=1)
            f = F.normalize(f, dim=1)
        return self.fc(feat), self.fc(f)


@register_model("resnet18")
def ResNet18(num_classes=2, pretrained=False):
    return ResNetBackbone("resnet18", num_classes, pretrained)


@register_model("resnet34")
def ResNet34(num_classes=2, pretrained=False):
    return ResNetBackbone("resnet34", num_classes, pretrained)


@register_model("resnet50")
def ResNet50(num_classes=2, pretrained=False):
    return ResNetBackbone("resnet50", num_classes, pretrained)


class ResNet50_Default(models.ResNet):
    def __init__(self):
        super(ResNet50_Default, self).__init__(Bottleneck, [3, 4, 6, 3])

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        f = torch.flatten(x, 1)
        x = F.normalize(f, dim=1)
        x = self.fc(x)

        return x, f


@register_model("resnet50_def")
def resnet50_def(num_classes=2, pretrained=False):
    """ResNet50 with a feature-normalized forward returning (logits, feat).

    ``pretrained`` is ignored (this variant is used as a from-scratch bias
    classifier); the head is resized to ``num_classes``.
    """
    return set_resnet_fc(ResNet50_Default(), num_classes)
