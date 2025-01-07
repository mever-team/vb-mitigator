import torch.nn as nn

from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn.functional as F
import torchvision.models as models
import torch

def set_resnet_fc(model, num_classes):
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        model = resnet18()
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 512
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)

    def forward(self, x, norm=False):
        feat = self.extractor(x)
        feat = feat.squeeze(-1).squeeze(-1)
        if norm:
            feat = F.normalize(feat, dim=1)
        logits = self.fc(feat)
        return logits, feat

    def badd_forward(self, x, f, m, norm=False):
        x = self.extractor(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        feat = feat + total_f * m  # /2
        logits = self.fc(feat)
        return logits

    def mavias_forward(self, x, f, norm=False):
        x = self.extractor(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc(feat)
        logits2 = self.fc(f)

        return logits, logits2

class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        model = resnet50()
        modules = list(model.children())[:-1]
        self.extractor = nn.Sequential(*modules)
        self.embed_size = 2048
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_size, num_classes)

    def forward(self, x, norm=False):
        feat = self.extractor(x)
        feat = feat.squeeze(-1).squeeze(-1)
        if norm:
            feat = F.normalize(feat, dim=1)
        logits = self.fc(feat)
        return logits, feat

    def badd_forward(self, x, f, m, norm=False):
        x = self.extractor(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        feat = feat + total_f * m  # /2
        logits = self.fc(feat)
        return logits

    def mavias_forward(self, x, f, norm=False):
        x = self.extractor(x)
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc(feat)
        logits2 = self.fc(f)

        return logits, logits2


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
