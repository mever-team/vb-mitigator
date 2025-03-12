from models.builder import get_model
import torch
from torch import nn

class DomainIndependentClassifier(nn.Module):
    def __init__(self, arch, num_classes, num_domain, pretrained):
        super(DomainIndependentClassifier, self).__init__()
        self.backbone = get_model(
            arch,
            num_classes,
            pretrained=pretrained
        )
        self.domain_classifier_list = nn.ModuleList(
            [
                nn.Linear(self.backbone.fc.in_features, num_classes)
                for _ in range(int(num_domain))
            ]
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, tuple):
                x, _ = x
        logits_per_domain = [
            classifier(x) for classifier in self.domain_classifier_list
        ]
        logits_per_domain = torch.stack(logits_per_domain, dim=1)

        if self.training:
            return logits_per_domain
        else:
            return logits_per_domain.mean(dim=1)