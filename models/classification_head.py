import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    # self, num_classes=10, pretrained=False, input_dim=384, hidden_dim=256, dr=0.3 for text
    # self, num_classes=10, pretrained=False, input_dim=768, hidden_dim=256, dr=0.0
    def __init__(
        self, num_classes=10, pretrained=False, input_dim=384, hidden_dim=256, dr=0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dr = nn.Dropout(dr)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, norm=False):
        feat = self.dr(self.relu(self.fc1(x)))
        feat = feat.squeeze(-1).squeeze(-1)
        if norm:
            feat = F.normalize(feat, dim=1)
        logits = self.fc2(feat)
        return logits, feat

    def badd_forward(self, x, f, m, norm=False):
        x = self.dr(self.relu(self.fc1(x)))
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        feat = feat + total_f * m  # /2
        logits = self.fc2(feat)
        return logits

    def mavias_forward(self, x, f, norm=False):
        x = self.dr(self.relu(self.fc1(x)))
        feat = torch.flatten(x, 1)
        if norm:
            feat = F.normalize(feat, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc2(feat)
        logits2 = self.fc2(f)

        return logits, logits2

    def set_input_dim(self, input_dim, hidden_dim=100, dr=0.3):
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dr = nn.Dropout(dr)
        self.fc2 = nn.Linear(hidden_dim, self.num_classes)
