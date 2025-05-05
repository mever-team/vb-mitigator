import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self,num_classes=10, pretrained=False, input_dim=100):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )

    def forward(self, z):
        return self.fc(z)
    
    def set_input_dim(self, input_dim):
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, self.num_classes)
        )