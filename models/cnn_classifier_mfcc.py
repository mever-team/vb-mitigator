import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=False,
    ):
        super(CNNModel, self).__init__()

        # Input shape expected: (batch_size, 1, 40, 173) -> (channels, height, width)
        self.features = nn.Sequential(
            # First Convolutional layer
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0
            ),  # 'valid' padding
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), ceil_mode=True
            ),  # 'same' padding effect
            # Second Convolutional layer
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0
            ),  # 'valid' padding
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), ceil_mode=True
            ),  # 'same' padding effect
            nn.Dropout(0.1),
            # Third Convolutional layer
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0
            ),  # 'valid' padding
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), ceil_mode=True
            ),  # 'same' padding effect
            nn.Dropout(0.1),
            # Fourth Convolutional layer
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0
            ),  # 'valid' padding
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), ceil_mode=True
            ),  # 'same' padding effect
            # nn.Dropout(0.3),
            # nn.Conv2d(
            #     in_channels=256, out_channels=512, kernel_size=(3, 3), padding=0
            # ),  # 'valid' padding
            # nn.ReLU(),
            # nn.MaxPool2d(
            #     kernel_size=(2, 2), stride=(2, 2), ceil_mode=True
            # ),  # 'same' padding effect
            # nn.Dropout(0.3),
        )

        # Calculate the size of the features before the fully connected layers
        # Based on calculations: (Batch, 256, 1, 9) after last MaxPool2d
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling

        # Fully Connected layers
        self.fc2 = nn.Sequential(
            # nn.Linear(512 * 1 * 1, 256),  # Input features after global avg pooling
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x, norm=False):
        # Input x expected shape: (batch_size, n_mfcc, num_frames)
        # Reshape to (batch_size, channels, height, width) -> (batch_size, 1, n_mfcc, num_frames)
        x = x.unsqueeze(1)  # Add channel dimension

        x = self.features(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten for linear layers
        if norm:
            x = F.normalize(x, dim=1)
        out = self.fc2(x)
        return out, x

    def badd_forward(self, x, f, m, norm=False):
        x = x.unsqueeze(1)  # Add channel dimension

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten for linear layers
        if norm:
            x = F.normalize(x, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        x = x + total_f * m  # /2
        out = self.fc2(x)
        return out

    def mavias_forward(self, x, f, norm=False):
        x = x.unsqueeze(1)  # Add channel dimension

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten for linear layers
        if norm:
            x = F.normalize(x, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc2(x)
        logits2 = self.fc2(f)

        return logits, logits2
