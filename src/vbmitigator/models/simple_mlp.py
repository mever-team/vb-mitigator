import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)  # Dense layer

    def forward(self, x):
        x = self.fc1(x)
        return x
