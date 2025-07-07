import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_layer import DotAttention, SoftAttention


class LSTM_Attn_Classifier(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrained=False,
        inp_size=768,
        hidden_size=256,
        return_attn_weights=False,
        attn_type="soft",
    ):
        super(LSTM_Attn_Classifier, self).__init__()
        self.return_attn_weights = return_attn_weights
        self.lstm = nn.LSTM(inp_size, hidden_size, batch_first=True)
        self.attn_type = attn_type

        if self.attn_type == "dot":
            self.attention = DotAttention()
        elif self.attn_type == "soft":
            self.attention = SoftAttention(hidden_size, hidden_size)

        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x, norm=False):

        lstm_out, (hidden, _) = self.lstm(x)

        if self.attn_type == "dot":
            attn_output = self.attention(lstm_out, hidden)
            # attn_weights = self.attention._get_weights(lstm_out, hidden)
        elif self.attn_type == "soft":
            attn_output = self.attention(lstm_out)
            # attn_weights = self.attention._get_weights(lstm_out)
        if norm:
            attn_output = F.normalize(attn_output, dim=1)
        out = self.fc2(attn_output)

        return out, attn_output

    def badd_forward(self, x, f, m, norm=False):
        lstm_out, (hidden, _) = self.lstm(x)

        if self.attn_type == "dot":
            attn_output = self.attention(lstm_out, hidden)
            # attn_weights = self.attention._get_weights(lstm_out, hidden)
        elif self.attn_type == "soft":
            attn_output = self.attention(lstm_out)
            # attn_weights = self.attention._get_weights(lstm_out)
        if norm:
            attn_output = F.normalize(attn_output, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        attn_output = attn_output + total_f * m  # /2
        out = self.fc2(attn_output)
        return out

    def mavias_forward(self, x, f, norm=False):
        lstm_out, (hidden, _) = self.lstm(x)

        if self.attn_type == "dot":
            attn_output = self.attention(lstm_out, hidden)
            # attn_weights = self.attention._get_weights(lstm_out, hidden)
        elif self.attn_type == "soft":
            attn_output = self.attention(lstm_out)
            # attn_weights = self.attention._get_weights(lstm_out)
        if norm:
            attn_output = F.normalize(attn_output, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc2(attn_output)
        logits2 = self.fc2(f)

        return logits, logits2
