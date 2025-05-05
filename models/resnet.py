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
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        model = resnet18(pretrained=pretrained)
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

class ResNet18MultiHead(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        
        model = resnet18(pretrained=pretrained)
        layers = list(model.children())
        # print(layers)
        # print(50*"-")
        self.layer1 = nn.Sequential(*layers[:4])  # Conv1 + BN + ReLU + MaxPool
        self.layer2 = layers[4]  # Layer1 (First residual block)
        self.layer3 = layers[5]  # Layer2
        self.layer4 = layers[6]  # Layer3
        self.layer5 = layers[7]  # Layer4
        self.global_avg_pool = layers[8]  # Global Average Pooling
        # print(50*"-")
        # print(self.layer1)
        # print(50*"-")
        # print(self.layer2)
        # print(50*"-")
        # print(self.layer3)
        # print(50*"-")
        # print(self.layer4)
        # print(50*"-")
        # print(self.layer5)
        self.embed_size = 512
        self.num_classes = num_classes
        
        # Classification heads at each stage
        self.fc1 = nn.Linear(64, num_classes)  # After first conv
        self.fc2 = nn.Linear(64, num_classes)  # After first residual block
        self.fc3 = nn.Linear(128, num_classes)  # After second residual block
        self.fc4 = nn.Linear(256, num_classes)  # After third residual block
        self.fc5 = nn.Linear(512, num_classes)  # After fourth residual block (final layer before pooling)

        self.pr1 = nn.Linear(768,64)
        self.pr2 = nn.Linear(768,64)
        self.pr3 = nn.Linear(768,128)
        self.pr4 = nn.Linear(768,256)
        self.pr5 = nn.Linear(768,512)

    def mavias_forward(self, x, f, norm=False):
        
        x = self.layer1(x)
        feat1 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        logits1 = self.fc1(feat1)
        clogits1 = self.fc1(self.pr1(f))
        
        x = self.layer2(x)
        feat2 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        logits2 = self.fc2(feat2)
        clogits2 = self.fc2(self.pr2(f))
        
        x = self.layer3(x)
        feat3 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        logits3 = self.fc3(feat3)
        clogits3 = self.fc3(self.pr3(f))

        x = self.layer4(x)
        feat4 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        logits4 = self.fc4(feat4)
        clogits4 = self.fc4((self.pr4(f)))
        
        x = self.layer5(x)
        # feat5 = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        feat5 = self.global_avg_pool(x)
        feat5 = feat5.squeeze(-1).squeeze(-1)
        logits5 = self.fc5(feat5)
        clogits5 = self.fc5((self.pr5(f)))

        return [logits1, logits2, logits3, logits4, logits5], [clogits1, clogits2, clogits3, clogits4, clogits5]
    
    def forward(self, x, norm=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)      
        x = self.layer4(x)
        x = self.layer5(x)
        feat = self.global_avg_pool(x)
        feat = feat.squeeze(-1).squeeze(-1)
        logits = self.fc5(feat)
        
        return logits, feat


class ResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()

        model = resnet50(pretrained=pretrained)
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
