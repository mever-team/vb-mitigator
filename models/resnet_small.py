from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["resnet"]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(
        self,
        depth,
        num_filters,
        block_name="BasicBlock",
        num_classes=10,
        pretrained=False,
    ):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == "basicblock":
            assert (
                depth - 2
            ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (
                depth - 2
            ) % 9 == 0, "When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.stage_channels = num_filters

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(
            block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1))
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError("ResNet unknown block error !!!")

        return [bn1, bn2, bn3]

    def get_stage_channels(self):
        return self.stage_channels

    def forward(self, x, norm=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        x, _ = self.layer1(x)  # 32x32
        x, _ = self.layer2(x)  # 16x16
        x, _ = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        avg = x.reshape(x.size(0), -1)
        if norm:
            avg = F.normalize(avg, dim=1)
        out = self.fc(avg)
        return out, avg

    def badd_forward(self, x, f, m, norm=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        x, _ = self.layer1(x)  # 32x32
        x, _ = self.layer2(x)  # 16x16
        x, _ = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        feat = x.reshape(x.size(0), -1)
        if norm:
            feat = F.normalize(feat, dim=1)
        total_f = torch.sum(torch.stack(f), dim=0)
        feat = feat + total_f * m  # /2
        logits = self.fc(feat)
        return logits

    def mavias_forward(self, x, f, norm=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        x, _ = self.layer1(x)  # 32x32
        x, _ = self.layer2(x)  # 16x16
        x, _ = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        feat = x.reshape(x.size(0), -1)
        if norm:
            feat = F.normalize(feat, dim=1)
            f = F.normalize(f, dim=1)

        logits = self.fc(feat)
        logits2 = self.fc(f)

        return logits, logits2


def resnet8(num_classes, pretrained=False, **kwargs):
    return ResNet(8, [16, 16, 32, 64], "basicblock", num_classes, pretrained, **kwargs)


def resnet14(num_classes, pretrained=False, **kwargs):
    return ResNet(14, [16, 16, 32, 64], "basicblock", num_classes, pretrained, **kwargs)


def resnet20(num_classes, pretrained=False, **kwargs):
    return ResNet(20, [16, 16, 32, 64], "basicblock", num_classes, pretrained, **kwargs)


def resnet32(num_classes, pretrained=False, **kwargs):
    return ResNet(32, [16, 16, 32, 64], "basicblock", num_classes, pretrained, **kwargs)


def resnet44(num_classes, pretrained=False, **kwargs):
    return ResNet(44, [16, 16, 32, 64], "basicblock", num_classes, pretrained, **kwargs)


def resnet56(num_classes, pretrained=False, **kwargs):
    return ResNet(56, [16, 16, 32, 64], "basicblock", num_classes, pretrained, **kwargs)


def resnet110(num_classes, pretrained=False, **kwargs):
    return ResNet(
        110, [16, 16, 32, 64], "basicblock", num_classes, pretrained, **kwargs
    )


def resnet8x4(num_classes, pretrained=False, **kwargs):
    return ResNet(
        8, [32, 64, 128, 256], "basicblock", num_classes, pretrained, **kwargs
    )


def resnet32x2(num_classes, pretrained=False, **kwargs):
    return ResNet(
        32, [16, 32, 64, 128], "basicblock", num_classes, pretrained, **kwargs
    )


def resnet32x4(num_classes, pretrained=False, **kwargs):
    return ResNet(
        32, [32, 64, 128, 256], "basicblock", num_classes, pretrained, **kwargs
    )
