import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """WideResNet basic residual block (3×3 convs, optional downsample)"""
    def __init__(self, in_planes, out_planes, stride=1, drop_rate=0.0):
        super().__init__()
        self.equalInOut = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.drop = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()
        # shortcut if dimensions change
        self.shortcut = nn.Identity() if self.equalInOut else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.drop(out)
        out = self.conv2(out)
        return out + self.shortcut(x)


class NetworkBlock(nn.Module):
    """Stack multiple BasicBlocks to form one stage"""
    def __init__(self, num_layers, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        layers = []
        for i in range(num_layers):
            s = stride if i == 0 else 1
            layers.append(BasicBlock(in_planes if i == 0 else out_planes, out_planes, s, drop_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    """
    WideResNet-28-10 for CIFAR datasets
    depth=28, widen_factor=10 => 4*n + 4 = 28 => n=4 blocks per stage
    """
    def __init__(self, num_classes=100, depth=28, widen_factor=10, drop_rate=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth should be 6n+4, e.g., 28"
        n = (depth - 4) // 6
        k = widen_factor
        nChannels = [16, 16*k, 32*k, 64*k]

        # 1) initial conv
        self.conv1 = nn.Conv2d(3, nChannels[0], 3, stride=1, padding=1, bias=False)
        # 2) 3 groups of residual blocks
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], stride=1, drop_rate=drop_rate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], stride=2, drop_rate=drop_rate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], stride=2, drop_rate=drop_rate)
        # 3) BN + ReLU + global average pool + FC
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def create_model(num_classes, device):
    """Create and move model to device"""
    model = WideResNet(num_classes=num_classes, depth=28, widen_factor=10, drop_rate=0.3)
    model = model.to(device)
    return model
