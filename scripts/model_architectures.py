# scripts/model_architectures.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super().__init__()
        self.equalInOut = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride=1 if self.equalInOut else stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = conv3x3(out_planes, out_planes, stride=1)
        self.shortcut = (nn.Identity() if self.equalInOut and stride == 1
                         else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False))

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        if not self.equalInOut:
            x = self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        if self.equalInOut:
            x = self.shortcut(x)
        return x + out

class NetworkBlock(nn.Module):
    def __init__(self, n_layers, in_planes, out_planes, block, stride, dropout):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(block(in_planes if i==0 else out_planes,
                                out_planes, stride if i==0 else 1, dropout))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=100, dropout=0.3):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        ch = [16, 16*k, 32*k, 64*k]
        self.conv1  = conv3x3(3, ch[0], 1)
        self.block1 = NetworkBlock(n, ch[0], ch[1], BasicBlock, 1, dropout)
        self.block2 = NetworkBlock(n, ch[1], ch[2], BasicBlock, 2, dropout)
        self.block3 = NetworkBlock(n, ch[2], ch[3], BasicBlock, 2, dropout)
        self.bn = nn.BatchNorm2d(ch[3])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(ch[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.); nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        x = self.conv1(x); x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.relu(self.bn(x)); x = self.pool(x); x = x.flatten(1)
        return self.fc(x)

# 你原来的 SimpleCNN 保留（作业需要）
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(128*4*4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

def create_model(num_classes, device, dropout=0.3):
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes, dropout=dropout).to(device)
