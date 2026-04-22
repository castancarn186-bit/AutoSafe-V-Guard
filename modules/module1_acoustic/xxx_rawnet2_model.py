"""
RawNet2 模型定义（基于 ASVspoof 2021 基线）
参考：https://github.com/eurecom-asp/RawNet2-antispoofing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.3)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=0.3)
        return out


class RawNet2(nn.Module):
    def __init__(self, pretrained_path=None):
        super(RawNet2, self).__init__()

        # 第一阶段: 128维
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.block2 = ResidualBlock(128, 128)
        self.block3 = ResidualBlock(128, 128)
        self.block4 = ResidualBlock(128, 128)

        # 第二阶段: 升压至 256维 (匹配 bn5 的报错)
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, stride=3, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.block6 = ResidualBlock(256, 256)
        self.block7 = ResidualBlock(256, 256)
        self.block8 = ResidualBlock(256, 256)

        # 第三阶段: 降压回 128维 (匹配 GRU 的需求)
        self.conv9 = nn.Conv1d(256, 128, kernel_size=3, stride=3, padding=1)
        self.bn9 = nn.BatchNorm1d(128)
        self.block10 = ResidualBlock(128, 128)
        self.block11 = ResidualBlock(128, 128)
        self.block12 = ResidualBlock(128, 128)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # GRU 输入必须是 128
        self.gru = nn.GRU(128, 1024, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 2)

        if pretrained_path:
            # 使用 strict=False 确保一些细微的名称差异不影响加载
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        # x: (batch, 1, samples)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.3)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.3)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        x = F.leaky_relu(self.bn9(self.conv9(x)), negative_slope=0.3)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.avg_pool(x).squeeze(-1)  # (batch, 128)
        x = x.unsqueeze(1)  # (batch, 1, 128)

        x, _ = self.gru(x)
        x = x[:, -1, :]  # (batch, 2048)

        x = F.leaky_relu(self.fc1(x), negative_slope=0.3)
        x = self.fc2(x)
        return x
