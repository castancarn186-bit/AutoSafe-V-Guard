"""
RawNet2 模型定义（基于 ASVspoof 2021 基线）
参考：https://github.com/eurecom-asp/RawNet2-antispoofing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """残差块，用于特征提取"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=1)
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
    """
    RawNet2 模型
    输入： (batch, 1, samples)  原始波形
    输出： (batch, 2)            logits（二分类）
    """
    def __init__(self, pretrained_path=None, num_classes=2, input_length=64600):
        super().__init__()
        self.input_length = input_length

        # 第一层卷积：处理原始波形
        self.conv1 = nn.Conv1d(1, 128, kernel_size=3, stride=3, padding=0)
        self.bn1 = nn.BatchNorm1d(128)

        # 残差块组
        self.block2 = ResidualBlock(128, 128, kernel_size=3, stride=1)
        self.block3 = ResidualBlock(128, 128, kernel_size=3, stride=1)
        self.block4 = ResidualBlock(128, 128, kernel_size=3, stride=1)

        # 下采样层
        self.conv5 = nn.Conv1d(128, 256, kernel_size=3, stride=3, padding=0)
        self.bn5 = nn.BatchNorm1d(256)

        self.block6 = ResidualBlock(256, 256, kernel_size=3, stride=1)
        self.block7 = ResidualBlock(256, 256, kernel_size=3, stride=1)
        self.block8 = ResidualBlock(256, 256, kernel_size=3, stride=1)

        self.conv9 = nn.Conv1d(256, 512, kernel_size=3, stride=3, padding=0)
        self.bn9 = nn.BatchNorm1d(512)

        self.block10 = ResidualBlock(512, 512, kernel_size=3, stride=1)
        self.block11 = ResidualBlock(512, 512, kernel_size=3, stride=1)
        self.block12 = ResidualBlock(512, 512, kernel_size=3, stride=1)

        # 统计池化
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.std_pool = nn.AdaptiveMaxPool1d(1)  # 或用标准差池化，此处简化

        # GRU 层
        self.gru = nn.GRU(512, 1024, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc1 = nn.Linear(1024 * 2, 1024)  # 双向 GRU 拼接后为 2048
        self.fc2 = nn.Linear(1024, num_classes)

        # 初始化权重
        self._init_weights()

        if pretrained_path:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

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

        # 统计池化：拼接均值和标准差
        mean = self.avg_pool(x).squeeze(-1)  # (batch, 512)
        std = self.std_pool(x).squeeze(-1)   # (batch, 512)
        stats = torch.cat((mean, std), dim=1)  # (batch, 1024)

        # GRU 处理（需要序列维度）
        # 将 x 转为序列：x shape (batch, channels, time) -> (batch, time, channels)
        x_seq = x.transpose(1, 2)  # (batch, time, 512)
        gru_out, _ = self.gru(x_seq)  # (batch, time, 2048)
        gru_out = gru_out[:, -1, :]    # 取最后时间步输出 (batch, 2048)

        # 融合统计特征和 GRU 输出
        combined = torch.cat((stats, gru_out), dim=1)  # (batch, 1024+2048=3072)

        # 全连接层
        out = F.leaky_relu(self.fc1(combined), negative_slope=0.3)
        logits = self.fc2(out)
        return logits