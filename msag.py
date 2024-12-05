import torch.nn as nn
import torch

# MSAG 模块（Multi-Scale Attention Gate）
class MSAG(nn.Module):
    """
    Multi-Scale Attention Gate (多尺度注意力门)
    """
    def __init__(self, channel):
        super(MSAG, self).__init__()
        self.channel = channel

        # 1x1 卷积，用于通道维度上的特征压缩或特征融合
        self.pointwiseConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=1, padding=0, bias=True),
            nn.BatchNorm2d(self.channel),  # 批归一化
        )

        # 3x3 标准卷积，用于提取局部特征
        self.ordinaryConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.BatchNorm2d(self.channel),  # 批归一化
        )

        # 3x3 膨胀卷积（dilation=2），用于扩大感受野
        self.dilationConv = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, padding=2, stride=1, dilation=2, bias=True),
            nn.BatchNorm2d(self.channel),  # 批归一化
        )

        # 投票卷积（融合多尺度特征并生成注意力图）
        self.voteConv = nn.Sequential(
            nn.Conv2d(self.channel * 3, self.channel, kernel_size=(1, 1)),  # 特征降维到原始通道数
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()  # 注意力权重映射到 [0, 1]
        )

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播过程：
        1. 输入特征图 `x` 分别通过点卷积、普通卷积和膨胀卷积提取多尺度特征。
        2. 将这些多尺度特征在通道维度上拼接，并通过 ReLU 激活。
        3. 使用投票卷积生成注意力权重，权重范围在 [0, 1]。
        4. 结合原始输入与加权后的输入特征，生成最终输出。
        """
        x1 = self.pointwiseConv(x)  # 点卷积特征
        x2 = self.ordinaryConv(x)  # 标准卷积特征
        x3 = self.dilationConv(x)  # 膨胀卷积特征

        # 拼接多尺度特征图
        _x = self.relu(torch.cat((x1, x2, x3), dim=1))  # 在通道维度 (dim=1) 上拼接

        # 生成注意力权重
        _x = self.voteConv(_x)

        # 加权特征与原始输入相结合
        x = x + x * _x  # 加上注意力加权后的输入，强调重要特征
        return x
