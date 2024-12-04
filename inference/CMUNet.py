import torch
import torch.nn as nn
from core.src.network.msag import MSAG

# Residual 残差模块 : 跳跃连接
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn  # 要应用于输入的函数或模块。

    def forward(self, x):
        return self.fn(x) + x  # 将输入 (残差连接) 与函数输出相加。

# ConvMixerBlock : 通过深度卷积与点卷积组合进行局部特征提取和通道混合。
class ConvMixerBlock(nn.Module):
    def __init__(self, dim=1024, depth=7, k=7):
        super(ConvMixerBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # 深度卷积
                    nn.Conv2d(dim, dim, kernel_size=(k, k), groups=dim, padding=(k // 2, k // 2)),
                    nn.GELU(),  # 激活函数
                    nn.BatchNorm2d(dim)  # 归一化
                )),
                nn.Conv2d(dim, dim, kernel_size=(1, 1)),  # 点卷积（通道间信息混合）
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for i in range(depth)]  # 深度决定了模块堆叠的层数
        )

    def forward(self, x):
        x = self.block(x)
        return x

# conv_block 卷积块 : 标准的卷积块，由两次卷积操作、批归一化和 ReLU 激活函数组成
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),  # 归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 实现上采样操作，用于扩大特征图的空间分辨率
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 上采样，扩大特征图尺寸
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),  # 归一化
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# CMUNet 网络 : 基于 U-Net 结构，结合 ConvMixer 模块和多尺度特征提取模块。
class CMUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=1, l=7, k=7):
        """
        Args:
            img_ch: 输入通道数（如 RGB 图像为 3）。
            output_ch: 输出通道数（如二分类为 1）。
            l: ConvMixer 的层数。
            k: ConvMixer 的卷积核大小。
        """
        super(CMUNet, self).__init__()

        # 编码器
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样
        self.Conv1 = conv_block(ch_in=img_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)
        self.ConvMixer = ConvMixerBlock(dim=1024, depth=l, k=k)

        # 解码器
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = conv_block(ch_in=512 * 2, ch_out=512)
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = conv_block(ch_in=256 * 2, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=128 * 2, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64 * 2, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

        # 跳跃连接模块
        self.msag4 = MSAG(512)  # 512 通道的多尺度特征提取
        self.msag3 = MSAG(256)
        self.msag2 = MSAG(128)
        self.msag1 = MSAG(64)

    def forward(self, x):
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
        x5 = self.ConvMixer(x5)

        x4 = self.msag4(x4)
        x3 = self.msag3(x3)
        x2 = self.msag2(x2)
        x1 = self.msag1(x1)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d1 = self.Conv_1x1(d2)
        return d1
