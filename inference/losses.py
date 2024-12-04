import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BCEDiceLoss']

class BCEDiceLoss(nn.Module):
    def __init__(self):
        """
        初始化 BCEDiceLoss 类。
        """
        super().__init__()

    def forward(self, input, target):
        """
        前向传播，计算 BCE 和 Dice 损失的加权和。

        Args:
            input (torch.Tensor): 模型输出，形状为 (N, C, H, W) 或 (N, H, W)，未经过 sigmoid 激活。
            target (torch.Tensor): 目标标签，形状与 input 一致。

        Returns:
            torch.Tensor: 最终的混合损失值。
        """
        # 二值交叉熵（Binary Cross Entropy, BCE）
        bce = F.binary_cross_entropy_with_logits(input, target)
        # Dice 损失计算
        smooth = 1e-5  # 防止分母为零的小常数
        input = torch.sigmoid(input)  # 将 logits 转换为概率值
        num = target.size(0)  # 批量大小
        input = input.view(num, -1)  # 展平为 (batch_size, total_pixels)
        target = target.view(num, -1)  # 同样展平
        intersection = (input * target)  # 计算预测与目标的交集（逐元素相乘）
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num  # 计算平均 Dice 损失，并取其补

        # 最终损失为 BCE 和 Dice 损失的加权和，BCE 权重为 0.5，Dice 权重为 1。
        return 0.5 * bce + dice

