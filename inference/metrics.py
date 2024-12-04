import numpy as np
import torch
import torch.nn.functional as F

# 计算 accuracy
def get_accuracy(SR, GT, threshold=0.5):
    """
    计算准确率 (Accuracy)。

    Args:
        SR (torch.Tensor): 模型预测输出 (Softmax 或 Sigmoid 输出)。
        GT (torch.Tensor): 真实标签。
        threshold (float): 阈值，确定二值化分割点。

    Returns:
        float: 准确率值。
    """
    SR = SR > threshold  # 将预测输出二值化
    GT = GT == torch.max(GT)  # 将标签二值化（假设标签为多类别 one-hot）
    corr = torch.sum(SR == GT)  # 计算预测正确的像素数
    tensor_size = SR.numel()  # 总像素数
    acc = float(corr) / float(tensor_size)  # 准确率公式
    return acc

# 计算 sensitivity(recall)
def get_sensitivity(SR, GT, threshold=0.5):
    """
    计算敏感度 (Sensitivity, Recall)。

    Args:
        SR (torch.Tensor): 模型预测输出。
        GT (torch.Tensor): 真实标签。
        threshold (float): 二值化阈值。

    Returns:
        float: 敏感度值。
    """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2  # 真正例
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2  # 假负例
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)  # 防止除零
    return SE

# 计算 specificity
def get_specificity(SR, GT, threshold=0.5):
    """
    计算特异性 (Specificity)。

    Args:
        SR (torch.Tensor): 模型预测输出。
        GT (torch.Tensor): 真实标签。
        threshold (float): 二值化阈值。

    Returns:
        float: 特异性值。
    """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2  # 真负例
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2  # 假正例
    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    return SP

# 计算 precision
def get_precision(SR, GT, threshold=0.5):
    """
    计算精确率 (Precision)。

    Args:
        SR (torch.Tensor): 模型预测输出。
        GT (torch.Tensor): 真实标签。
        threshold (float): 二值化阈值。

    Returns:
        float: 精确率值。
    """
    SR = SR > threshold
    GT = GT == torch.max(GT)
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
    return PC

# 计算 IoU 和其他评估指标
def iou_score(output, target):
    """
    计算 IoU 和其他评估指标。

    Args:
        output (torch.Tensor): 模型输出。
        target (torch.Tensor): 真实标签。

    Returns:
        tuple: 包含 IoU、Dice、Sensitivity、Precision、F1、Specificity、Accuracy 的值。
    """
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)  # IoU 计算公式
    dice = (2 * iou) / (iou + 1)

    output_ = torch.tensor(output_)
    target_ = torch.tensor(target_)
    SE = get_sensitivity(output_, target_, threshold=0.5)
    PC = get_precision(output_, target_, threshold=0.5)
    SP = get_specificity(output_, target_, threshold=0.5)
    ACC = get_accuracy(output_, target_, threshold=0.5)
    F1 = 2 * SE * PC / (SE + PC + 1e-6)  # F1 分数公式

    return iou, dice, SE, PC, F1, SP, ACC

# 计算 Dice 系数
def dice_coef(output, target):
    """
    计算 Dice 系数。

    Args:
        output (torch.Tensor): 模型输出。
        target (torch.Tensor): 真实标签。

    Returns:
        float: Dice 系数值。
    """
    smooth = 1e-5

    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)



