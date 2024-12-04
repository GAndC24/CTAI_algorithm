import argparse

# 将字符串或数值转换为布尔值
def str2bool(v):
    """
    将字符串或数值转换为布尔值。

    Args:
        v (str or int): 输入的值，通常为命令行参数。

    Returns:
        bool: 转换后的布尔值。

    Raises:
        argparse.ArgumentTypeError: 当输入无法解析为布尔值时抛出错误。
    """
    if v.lower() in ['true', 1]:  # 允许的“真”值
        return True
    elif v.lower() in ['false', 0]:  # 允许的“假”值
        return False
    else:
        # 输入无效时抛出异常
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 计算模型参数数量
def count_params(model):
    """
    统计模型中可训练参数的数量。

    Args:
        model (torch.nn.Module): PyTorch 模型实例。

    Returns:
        int: 模型中可训练参数的总数。
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 用于计算和存储平均值及当前值
class AverageMeter(object):
    """用于计算和存储平均值及当前值"""

    def __init__(self):
        """
        初始化 AverageMeter 实例。
        """
        self.reset()

    def reset(self):
        """
        重置计数器，包括当前值、总和、计数和平均值。
        """
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 样本数

    def update(self, val, n=1):
        """
        更新计数器。

        Args:
            val (float): 新增的值。
            n (int): 值的权重（默认为 1，表示单个样本）。
        """
        self.val = val  # 更新当前值
        self.sum += val * n  # 增加总和
        self.count += n  # 增加计数
        self.avg = self.sum / self.count  # 计算新的平均值
