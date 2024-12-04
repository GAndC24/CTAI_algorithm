import os
import random
import pickle

import torchvision.datasets
from PIL import Image
from torch.utils.data import Dataset

random.seed(1)

# COVID 数据集类
class COVIDDataset(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):
        """
        初始化 COVIDDataset 类，用于加载 COVID 图像数据。

        Args:
            root_dir (str): 数据集的根目录。
            label_dir (str): 当前类别的文件夹名。
            transform (callable, optional): 图像增强或变换函数。
        """
        self.root_dir = root_dir  # 数据集根目录
        self.label_dir = label_dir  # 当前类别的文件夹名（如 "benign", "malignant"）
        self.path = os.path.join(self.root_dir, self.label_dir)  # 完整路径
        self.img_path = os.listdir(self.path)  # 当前类别下所有图像文件的列表
        self.transform = transform  # 数据变换（如归一化、随机裁剪等）
        self.label_name = {"benign": 0, "malignant": 1, "normal": 2}  # 类别映射

    def __getitem__(self, index):
        """
        根据索引返回单个数据样本，包括图像及其标签。

        Args:
            index (int): 样本索引。

        Returns:
            tuple: 包含图像数据和对应标签。
        """
        img_name = self.img_path[index]  # 根据索引获取图像文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 图像完整路径
        img = Image.open(img_item_path).convert('RGB')  # 加载图像并转换为 RGB 格式
        label = self.label_dir  # 当前目录名即为标签
        if self.transform is not None:  # 如果定义了数据增强，则应用到图像
            img = self.transform(img)
        labell = self.label_name[label]  # 将标签转换为数值形式
        return img, labell  # 返回图像和数值标签

    def __len__(self):
        return len(self.img_path)

# 导入了 PyTorch 自带的 CIFAR10 数据集，可能用于调试或参考，但未实际与 COVIDDataset 集成。
r = torchvision.datasets.CIFAR10