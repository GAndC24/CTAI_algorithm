import os
import cv2
import numpy as np
import torch
import torch.utils.data

# 数据集类
class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        初始化 Dataset 类

        Args:
            img_ids (list): 图像 ID 列表，用于定位图像文件。
            img_dir (str): 图像文件夹路径。
            mask_dir (str): 掩码文件夹路径。
            img_ext (str): 图像文件扩展名（如 '.jpg', '.png'）。
            mask_ext (str): 掩码文件扩展名（如 '.png'）。
            num_classes (int): 掩码类别数。
            transform (callable, optional): 数据增强函数，用于图像和掩码的同时增强。
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform  # 可选的数据增强操作


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        """
        获取单个数据样本

        Args:
            idx (int): 样本索引。

        Returns:
            tuple: 包含以下内容：
                - img (ndarray): 预处理后的图像数据，形状为 (C, H, W)。
                - mask (ndarray): 预处理后的掩码数据，形状为 (num_classes, H, W)。
                - meta (dict): 元数据信息，包含图像 ID。
        """
        img_id = self.img_ids[idx]  # 根据索引获取图像 ID

        # 读取图像
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        # 初始化掩码列表
        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(
                os.path.join(self.mask_dir, str(i), img_id + "_mask" + self.mask_ext),
                cv2.IMREAD_GRAYSCALE  # 以灰度图模式读取
            )[..., None])  # 添加一个通道维度
        mask = np.dstack(mask)  # 将多个通道的掩码按深度堆叠，形成 (H, W, num_classes)

        # 如果定义了 transform，则同时增强图像和掩码
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        # 图像归一化并调整维度顺序为 (C, H, W)
        img = img.astype('float32') / 255  # 归一化到 [0, 1]
        img = img.transpose(2, 0, 1)  # 从 (H, W, C) 转换为 (C, H, W)

        # 掩码归一化并调整维度顺序为 (num_classes, H, W)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)  # 从 (H, W, num_classes) 转换为 (num_classes, H, W)

        return img, mask, {'img_id': img_id}  # 返回图像、掩码和元数据信息
