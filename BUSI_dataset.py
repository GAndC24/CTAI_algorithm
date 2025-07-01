from torch.utils.data import Dataset
from PIL import Image
import os
import torch

# BUSI 数据集类（用于分类）
class BUSI_dataset_classify(Dataset):
    def __init__(self, image_dir, transform = None):
        '''
        初始化
        :param image_dir:  图像文件夹路径
        :param transform:  对图像的预处理操作（如缩放、裁剪等）
        '''
        self.image_dir = image_dir
        self.transform = transform

        # 列出文件夹内的所有文件，并过滤出所有图像文件（png, jpg, jpeg）
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)
                            if img.endswith(('png', 'jpg', 'jpeg')) and "mask" not in img]

        # 标签编码
        self.labels_encoding = {
            "benign": 0, "malignant": 1, "normal": 2
        }

    def __len__(self):
        '''
        返回数据集中样本数
        :return: 数据集中样本数量
        '''
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''
        获取指定索引处的图像和标签
        :param idx: 指定的索引
        :return: image(图像), label_one_hot(one-hot编码的标签)
        '''
        img_path = self.image_paths[idx]        # 获取图像路径
        image = Image.open(img_path).convert("RGB")         # 打开图像并转换为RGB格式

        # 应用数据增强
        if self.transform:
            image = self.transform(image)

        # 提取图像文件名的前缀作为标签
        label_str = os.path.basename(img_path).split('(')[0]
        label_str = label_str.strip()

        # 将字符串标签转换为整数标签
        label = self.labels_encoding[label_str]

        # 将标签转换为one-hot编码
        label_one_hot = torch.zeros(len(self.labels_encoding))
        label_one_hot[label] = 1

        return image, label_one_hot

# BUSI 数据集类（用于分割）
class BUSI_dataset_segment(Dataset):
    def __init__(self, image_dir, transform=None, transform_label=None):
        '''
        初始化

        :param image_dir:  图像文件夹路径
        :param transform:  对图像的预处理操作
        '''
        self.image_dir = image_dir
        self.transform = transform
        self.transform_label = transform_label

        # 列出文件夹内的所有图像文件（png, jpg, jpeg）
        self.image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)
                            if img.endswith(('png', 'jpg', 'jpeg')) and "mask" not in img]

        # 列出文件夹内的所有掩码文件（png, jpg, jpeg）
        self.mask_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir)
                           if img.endswith(('png', 'jpg', 'jpeg')) and "mask" in img]

    def __len__(self):
        '''
        返回数据集中样本数
        :return: 数据集中样本数量
        '''
        return len(self.image_paths)

    def __getitem__(self, idx):
        '''
        获取指定索引处的图像和掩码
        :param idx: 指定的索引
        :return: image(图像), mask(掩码)
        '''
        img_path = self.image_paths[idx]  # 获取图像路径
        mask_path = self.mask_paths[idx]  # 获取掩码路径

        image = Image.open(img_path).convert("RGB")  # 打开图像并转换为RGB格式
        mask = Image.open(mask_path).convert("L")  # 打开掩码并转换为灰度图

        # 应用数据增强
        if self.transform:
            image = self.transform(image)
            mask = self.transform_label(mask)

        return image, mask

