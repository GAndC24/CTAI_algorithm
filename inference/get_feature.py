import inspect
from random import random

import cv2
import numpy as np
import pandas as pd
from numba import jit

np.set_printoptions(suppress=True)  # 输出时禁止科学表示法，直接输出小数值

column_all_c = ['良性肿瘤', '恶性肿瘤', '正常乳腺']

features_list = ['Benign', 'Malignant', 'Normal']


# 最后俩偏度 峰度


# 获取变量的名
def get_variable_name(variable):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is variable]

# 灰度梯度共生矩阵 (GLCM) 特征提取
# 计算灰度梯度共生矩阵 (GLCM)
def glcm(img_gray, ngrad=16, ngray=16):
    """
    计算灰度梯度共生矩阵 (GLCM)。

    Args:
        img_gray (numpy.ndarray): 输入灰度图像。
        ngrad (int): 梯度量化级数。
        ngray (int): 灰度量化级数。
    """
    # 计算梯度图像
    gsx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gsy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    grad = (gsx ** 2 + gsy ** 2) ** 0.5  # 梯度幅值
    grad = np.asarray(grad * (ngrad - 1) / grad.max(), dtype=np.int16)  # 梯度归一化
    gray = np.asarray(img_gray * (ngray - 1) / img_gray.max(), dtype=np.int16)  # 灰度归一化

    # 初始化 GLCM
    height, width = img_gray.shape
    gray_grad = np.zeros([ngray, ngrad])
    for i in range(height):
        for j in range(width):
            gray_value = gray[i][j]
            grad_value = grad[i][j]
            gray_grad[gray_value][grad_value] += 1
    gray_grad /= (height * width)  # 归一化
    get_glcm_features(gray_grad)

# 计算 GLCM 的纹理特征
def get_glcm_features(mat):
    """
    计算 GLCM 的纹理特征。
    包括小梯度优势、大梯度优势、灰度均值、梯度均值等。

    Args:
        mat (numpy.ndarray): 灰度梯度共生矩阵。
    """
    sum_mat = mat.sum()
    small_grads_dominance = big_grads_dominance = gray_asymmetry = grads_asymmetry = 0
    gray_variance = grads_variance = energy = gray_entropy = grads_entropy = entropy = 0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            energy += mat[i][j] ** 2
            if mat[i][j] > 0:
                gray_entropy -= mat[i][j] * np.log(mat[i][j])
    glgcm_features = [small_grads_dominance, big_grads_dominance, energy, gray_entropy]
    # 存储到全局特征字典
    for feature in glgcm_features:
        feature_name = get_variable_name(feature)[0]
        c_features[feature_name].append(np.round(feature, 4))

@jit
def get_gray_feature():
    # 灰度特征提取算法
    hist = cv2.calcHist([image_ROI_uint8[index]], [0], None, [256], [0, 256])
    # 假的 还没用灰度直方图

    c_features['mean'].append(np.mean(image_ROI[index]))
    c_features['std'].append(np.std(image_ROI[index]))

    s = pd.Series(image_ROI[index])
    c_features['piandu'].append(s.skew())
    c_features['fengdu'].append(s.kurt())

# 从分割掩码中提取形态特征
def get_geometry_feature():
    """
    从分割掩码中提取形态特征。
    包括区域面积、周长、质心、椭圆长短轴差等。
    """
    contours, _ = cv2.findContours(mask_array.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    tarea, tperimeter = [], []

    for c in contours:
        try:
            M = cv2.moments(c)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            c_features['focus_x'].append(cx)
            c_features['focus_y'].append(cy)
        except ZeroDivisionError:
            print("Division by zero in centroid calculation.")

        try:
            (x, y), (MA, ma), angle = cv2.fitEllipse(c)
            c_features['ellipse'].append(ma - MA)
        except:
            continue

        tarea.append(cv2.contourArea(c))
        tperimeter.append(cv2.arcLength(c, True))

    try:
        c_features['area'].append(max(tarea))
        c_features['perimeter'].append(round(max(tperimeter), 4))
    except ValueError:
        print("Area calculation error.")

# 提取肿瘤特征
def get_feature(image, mask):
    """
    提取肿瘤区域的纹理特征、形态特征等。

    Args:
        image (str): 图像文件路径。
        mask (str): 掩码文件路径。

    Returns:
        dict: 提取的特征字典。
    """
    global image_ROI, mask_array
    mask_array = cv2.imread(mask, 0)
    image_array = cv2.imread(image)

    index = np.nonzero(mask_array)
    if not index[0].any():
        return None

    # 提取肿瘤区域
    image_ROI = np.zeros_like(image_array)
    image_ROI[index] = image_array[index]

    # 提取特征
    get_geometry_feature()
    glcm(image_ROI, ngrad=15, ngray=15)
    return c_features

# 该代码实现了一个主函数 main，用于以下任务：
# 加载图像并进行预处理。
# 使用预训练的深度学习模型对图像进行分类推理。
# 提取分类结果并将其格式化为特征字典。
def main(pid):
    import os
    import sys
    import time
    import random
    import argparse

    import cv2
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torch.optim as optim

    #from core.tools.my_dataset import COVIDDataset
    #from core.resnet_uscl import ResNetUSCL
    from resnet_uscl import ResNetUSCL

    # 图像加载与预处理
    img = cv2.imread(pid)

    valid_transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # 归一化
    ])

    img_classi = valid_transform(img)

    # 模型加载
    pretrained = False
    selfsup = True
    net = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=pretrained)

    print(net)
    if pretrained:
        print('\nThe ImageNet pretrained parameters are loaded.')
    else:
        print('\nThe ImageNet pretrained parameters are not loaded.')

    # 加载预训练模型权重
    if selfsup:
        state_dict_path = "path/to/best_model.pth"
        state_dict = torch.load(state_dict_path)

        # 过滤权重：去掉多层感知器（MLP）和全连接层（fc）的参数
        new_dict = {k: state_dict[k] for k in list(state_dict.keys())
                    if not (k.startswith('l') | k.startswith('fc'))}

        model_dict = net.state_dict(new_dict)
        model_dict.update(new_dict)
        net.load_state_dict(model_dict, False)

    # 模型分类头重定义
    # add a classifier for linear evaluation
    num_ftrs = net.linear.in_features
    net.linear = nn.Linear(num_ftrs, 3)  # 修改线性层
    net.fc = nn.Linear(3, 3)  # 添加额外分类头

    # 加载微调后的模型
    state_dict_path = "path/to/best_finetune_model.pth"
    state_dict = torch.load(state_dict_path)
    net.load_state_dict(state_dict)
    net.eval()

    for name, param in net.named_parameters():
        print(name, '\t', 'requires_grad=', param.requires_grad)
    # 数据推理
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net.to(device)

    img_classi = img_classi.numpy()
    img_classi = np.expand_dims(img_classi, axis=0)  # 增加批次维度
    img_classi = torch.tensor(img_classi).to(device)

    outputs = net(img_classi)
    result = torch.sigmoid(outputs).data.cpu().numpy()

    # 分类结果保存
    global c_features
    c_features = {}
    for i in range(len(features_list)):
        c_features[features_list[i]] = [column_all_c[i], float(result[i])]

    return c_features

if __name__ == '__main__':
    main()
