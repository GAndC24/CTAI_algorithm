import os
import cv2

# 数据归一化
def data_in_one(inputdata):
    """
    对输入数据进行归一化处理，将其值域映射到 [0, 1]。

    Args:
        inputdata (numpy.ndarray): 输入数据（如图像或数组）。

    Returns:
        numpy.ndarray: 归一化后的数据。如果输入为空，直接返回输入。
    """
    if not inputdata.any():  # 检查数据是否为空
        return inputdata

    # 归一化公式：(x - min) / (max - min)
    inputdata = (inputdata - inputdata.min()) / (inputdata.max() - inputdata.min())
    return inputdata

# 路径预处理
def pre_process(data_path):
    """
    从给定路径中提取文件名（无扩展名）并返回路径和文件名。

    Args:
        data_path (str): 数据文件的完整路径。

    Returns:
        tuple: 包含原始路径和提取的文件名。
    """
    file_name = os.path.split(data_path)[1].split('.')[0]  # 提取文件名（无扩展名）
    return data_path, file_name

# 后处理
def last_process(file_name):
    """
    对分割结果进行后处理，在原始图像上绘制分割轮廓。

    Args:
        file_name (str): 文件名（不含扩展名）。
    """
    # 读取原始图像和分割掩码
    image = cv2.imread(f'./tmp/ct/{file_name}.png')  # 原始图像
    mask = cv2.imread(f'./tmp/mask/{file_name}_mask.png', 0)  # 分割掩码（灰度图）

    # 提取掩码中的轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 在原始图像上绘制轮廓
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # 绿色轮廓，线宽为 2 像素

    # 保存绘制结果
    cv2.imwrite('./tmp/draw/{}.png'.format(file_name), image)

