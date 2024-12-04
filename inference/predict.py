import numpy as np
import cv2

rate = 0.5

# 使用模型对输入数据进行预测，并将结果保存为掩码图像。
def predict(dataset, model):
    """
    使用模型对输入数据进行预测，并将结果保存为掩码图像。

    Args:
        dataset (tuple): 包含图像路径和文件名的元组。
            - dataset[0] (str): 图像路径。
            - dataset[1] (str): 图像文件名（不带扩展名）。
        model (callable): 预测模型，接受图像路径作为输入，返回预测结果。

    Global Vars:
        img_y (numpy.ndarray): 模型的预测结果（经过处理的掩码）。

    Returns:
        None
    """
    global img_y
    # 图像路径处理
    x = dataset[0].replace('\\', '/')  # 将路径中的反斜杠替换为斜杠（兼容 Windows 和 Linux）
    file_name = dataset[1]  # 提取文件名（无扩展名）
    print(x)
    print(file_name)

    # 模型推理
    img_y = model(x)  # 使用模型对图像进行预测
    img_y = img_y * 255  # 将预测结果的值从 [0, 1] 映射到 [0, 255]
    img_y = img_y.astype(np.int)  # 将结果转换为整数类型

    cv2.imwrite(f'./tmp/mask/{file_name}_mask.png', img_y,
                (cv2.IMWRITE_PNG_COMPRESSION, 0))

