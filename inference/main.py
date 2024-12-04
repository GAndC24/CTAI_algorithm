from . import get_feature
import os.path
import numpy as np
import cv2
import torch
from torchvision import transforms


def c_main(path, model):
    """
    图像处理主函数，完成图像预处理、推理、结果保存和特征提取。

    Args:
        path (str): 输入图像的路径。
        model (torch.nn.Module): 预训练的深度学习模型。

    Returns:
        tuple: 包含处理后的文件名和图像特征信息。
    """
    # 图像读取与保存
    img = cv2.imread(path)  # 读取输入图像
    file_name = os.path.split(path)[1].split('.')[0]  # 提取文件名（不含扩展名）
    print(file_name)
    # 保存原始图像到指定路径，未压缩
    cv2.imwrite(
        f'C:/Users/26421/Desktop/PaddleX-Flask-VUE-demo-master/PaddleX-Flask-VUE-demo-master/CTAI_flask/tmp/image/{file_name}.png',
        img,
        (cv2.IMWRITE_PNG_COMPRESSION, 0)
    )

    # 图像预处理
    trans_totensor = transforms.ToTensor()  # 转换为张量
    trans_resize = transforms.Resize((256, 256))  # 调整大小
    trans_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    trans_compose = transforms.Compose([trans_totensor, trans_resize, trans_norm])  # 组合预处理步骤
    img2 = trans_compose(img)  # 对图像应用预处理
    img2 = img2.numpy()  # 转换为 NumPy 数组
    img2 = img2.astype('float32') / 255  # 归一化到 [0, 1]

    # 模型推理
    img3 = np.expand_dims(img2, axis=0)  # 增加批次维度
    img3 = torch.tensor(img3)  # 转换为 PyTorch 张量
    print(img3.shape)
    img4 = img3.cuda()  # 将张量加载到 GPU
    output = model(img4)  # 模型推理
    print(output)
    output1 = torch.sigmoid(output).data.cpu().numpy()[0][0]  # 获取输出并转换为 NumPy 数组
    print(output1)
    output2 = output1 * 255  # 将概率值映射到 [0, 255]

    # 推理结果保存
    cv2.imwrite(
        f'C:/Users/26421/Desktop/PaddleX-Flask-VUE-demo-master/PaddleX-Flask-VUE-demo-master/CTAI_flask/tmp/draw/{file_name}.png',
        output2
    )  # 保存概率图
    im_color = cv2.applyColorMap(output2.astype(np.uint8), cv2.COLORMAP_JET)  # 应用伪彩色映射
    cv2.imwrite(
        f'C:/Users/26421/Desktop/PaddleX-Flask-VUE-demo-master/PaddleX-Flask-VUE-demo-master/CTAI_flask/tmp/draw2/{file_name}.png',
        im_color
    )  # 保存伪彩色图

    # 特征提取
    image_info = get_feature.main(path)

    # 返回处理后的文件名（带 .png 后缀）和特征信息字典。
    return file_name + '.png', image_info

if __name__ == '__main__':
    pass

