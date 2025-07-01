import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# 获取准确率(分类)
def get_accuracy(model, data_loader, device):
    '''
    获取模型在数据集上的准确率
    :param model: 模型
    :param data_loader: 数据加载器
    :param device: 设备
    :return: 准确率
    '''
    correct = 0
    total = 0
    model.eval()  # 评估模式
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            X_output = model(X)  # 前向传播
            _, predicted = torch.max(X_output.data, 0)  # 获取最大值对应的索引(预测类别)
            labels = torch.argmax(y, dim=1)  # 将one-hot编码的标签转换为类索引
            total += labels.size(0)  # 累加样本数
            correct += (predicted == labels).sum().item()  # 统计正确分类的样本数
    return correct / total

# 获取混淆矩阵
def get_confusion_matrix(pred, label, num_classes):
    '''
    获取混淆矩阵
    :param pred:  预测结果
    :param label:  标签
    :param num_classes:  类别数
    :return:  混淆矩阵
    '''
    pred = pred.flatten()  # 展平
    label = label.flatten()  # 展平
    # 计算混淆矩阵
    confusion_matrix = np.bincount(
        num_classes * label.astype(int) + pred.astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return confusion_matrix

# 获取 IoU
def get_IoU(confusion_matrix):
    '''
    获取交并比
    :param confusion_matrix:  混淆矩阵
    :return: 各类别 IoU
    '''
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    iou = intersection / (union + 1e-6)

    return iou

# 获取 precision, recall, F1
def get_precision_recall_F1(pred, label, smooth=1e-6):
    '''
    获取 precision, recall, F1
    :param pred:  预测结果
    :param label:  标签
    :param smooth:  平滑项
    :return: precision, recall, F1
    '''
    pred = pred.flatten()  # 展平
    label = label.flatten()  # 展平

    TP = np.sum(pred * label)
    FP = np.sum(pred * (1 - label))
    FN = np.sum((1 - pred) * label)

    precision = TP / (TP + FP + smooth)
    recall = TP / (TP + FN + smooth)
    F1 = 2 * precision * recall / (precision + recall + smooth)

    return precision, recall, F1

def evaluate_segment_model(model, device, img_path, mask_path):
    '''
    评估分割模型
    :param model:  模型
    :param device:  设备
    :param img_path:  图像路径
    :param mask_path:  标签路径
    '''
    model.eval()

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # 图像标准化
    ])
    transform_label = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))  # 调整图像大小为 224 x 224
    ])

    # 读取图片
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    img_mask = Image.open(mask_path).convert('L')
    img_mask = transform_label(img_mask).to(device)

    # 评估模型
    with torch.no_grad():
        pred = torch.sigmoid(model(img))
        predicted = (pred > 0.5).float()  # 使用0.5的阈值来判断类别（0 或 1）

        correct = (predicted == img_mask).sum().float()  # 统计预测正确的像素数
        total = img_mask.numel()  # 统计总像素数

        PA = correct / total  # 计算像素准确率
        confusion_matrix = get_confusion_matrix(predicted.cpu().numpy(), img_mask.cpu().numpy(), 2)  # 获取混淆矩阵
        IoU = get_IoU(confusion_matrix)  # 获取交并比
        precision, recall, F1 = get_precision_recall_F1(predicted.cpu().numpy(),
                                                        img_mask.cpu().numpy())  # 获取精确率、召回率和 F1 值

        print(f"PA：{PA.item() * 100:.2f} %")
        print(f"IoU：{IoU}")
        print(f"Precision：{precision}")
        print(f"Recall：{recall}")
        print(f"F1 - value：{F1}")

def segment_result_compare(model, device, img_path, mask_path):
    '''
    分割结果对比
    :param model:  模型
    :param device:  设备
    :param img_path:  图像路径
    :param mask_path:  标签路径
    '''
    model.eval()

    # 定义数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # 图像标准化
    ])
    transform_label = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224))  # 调整图像大小为 224 x 224
    ])

    # 读取图片
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)

    img_mask = Image.open(mask_path).convert('L')
    img_mask = transform_label(img_mask).to(device)

    # 评估模型
    with torch.no_grad():
        pred = torch.sigmoid(model(img))
        predicted = (pred > 0.5).float()  # 使用0.5的阈值来判断类别（0 或 1）

    # 原始图像、标签和预测结果进行对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原始图像
    img = img.squeeze(0).squeeze(0).cpu().numpy()
    axes[0].imshow(img)
    axes[0].set_title('Image')

    # 标签图像
    img_mask = img_mask.squeeze(0).squeeze(0).cpu().numpy()
    axes[1].imshow(img_mask, cmap='gray')
    axes[1].set_title('Label')

    # 预测结果
    pred = predicted.squeeze(0).squeeze(0).cpu().numpy()
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Predicted')

    plt.show()
