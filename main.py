import torch
from CMUNet import CMUNet
from torchvision import transforms
from BUSI_dataset import BUSI_dataset_segment
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# 获取准确率
def get_pixel_accuracy(model, data_loader, device):
    '''
    获取准确率
    :param model: 模型
    :param data_loader: 数据加载器
    :param device: 设备
    :return: 准确率
    '''
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_output = torch.sigmoid(model(X))  # 获取概率值
            predicted = (y_output > 0.5).float()  # 使用0.5的阈值来判断类别（0 或 1）
            correct += (predicted == y).sum().float()  # 统计预测正确的像素数
            total += y.numel()  # 统计总像素数

    accuracy = correct / total  # 计算像素准确率
    return accuracy

lr = 1e-4
epochs = 15
batch_size = 32

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224 x 224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]) # 图像标准化
])

# 定义数据增强
transform_extend = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224 x 224
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # 图像标准化
])

# 定义数据预处理(label)
transform_label = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # 调整图像大小为 224 x 224
])

# 加载数据集
train_dataset = BUSI_dataset_segment(image_dir='BUSI/train', transform=transform, transform_label=transform_label)
# train_dataset_extend = BUSI_dataset_segment(image_dir='BUSI/train', transform=transform_extend, transform_label=transform_label)
# train_dataset += train_dataset_extend
test_dataset = BUSI_dataset_segment(image_dir='BUSI/test', transform=transform, transform_label=transform_label)

# 数据加载
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建 CMUNet 模型
model = CMUNet().to(device)

# 定义损失函数
criterion = nn.BCEWithLogitsLoss()

# 定义优化器
optimizer = optim.AdamW(
        params=model.parameters(),
        lr=lr,
)

train_acc_list = []  # 训练准确率列表
test_acc_list = []  # 测试准确率列表
for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_output = torch.sigmoid(model(X))
            loss = criterion(y_output.squeeze(1), y.squeeze(1).float())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Step: {step + 1}, Loss: {loss.item():.4f}")

        average_loss = epoch_loss / len(train_loader)
        train_acc = get_pixel_accuracy(model, train_loader, device).to('cpu')
        test_acc = get_pixel_accuracy(model, test_loader, device).to('cpu')
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"Epoch: {epoch}, Loss: {average_loss:.4f}, Train acc: {train_acc * 100:.2f}, Test acc: {test_acc * 100:.2f}")

# 绘制准确率曲线
x = np.arange(epochs)
plt.plot(x, train_acc_list, label='train', markevery=2)
plt.plot(x, test_acc_list, label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
