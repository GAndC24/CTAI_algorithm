import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from BUSI_dataset import BUSI_dataset_segment
import time
import numpy as np
import matplotlib.pyplot as plt

class Trainer():
    def __init__(self, model, lr, warmup_proportion, weight_decay, batch_size):
        '''
        初始化训练器

        :param lr: 学习率
        :param warmup_proportion: 学习率预热比例
        :param weight_decay: 学习率衰减系数
        :param batch_size: 批次大小
        '''
        self.lr = lr
        self.warmup_proportion = warmup_proportion
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.train_mini_epochs = 10
        self.epochs = 15
        self.num_warmup_epochs = int(self.warmup_proportion * self.epochs)  # 预热轮数

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
        mini_trian_dataset = BUSI_dataset_segment(image_dir='BUSI/train_mini', transform=transform, transform_label=transform_label)
        train_dataset = BUSI_dataset_segment(image_dir='BUSI/train', transform=transform, transform_label=transform_label)
        train_dataset_extend = BUSI_dataset_segment(image_dir='BUSI/train', transform=transform_extend, transform_label=transform_label)
        train_dataset += train_dataset_extend
        validation_dataset = BUSI_dataset_segment(image_dir='BUSI/validation', transform=transform, transform_label=transform_label)
        test_dataset = BUSI_dataset_segment(image_dir='BUSI/test', transform=transform, transform_label=transform_label)

        # 数据加载
        self.mini_train_loader = DataLoader(dataset=mini_trian_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        self.validation_loader = DataLoader(dataset=validation_dataset, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)

        # 使用 cuda
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 初始化模型
        self.model = model.to(self.device)

        # 定义损失函数
        self.criterion = nn.BCEWithLogitsLoss()

        # 定义优化器
        self.optimizer = optim.AdamW(
            params=self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # 定义学习率预热调度器
        self.lr_warmup_scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self.lr_lambda
        )

        # 定义学习率衰减调度器
        self.lr_decay_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=30,
            gamma=self.weight_decay
        )

    # 学习率调整函数(预热)
    def lr_lambda(self, current_epoch):
        '''
        学习率调整函数(预热)
        :param current_epoch: 当前训练轮数
        :return:
        '''
        # 初期线性预热，从 0 开始到学习率
        if current_epoch < self.num_warmup_epochs:
            return float(current_epoch) / float(max(1, self.num_warmup_epochs))
        else:
            return 1        # 预热结束后，返回学习率不变

    # 训练(超参数优化)
    def train_HP_optim(self, index):
        '''
        训练
        :param index: 种群个体序号
        :return: 最终准确率，最终验证时间
        '''
        print(
            f"Index: {index + 1}\n"
            f"lr : {self.lr}\n"
            f"weight_decay : {self.weight_decay}\n"
            f"warmup_proportion : {self.warmup_proportion}\n"
            f"batch_size : {self.batch_size}")
        for epoch in range(1, self.train_mini_epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for step, (X, y) in enumerate(self.mini_train_loader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_output = torch.sigmoid(self.model(X))
                loss = self.criterion(y_output.squeeze(1), y.squeeze(1).float())
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.lr_warmup_scheduler.step()
                self.lr_decay_scheduler.step()
                # print(f"Epoch: {epoch}, Step: {step + 1}, Loss: {loss.item():.4f}")

            average_loss = epoch_loss / len(self.mini_train_loader)
            train_acc = self.get_pixel_accuracy(self.model, self.mini_train_loader, self.device)
            validation_acc = self.get_pixel_accuracy(self.model, self.validation_loader, self.device)
            print(f"Epoch: {epoch}, Loss: {average_loss:.4f}, Train acc: {train_acc * 100:.2f}, Validation acc: {validation_acc * 100:.2f}")

        final_accuracy = self.get_pixel_accuracy(self.model, self.validation_loader, self.device)     # 最终准确率
        final_verification_time = self.get_verification_time(self.model, self.validation_loader, self.device)       # 最终验证时间

        # 将结果写入到文件中进行记录
        with open("training_log.txt", "a") as f:
            f.write(f"\nIndex: {index + 1}\n")
            f.write(f"lr: {self.lr}, warmup_proportion: {self.warmup_proportion}, weight_decay: {self.weight_decay}, batch_size: {self.batch_size}\n")
            f.write(f"Final Accuracy: {final_accuracy:.4f}, Verification Time: {final_verification_time:.4f}s\n")

        return final_accuracy, final_verification_time

    # 获取准确率
    def get_pixel_accuracy(self, model, data_loader, device):
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

    # 获取验证时间
    def get_verification_time(self, model, data_loader, device):
        '''
        获取验证时间
        :param model: 模型
        :param data_loader: 数据加载器
        :param device: 设备
        :return: 验证时间
        '''
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                _ = model(X)
                break  # 只测量一个批次的时间
        end_time = time.time()
        verification_time = end_time - start_time
        return verification_time

    # 训练
    def train(self):
        '''
        模型训练
        '''
        print(
            f"lr : {self.lr}\n"
            f"weight_decay : {self.weight_decay}\n"
            f"warmup_proportion : {self.warmup_proportion}\n"
            f"batch_size : {self.batch_size}")
        train_acc_list = []  # 训练准确率列表
        test_acc_list = []  # 测试准确率列表
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0

            for step, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_output = torch.sigmoid(self.model(X))
                loss = self.criterion(y_output.squeeze(1), y.squeeze(1).float())
                epoch_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.lr_warmup_scheduler.step()
                self.lr_decay_scheduler.step()
                # print(f"Epoch: {epoch}, Step: {step + 1}, Loss: {loss.item():.4f}")

            average_loss = epoch_loss / len(self.train_loader)
            train_acc = self.get_pixel_accuracy(self.model, self.train_loader, self.device).to('cpu')
            test_acc = self.get_pixel_accuracy(self.model, self.test_loader, self.device).to('cpu')
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(f"Epoch: {epoch}, Loss: {average_loss:.4f}, Train acc: {train_acc * 100:.2f}, Test acc: {test_acc * 100:.2f}")

        final_accuracy = self.get_pixel_accuracy(self.model, self.test_loader, self.device)  # 最终准确率
        final_verification_time = self.get_verification_time(self.model, self.test_loader,self.device)  # 最终验证时间

        print(f"final_accuracy : {final_accuracy}\nfinal_verification_time : {final_verification_time}")

        # 绘制准确率曲线
        x = np.arange(self.epochs)
        plt.plot(x, train_acc_list, label='train', markevery=2)
        plt.plot(x, test_acc_list, label='test', markevery=2)
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

    def save_model(self):
        '''
        保存模型
        '''
        torch.save(self.model.state_dict(), "models/best_segment_model.pth")
        file_path = "models/best_segment_model.pth"
        print(f"Model parameters saved to {file_path}")



