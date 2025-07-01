import torch
import torch.nn as nn
import torchvision.transforms as transforms
from resnet_uscl import ResNetUSCL
from BUSI_dataset import BUSI_dataset_classify
from utils import get_accuracy

# 图像预处理
test_transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])  # 归一化
])

# 加载数据集
test_dataset = BUSI_dataset_classify(image_dir='BUSI/test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

# 使用 cuda
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 模型加载
pretrained = False
selfsup = True
net = ResNetUSCL(base_model='resnet18', out_dim=3, pretrained=pretrained)

print(net)
if pretrained:
    print('\nThe ImageNet pretrained parameters are loaded.')
else:
    print('\nThe ImageNet pretrained parameters are not loaded.')

# # 加载预训练模型权重
# if selfsup:
#     state_dict_path = "pretrained_model/best_model.pth"
#     state_dict = torch.load(state_dict_path)
#
#     # 过滤权重：去掉多层感知器（MLP）和全连接层（fc）的参数
#     new_dict = {k: state_dict[k] for k in list(state_dict.keys())
#                 if not (k.startswith('l') | k.startswith('fc'))}
#
#     model_dict = net.state_dict(new_dict)
#     model_dict.update(new_dict)
#     net.load_state_dict(model_dict, False)
#     for name, param in net.named_parameters():
#         print(name, '\t', 'requires_grad=', param.requires_grad)

# # 模型分类头重定义
# num_ftrs = net.linear.in_features
# net.linear = nn.Linear(num_ftrs, 3)  # 修改线性层
# net.fc = nn.Linear(3, 3)  # 添加额外分类头

# 加载预训练模型权重
state_dict_path = "pretrained_model/best_finetune_model.pth"
state_dict = torch.load(state_dict_path)
net.load_state_dict(state_dict)
net = net.to(device)

for name, param in net.named_parameters():
    print(name, '\t', 'requires_grad=', param.requires_grad)

# 分类推理
accuracy = get_accuracy(net, test_loader, device)
print(f'Test accuracy: {accuracy * 100:.2f}%')



