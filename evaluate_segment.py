from res_unet import Res_Unet
from torchvision import transforms
from PIL import Image
import cv2
from utils import *

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Res_Unet(output_channel=1, pretrained_model_path="pretrained_model/best_finetune_model.pth").to(device)
model.load_state_dict(torch.load("models/best_segment_model.pth"))
model.eval()

# 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]) # 图像标准化
])
transform_label = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224))  # 调整图像大小为 224 x 224
])

# 读取图片
img_path = "test_segment_images/benign (324).png"
img = Image.open(img_path).convert('RGB')
img = transform(img).unsqueeze(0).to(device)

img_mask = Image.open("test_segment_images/benign (324)_mask.png").convert('L')
img_mask = transform_label(img_mask).to(device)

# 评估模型
with torch.no_grad():
    pred = torch.sigmoid(model(img))
    predicted = (pred > 0.5).float()    # 使用0.5的阈值来判断类别（0 或 1）

    correct = (predicted == img_mask).sum().float()     # 统计预测正确的像素数
    one_num_pred = (predicted == 1).sum().float()   # 统计预测中的1的像素数
    one_num_mask = (img_mask == 1).sum().float()    # 统计标签中的1的像素数
    total = img_mask.numel()  # 统计总像素数

    PA = correct / total  # 计算像素准确率
    confusion_matrix = get_confusion_matrix(predicted.cpu().numpy(), img_mask.cpu().numpy(), 2)     # 获取混淆矩阵
    IoU = get_IoU(confusion_matrix)  # 获取交并比
    dice = get_dice_coefficient(predicted.cpu().numpy(), img_mask.cpu().numpy())  # 获取 Dice 系数
    precision, recall, F1 = get_precision_recall_F1(predicted.cpu().numpy(), img_mask.cpu().numpy())  # 获取精确率、召回率和 F1 值

    print(f"预测正确像素数：{int(correct.item())} ; 预测中的1的像素数：{one_num_pred.item()}")
    print(f"标签像素数：{total} ; 标签中的1的像素数：{one_num_mask.item()}")
    print(f"像素准确率：{PA.item() * 100:.2f} %")
    print(f"混淆矩阵：\n{confusion_matrix}")
    print(f"IoU：{IoU}")
    print(f"Dice 系数：{dice}")
    print(f"精确率：{precision}")
    print(f"召回率：{recall}")
    print(f"F1 值：{F1}")

# 标签和预测结果保存为图片进行对比
img_mask = img_mask.squeeze(0).squeeze(0).cpu().numpy()
img_mask = (img_mask * 255).astype(np.uint8)
cv2.imwrite("test_segment_images/benign (324)_pred.png", img_mask)

# predicted = predicted.squeeze(0).squeeze(0).cpu().numpy()
# predicted = (predicted * 255).astype(np.uint8)
# cv2.imwrite("predicted_segment_images/benign (1)_pred.png", predicted)
