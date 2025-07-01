import torch
import torch.nn as nn
from resnet_uscl import ResNetUSCL


# conv_block 卷积块 : 标准的卷积块，由两次卷积操作、批归一化和 ReLU 激活函数组成
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),  # 归一化
            nn.ReLU(inplace=True),  # 激活函数
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

# 实现上采样操作，用于扩大特征图的空间分辨率
class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 上采样，扩大特征图尺寸
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),  # 归一化
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class Res_Unet(nn.Module):
    def __init__(self, output_channel, pretrained_model_path=None):
        '''

        初始化模型

        :param pretrained_model_path: 预训练模型路径
        :param output_channel: 输出通道数
        '''
        super(Res_Unet, self).__init__()

        # Encoder : ResNetUSCL

        # 加载 ResNetUSCL 作为 encoder
        self.encoder = ResNetUSCL(base_model='resnet18', out_dim=3)

        # 加载预训练模型权重
        pretrained_weights = torch.load(pretrained_model_path)
        self.encoder.load_state_dict(pretrained_weights)

        # 去除投影头和分类器
        self.encoder.linear = nn.Identity()
        self.encoder.fc = nn.Identity()

        # 提取各层用于跳跃连接

        self.encoder_layers = list(self.encoder.children())
        # layer0 : (conv1, bn1, relu), output_channel=64
        self.encoder_layer0 = nn.Sequential(*self.encoder_layers[0][:3])
        # layer1 : maxpool
        self.encoder_layer1 = self.encoder_layers[0][3]
        # layer2 : (BasicBlock1(conv1, bn1, relu, conv2, bn2), BasicBlock2(conv1, bn1, relu, conv2, bn2)), output_channel=64
        self.encoder_layer2 = self.encoder_layers[0][4]
        # layer3 : (BasicBlock1(conv1, bn1, relu, conv2, bn2, downsample(conv1, bn1)), BasicBlock2(conv1, bn1, relu, conv2, bn2)), output_channel=128
        self.encoder_layer3 = self.encoder_layers[0][5]
        # layer4 : (BasicBlock1(conv1, bn1, relu, conv2, bn2, downsample(conv1, bn1)), BasicBlock2(conv1, bn1, relu, conv2, bn2)), output_channel=256
        self.encoder_layer4 = self.encoder_layers[0][6]
        # layer5 : (BasicBlock1(conv1, bn1, relu, conv2, bn2, downsample(conv1, bn1)), BasicBlock2(conv1, bn1, relu, conv2, bn2)), output_channel=512
        self.encoder_layer5 = self.encoder_layers[0][7]

        # Decoder : UNet

        # self.Up_conv4 = conv_block(ch_in=512 * 2, ch_out=512)
        # self.Up3 = up_conv(ch_in=512, ch_out=256)
        # self.Up_conv3 = conv_block(ch_in=256 * 2, ch_out=256)
        # self.Up2 = up_conv(ch_in=256, ch_out=128)
        # self.Up_conv2 = conv_block(ch_in=128 * 2, ch_out=128)
        # self.Up1 = up_conv(ch_in=128, ch_out=64)
        # self.Up_conv1 = conv_block(ch_in=64 * 2, ch_out=64)
        # self.Conv_1x1 = nn.Conv2d(64, output_channel, kernel_size=1, stride=1, padding=0)

        self.Up_conv4 = conv_block(ch_in=256 * 2, ch_out=256)
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = conv_block(ch_in=128 * 2, ch_out=128)
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = conv_block(ch_in=64 * 2, ch_out=64)
        self.Up1 = up_conv(ch_in=64, ch_out=64)
        self.Up_conv1 = conv_block(ch_in=64 * 2, ch_out=64)
        self.Up0 = up_conv(ch_in=64, ch_out=64)
        self.Conv_1x1 = nn.Conv2d(64, output_channel, kernel_size=1, stride=1, padding=0)

    # def forward(self, x):
    #     # encoder
    #     # x(224, 224, 3)
    #     x0 = self.encoder_layer0(x)     # x0(112, 112, 64)
    #     x1 = self.encoder_layer1(x0)        # x1(56, 56, 64)
    #     x2 = self.encoder_layer2(x1)        # x2(56, 56, 64)
    #     x3 = self.encoder_layer3(x2)        # x3(28, 28, 128)
    #     x4 = self.encoder_layer4(x3)        # x4(14, 14, 256)
    #     x5 = self.encoder_layer5(x4)        # x5(7, 7, 512)
    #
    #     # decoder
    #     d1 = x5
    #     d1 = torch.cat((d1, x5), dim=1)
    #     d1 = self.Up_conv4(d1)
    #
    #     d2 = self.Up3(d1)
    #     d2 = torch.cat((d2, x4), dim=1)
    #     d2 = self.Up_conv3(d2)
    #
    #     d3 = self.Up2(d2)
    #     d3 = torch.cat((d3, x3), dim=1)
    #     d3 = self.Up_conv2(d3)
    #
    #     d4 = self.Up1(d3)
    #     d4 = torch.cat((d4, x2), dim=1)
    #     d4 = self.Up_conv1(d4)
    #
    #     d5 = self.Conv_1x1(d4)
    #
    #     return d5

    def forward(self, x):
        # encoder
        # x(224, 224, 3)
        x0 = self.encoder_layer0(x)     # x0(112, 112, 64)
        x1 = self.encoder_layer1(x0)        # x1(56, 56, 64)
        x2 = self.encoder_layer2(x1)        # x2(56, 56, 64)
        x3 = self.encoder_layer3(x2)        # x3(28, 28, 128)
        x4 = self.encoder_layer4(x3)        # x4(14, 14, 256)

        # decoder
        d1 = x4
        d1 = torch.cat((d1, x4), dim=1)
        d1 = self.Up_conv4(d1)

        d2 = self.Up3(d1)
        d2 = torch.cat((d2, x3), dim=1)
        d2 = self.Up_conv3(d2)

        d3 = self.Up2(d2)
        d3 = torch.cat((d3, x2), dim=1)
        d3 = self.Up_conv2(d3)

        d4 = self.Up1(d3)
        d4 = torch.cat((d4, x0), dim=1)
        d4 = self.Up_conv1(d4)

        d5 = self.Up0(d4)
        output = self.Conv_1x1(d5)
        # output = self.Conv_1x1(d4)


        return output

# test
if __name__ == "__main__":
    model = Res_Unet(output_channel=2, pretrained_model_path="pretrained_model/best_finetune_model.pth")
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output.shape)  # 应输出 [1, num_classes, 224, 224]














