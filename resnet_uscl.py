import torch.nn as nn
import torchvision.models as models

# ResNetUSCL 模型 : 由 ResNet 特征提取器、投影头（projection head）和分类器（classifier）组成。
#              它主要用于无监督对比学习 (USCL) 和下游任务的分类
class ResNetUSCL(nn.Module):
    ''' The ResNet feature extractor + projection head + classifier for USCL '''

    def __init__(self, base_model, out_dim, pretrained=False):
        """
        初始化 ResNetUSCL 模型。

        Args:
            base_model (str): 基础 ResNet 模型名称（如 'resnet18', 'resnet50'）。
            out_dim (int): 投影头的输出维度。
            pretrained (bool): 是否加载 ImageNet 的预训练权重。
        """
        super(ResNetUSCL, self).__init__()

        # 定义支持的 ResNet 模型字典
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=pretrained),
                            "resnet50": models.resnet50(pretrained=pretrained)}

        # 根据是否加载预训练权重输出提示信息
        if pretrained:
            print('\nModel parameters loaded.\n')
        else:
            print('\nRandom initialize model parameters.\n')

        # 加载指定的 ResNet 模型
        resnet = self._get_basemodel(base_model)

        # 提取 ResNet 的全连接层输入特征数
        num_ftrs = resnet.fc.in_features

        # 特征提取部分：丢弃 ResNet 的最后全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # 投影头：将特征投影到指定的维度
        self.linear = nn.Linear(num_ftrs, out_dim)

        # 分类器：将投影后的特征用于分类
        num_classes = 3  # 假设分类任务有 3 类
        self.fc = nn.Linear(out_dim, num_classes)

    def _get_basemodel(self, model_name):
        """
        根据模型名称加载指定的 ResNet 模型。

        Args:
            model_name (str): 模型名称（'resnet18' 或 'resnet50'）。

        Returns:
            torch.nn.Module: 加载的 ResNet 模型。

        Raises:
            Exception: 如果输入的模型名称无效。
        """
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except KeyError:
            raise ValueError("Invalid model name. Choose 'resnet18' or 'resnet50'.")

    def forward(self, x):
        """
        定义前向传播。

        Args:
            x (torch.Tensor): 输入图像张量，形状为 [batch_size, channels, height, width]。

        Returns:
            torch.Tensor: 投影后的特征。
        """
        h = self.features(x)  # 提取特征
        h = h.squeeze()  # 去掉多余的维度

        x = self.linear(h)  # 投影到指定维度

        return x

