import torch
import torch.nn as nn
import torchvision.models as models

class ImageBranch(nn.Module):
    """轻量级图像特征提取分支，使用 MobileNetV3-Small"""
    def __init__(self, feature_dim=512, backbone='mobilenet_v3_small'):
        """
        初始化图像分支
        Args:
            feature_dim (int): 输出特征维度，默认为512
            backbone (str): 骨干网络类型，支持 'mobilenet_v3_small', 'efficientnet_b0', 'squeezenet'
        """
        super(ImageBranch, self).__init__()
        # 支持多种轻量级骨干网络
        if backbone == 'mobilenet_v3_small':
            self.backbone = models.mobilenet_v3_small(pretrained=True)
            default_dim = 1024  # MobileNetV3-Small 的默认输出维度
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=True)
            default_dim = 1280  # EfficientNet-B0 的默认输出维度
        elif backbone == 'squeezenet':
            self.backbone = models.squeezenet1_0(pretrained=True)
            default_dim = 512  # SqueezeNet 的默认输出维度
        else:
            raise ValueError(f"不支持的骨干网络: {backbone}")
        
        # 替换全连接层，适配指定特征维度
        if backbone == 'mobilenet_v3_small':
            self.backbone.classifier[-1] = nn.Linear(default_dim, feature_dim)
        elif backbone == 'efficientnet_b0':
            self.backbone.classifier[-1] = nn.Linear(default_dim, feature_dim)
        elif backbone == 'squeezenet':
            self.backbone.classifier[1] = nn.Conv2d(512, feature_dim, kernel_size=1)
            self.backbone.classifier.add_module('flatten', nn.Flatten())
        
        self.bn = nn.BatchNorm1d(feature_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入图像 (batch_size, 3, H, W)
        Returns:
            torch.Tensor: 提取的特征 (batch_size, feature_dim)
        """
        x = self.backbone(x)  # (batch_size, feature_dim)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


# import torch
# import torch.nn as nn
# import torchvision.models as models

# class ImageBranch(nn.Module):
#     """图像特征提取分支"""
#     def __init__(self, feature_dim=512, backbone='resnet18'):
#         """
#         初始化图像分支
#         Args:
#             feature_dim (int): 输出特征维度
#             backbone (str): 骨干网络类型，支持 'resnet18', 'resnet50', 'resnet101'
#         """
#         super(ImageBranch, self).__init__()
#         # 支持多种骨干网络，默认为 ResNet18
#         if backbone == 'resnet18':
#             self.resnet = models.resnet18(pretrained=False) # 若要使用预训练的参数，则设置 pretrained=True
#             default_dim = 512
#         elif backbone == 'resnet50':
#             self.resnet = models.resnet50(pretrained=False)
#             default_dim = 2048
#         elif backbone == 'resnet101':
#             self.resnet = models.resnet101(pretrained=False)
#             default_dim = 2048
#         else:
#             raise ValueError(f"不支持的骨干网络: {backbone}")
        
#         # 替换全连接层，仅当 feature_dim 与默认维度不同时降维
#         self.resnet.fc = nn.Linear(default_dim, feature_dim)
#         self.bn = nn.BatchNorm1d(feature_dim)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.3)
    
#     def forward(self, x):
#         """
#         前向传播
#         Args:
#             x (torch.Tensor): 输入图像 (batch_size, 3, H, W)
#         Returns:
#             torch.Tensor: 提取的特征 (batch_size, feature_dim)
#         """
#         x = self.resnet(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         return x