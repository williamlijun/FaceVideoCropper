import torch
import torch.nn as nn
from .image_branch import ImageBranch
from .structured_branch import StructuredBranch
from .fusion import FusionModule
from .classifier import Classifier

class FaceNet(nn.Module):
    """通用人脸分类网络"""
    def __init__(self, config):
        """
        初始化网络
        Args:
            config (dict): 配置文件，包含 mode, num_classes, image_feature_dim, struct_feature_dim, fused_dim 等
        """
        super(FaceNet, self).__init__()
        self.mode = config['mode']
        self.num_classes = config['num_classes']
        self.image_feature_dim = config.get('image_feature_dim')
        self.struct_feature_dim = config.get('struct_feature_dim')
        self.fused_dim = config.get('fused_dim')
        
        # 初始化分支
        if self.mode in ['image_only', 'fusion']:
            self.image_branch = ImageBranch(feature_dim=self.image_feature_dim, backbone=config.get('backbone'))
        if self.mode in ['structured_only', 'fusion']:
            input_dim = len(config['structured_features'])
            self.structured_branch = StructuredBranch(input_dim=input_dim, feature_dim=self.struct_feature_dim)
        
        # 初始化融合模块
        if self.mode == 'fusion':
            self.fusion = FusionModule(
                image_dim=self.image_feature_dim,
                struct_dim=self.struct_feature_dim,
                fused_dim=self.fused_dim
            )
        
        # 初始化分类器
        input_dim = self.fused_dim if self.mode == 'fusion' else (self.image_feature_dim if self.mode == 'image_only' else self.struct_feature_dim)
        self.classifier = Classifier(input_dim=input_dim, num_classes=self.num_classes)
    
    def forward(self, image, structured):
        """
        前向传播
        Args:
            image (torch.Tensor): 输入图像 (batch_size, 3, H, W) 或 None
            structured (torch.Tensor): 输入结构化特征 (batch_size, input_dim) 或 None
        Returns:
            torch.Tensor: 分类 logits (batch_size, num_classes)
        """
        if self.mode == 'image_only':
            features = self.image_branch(image)
        elif self.mode == 'structured_only':
            features = self.structured_branch(structured)
        else:  # fusion
            image_features = self.image_branch(image)
            structured_features = self.structured_branch(structured)
            features = self.fusion(image_features, structured_features)
        
        logits = self.classifier(features)
        return logits