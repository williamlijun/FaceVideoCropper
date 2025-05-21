import torch
import torch.nn as nn

class Classifier(nn.Module):
    """分类模块"""
    def __init__(self, input_dim=512, num_classes=8):
        """
        初始化分类器
        Args:
            input_dim (int): 输入特征维度，默认为融合后的维度
            num_classes (int): 类别数量
        """
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入特征 (batch_size, input_dim)
        Returns:
            torch.Tensor: 分类 logits (batch_size, num_classes)
        """
        return self.fc(x)