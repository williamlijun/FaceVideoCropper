import torch
import torch.nn as nn

class FusionModule(nn.Module):
    """特征融合模块，结合双向交叉注意力和门控机制"""
    def __init__(self, image_dim=512, struct_dim=256, fused_dim=512, num_heads=8):
        """
        初始化融合模块
        Args:
            image_dim (int): 图像特征维度
            struct_dim (int): 结构化特征维度
            fused_dim (int): 融合后特征维度
            num_heads (int): 注意力头数
        """
        super(FusionModule, self).__init__()
        self.image_proj = nn.Linear(image_dim, fused_dim)
        self.struct_proj = nn.Linear(struct_dim, fused_dim)
        
        # 双向交叉注意力
        self.image_to_struct = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        self.struct_to_image = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        
        # 门控机制
        self.gate_fc = nn.Linear(fused_dim * 2, fused_dim)
        self.sigmoid = nn.Sigmoid()
        
        # 融合后处理
        self.fc = nn.Linear(fused_dim * 2, fused_dim)
        self.norm = nn.LayerNorm(fused_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, image_features, structured_features):
        """
        前向传播
        Args:
            image_features (torch.Tensor): 图像特征 (batch_size, image_dim)
            structured_features (torch.Tensor): 结构化特征 (batch_size, struct_dim)
        Returns:
            torch.Tensor: 融合特征 (batch_size, fused_dim)
        """
        # 投影到统一维度
        image_features = self.image_proj(image_features)  # (batch_size, fused_dim)
        structured_features = self.struct_proj(structured_features)  # (batch_size, fused_dim)
        
        # 调整形状以适应注意力机制
        image_seq = image_features.unsqueeze(1)  # (batch_size, 1, fused_dim)
        struct_seq = structured_features.unsqueeze(1)  # (batch_size, 1, fused_dim)
        
        # 双向交叉注意力
        img_attn, _ = self.image_to_struct(image_seq, struct_seq, struct_seq)  # 图像查询结构化
        struct_attn, _ = self.struct_to_image(struct_seq, image_seq, image_seq)  # 结构化查询图像
        
        # 残差连接
        img_attn = self.norm(image_seq + self.dropout(img_attn)).squeeze(1)  # (batch_size, fused_dim)
        struct_attn = self.norm(struct_seq + self.dropout(struct_attn)).squeeze(1)  # (batch_size, fused_dim)
        
        # 门控融合
        concat = torch.cat([img_attn, struct_attn], dim=-1)  # (batch_size, 2 * fused_dim)
        # gate = self.sigmoid(self.gate_fc(concat))  # (batch_size, fused_dim)
        # fused = gate * img_attn + (1 - gate) * struct_attn  # (batch_size, fused_dim)
        
        # 最终融合
        fused = self.fc(concat)  # 使用 concat 以保留更多信息
        fused = self.norm(fused)
        fused = self.relu(fused)
        fused = self.dropout(fused)
        return fused