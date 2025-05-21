import torch
import torch.nn as nn

class StructuredBranch(nn.Module):
    """结构化特征编码分支 (使用 MLP 和注意力机制)"""
    def __init__(self, input_dim, feature_dim=256, num_layers=3, hidden_dim_factor=2, num_attention_heads=4):
        """
        初始化结构化特征分支
        Args:
            input_dim (int): 输入特征维度（结构化特征数量）
            feature_dim (int): 输出特征维度
            num_layers (int): MLP 的层数
            hidden_dim_factor (int): 隐藏层维度倍数
            num_attention_heads (int): 多头自注意力头数
        """
        super(StructuredBranch, self).__init__()
        self.mlp = nn.Sequential()
        in_dim = input_dim

        # 构建 MLP 层
        for i in range(num_layers - 1):
            hidden_dim = feature_dim * hidden_dim_factor
            self.mlp.append(nn.Linear(in_dim, hidden_dim))
            self.mlp.append(nn.ReLU())
            self.mlp.append(nn.BatchNorm1d(hidden_dim))  # 添加批归一化
            in_dim = hidden_dim

        # 输出层
        self.mlp.append(nn.Linear(in_dim, feature_dim))
        self.relu_final = nn.ReLU()

        # 自注意力层
        self.attention = nn.MultiheadAttention(feature_dim, num_attention_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        前向传播
        Args:
            x (torch.Tensor): 输入结构化特征 (batch_size, input_dim)
        Returns:
            torch.Tensor: 编码后的特征 (batch_size, feature_dim)
        """
        mlp_output = self.relu_final(self.mlp(x))
        attn_input = mlp_output.unsqueeze(1)  # (batch_size, 1, feature_dim)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        attended_output = self.norm(attn_input + self.dropout(attn_output)).squeeze(1)
        return attended_output