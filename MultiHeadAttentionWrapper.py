import torch
import torch.nn as nn

from CausalAttention import CausalAttention
#一个实现多头注意力的封装类
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # nn.ModuleList: PyTorch的特殊列表,用于存储多个子模块
        # 会自动注册所有子模块的参数
        # 调用 model.to(device) 时会移动所有子模块
        # 保存/加载模型时会包含所有子模块
        # 列表推导式: 创建 num_heads 个独立的 CausalAttention 实例
        # 每个头都有自己独立的权重矩阵(W_query, W_key, W_value)
        # 每个头学习不同的注意力模式
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        results = [head(x) for head in self.heads]
        # 沿着最后一个维度(特征维度)拼接所有头的输出
        # 每个头输出形状: (batch_size, num_tokens, d_out)
        # 拼接后形状: (batch_size, num_tokens, num_heads * d_out)
        return torch.cat(results, dim=-1)
