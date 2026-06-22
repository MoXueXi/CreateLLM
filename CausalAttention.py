import torch
import torch.nn as nn

#实现一个简化的因果注意力类
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.context_length = context_length
        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 注册一个缓冲区(buffer),这是PyTorch模块的特殊属性:
        # 不是模型参数: 不会被梯度下降更新(不需要训练)
        # 随模型移动: 当调用 model.to(device) 时,会自动跟随模型移动到CPU或GPU
        # 随模型保存: 使用 torch.save() 和 torch.load() 时会被保存和加载
        # 通常用于: 固定不变的张量,如掩码、位置编码等
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores = torch.masked_fill(attn_scores, self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec

