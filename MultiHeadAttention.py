import torch
import torch.nn as nn
#一个高效的多头注意力类
#核心设计思想:
# 与 MultiHeadAttentionWrapper 不同,这个实现:
# 不使用 nn.ModuleList 创建多个独立头
# 使用 单个线性层 + 张量重塑实现多头
# 更高效: 利用GPU并行计算,减少内存开销
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        #确保输出维度能被头数整除,保证每个头能均匀分配特征维度。    
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        # 每个线性层输出 d_out 维(而不是 d_out // num_heads)
        # 后续通过 view 操作拆分成多个头
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))


    # 输入: (2, 3, 4)
    # ↓ W_query
    # (2, 3, 6)
    # ↓ view
    # (2, 3, 2, 3)  # 拆分成2个头
    # ↓ transpose(1, 2)
    # (2, 2, 3, 3)  # 按头组织,便于批量计算
    # ↓ @ (矩阵乘法)
    # (2, 2, 3, 3)  # 每个头独立计算注意力
    # ↓ transpose(1, 2)
    # (2, 3, 2, 3)  # 恢复token维度在前
    # ↓ view
    # (2, 3, 6)     # 合并多头
    # ↓ out_proj
    # (2, 3, 6)     # 最终输出
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        values = self.W_value(x)
        queries = self.W_query(x)
        #拆分成多个头
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        #按头组织,便于批量计算
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        queries = queries.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = torch.masked_fill(attn_scores, mask_bool, -torch.inf)
        attn_weights = torch.softmax(attn_scores  / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2)# 恢复token维度在前
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)# 合并多头
        # 这个输出投影层并不是必需的​，但它在许多大语言模型架构中被广泛使用，这就是我们在这里添加它以保持完整性的原因
        context_vec = self.out_proj(context_vec)#最终线性变换,输出形状
        return context_vec
