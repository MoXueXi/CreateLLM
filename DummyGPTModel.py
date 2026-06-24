import torch
import torch.nn as nn
class DummyGPTModel(nn.Module):
    GPT_CONFIG_124M = {
    "vocab_size": 50257,     # 词汇表大小
    "context_length": 1024,  # 上下文长度
    "emb_dim": 768,          # 嵌入维度
    "n_heads": 12,           # 注意力头的数量
    "n_layers": 12,          # 层数
    "drop_rate": 0.1,        # dropout率
    "qkv_bias": False        # 查询-键-值偏置
    }
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        #Token嵌入,将ID转换为向量
        tok_embs = self.tok_emb(in_idx)
        #生成位置编码,使用 torch.arange 创建位置索引序列
        pos_embs = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        #将Token嵌入和位置嵌入相加,融合内容和位置信息
        x = tok_embs + pos_embs
        #应用Dropout正则化
        x = self.drop_emb(x)
        #通过所有Transformer块,每个块包含注意力机制和前馈神经网络
        x = self.trf_blocks(x)
        #应用层归一化
        x = self.final_norm(x)
        # 输出头将隐藏状态转换为词表上的logits
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x
