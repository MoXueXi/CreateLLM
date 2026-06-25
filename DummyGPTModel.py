import torch
import torch.nn as nn
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
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
    
# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5#变量eps是一个小常数(epsilon)，在归一化过程中会被加到方差上以防止除零错误
        #scale和shift是两个可训练的参数（与输入维度相同）​，如果在训练过程中发现调整它们可以改善模型的训练任务表现，
        # 那么大语言模型会自动进行调整。这使得模型能够学习适合其数据处理的最佳缩放和偏移。
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        #设置unbiased=False,这意味着在方差计算中，我们会使用样本数量n作为方差公式的除数。这种方法没有使用贝塞尔修正。
        # 贝塞尔修正通常在样本方差的估计中使用n-1作为分母，以调整偏差。
        var = x.var(dim = -1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
