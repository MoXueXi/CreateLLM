import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    # 功能解析：
    # 初始化容器：创建两个空列表 input_ids 和 target_ids，分别存储输入序列和目标序列
    # 文本编码：使用 tokenizer.encode(txt) 将原始文本转换为 token id 序列
    # 滑动窗口切分（核心逻辑）：
    # 从序列起始位置开始，以 stride 为步长滑动
    # 每次截取长度为 max_length 的片段作为 input_chunk
    # target_chunk 是 input_chunk 向右偏移 1 位的序列（实现自回归预测）
    # 转换张量：将每个 chunk 转换为 PyTorch 张量，存入对应列表
    # 示例说明：
    # 假设 ids = [1,2,3,4,5,6,7]，max_length=3，stride=2：

    # 第1轮：input_chunk=[1,2,3]，target_chunk=[2,3,4]
    # 第2轮：input_chunk=[3,4,5]，target_chunk=[4,5,6]
    # 这是标准的**因果语言建模（CLM）**数据预处理方式，让模型学习根据前面的 token 预测下一个 token。
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        ids = tokenizer.encode(txt)
        for i in range(0, len(ids) - max_length, stride):
            input_chunk = ids[i:i+max_length]
            target_chunk = ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]