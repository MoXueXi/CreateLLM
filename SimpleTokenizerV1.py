import re

# 简单的分词器
# 支持：
# 1. 分割文本为单词
# 2. 过滤空字符串
# 3. 去重并排序
# 4. 编码文本为ID列表
# 5. 解码ID列表为文本
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: str for str, i in vocab.items()}
    
    # 编码文本为ID列表
    def encode(self, text):
        result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        result = [item.strip() for item in result if item.strip()]
        result = [item if item in self.str_to_int else "<|unk|>" for item in result]
        ids = [self.str_to_int[item] for item in result]
        return ids
    
    # 解码ID列表为文本
    def decode(self, ids):
        result = [self.int_to_str[i] for i in ids]
        text = " ".join(result)
        #正则表达式 \s+([,.?!"()\']) 匹配：一个或多个空格 + 指定标点符号
        # 替换为 \1（即只保留标点符号本身）
        # 效果："Hello . world" → "Hello. world"
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
