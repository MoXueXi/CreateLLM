#import urllib.request
# url = ("https://raw.githubusercontent.com/rasbt/"
#        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
#        "the-verdict.txt")
# file_path = "the-verdict.txt"
# urllib.request.urlretrieve(url, file_path)

import re
from CausalAttention import CausalAttention
from GPTDatasetV1 import GPTDatasetV1
from MultiHeadAttention import MultiHeadAttention
from MultiHeadAttentionWrapper import MultiHeadAttentionWrapper
from NeuralNetwork import NeuralNetwork
from SimpleTokenizerV1 import SimpleTokenizerV1
from importlib.metadata import version
import tiktoken
import torch
from ToyDataset import ToyDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from SelfAttention_v1 import SelfAttention_v1

#1、文本预处理
# print(version("tiktoken"))
# 读取文件内容
with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_txet = file.read()
# raw_txet = "Hello, world. This is a -- test."
# 分割文本为单词
result = re.split(r'([,.:;?_!"()\']|--|\s)', raw_txet)
# 过滤空字符串
result = [item for item in result if item.strip()]
# print(len(result))
# print(result[:30])
# 去重并排序
all_words = sorted(set(result))
all_words.extend(["<|endoftext|>", "<|unk|>"])

# print(len(all_words))

vocab = {token: i for i, token in enumerate(all_words)}
# enumerate_list = enumerate(list(vocab.items())[-5:])
# for i, (token, id) in enumerate_list:
#     print(i, token, id)
tokenizer = SimpleTokenizerV1(vocab)
# test_text = "I HAD always thought Jack Gisburn rather a cheap genius -- though a good fellow enough --"
# ids = tokenizer.encode(test_text)
# test_text = tokenizer.decode(ids)
# print(ids)
# print(test_text)

# text = "Hello, do you like tea?"
# print(tokenizer.encode(text))

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
txet = " <|endoftext|> ".join((text1, text2))
txet = "Akwirw ier"
#2、BPE分词器
# print(tokenizer.decode(tokenizer.encode(text)))
#BPE分词器(我们将使用现有的Python开源库tiktoken，它基于Rust的源代码非常高效地实现了BPE算法)
# 使用 tiktoken 编码
tiktoken_obj = tiktoken.get_encoding("gpt2")
# ids = (tiktoken_obj.encode(txet, allowed_special={"<|endoftext|>"}))
# str = tiktoken_obj.decode(ids)
# print(str)
enc_text = tiktoken_obj.encode(raw_txet)
# print(len(enc_text))
enc_sample = enc_text[50:]
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1 : context_size+1]
# print(f"x: {x}")
# print(f"y: {y}")

# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0))

#3、创建一个小型示例数据集
X_train = torch.tensor([
    [-1.2, 3.1],
    [-0.9, 2.9],
    [-0.5, 2.6],
    [2.3, -1.1],
    [2.7, -1.5]
])
y_train = torch.tensor([0, 0, 0, 1, 1])

X_test = torch.tensor([
    [-0.8, 2.8],
    [2.6, -1.6],
])
y_test = torch.tensor([0, 1])

train_dataset = ToyDataset(X_train, y_train)
test_dataset = ToyDataset(X_test, y_test)
torch.manual_seed(123)
# 打乱数据集
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0, drop_last=True)

# for i, (X_batch, y_batch) in enumerate(train_dataloader):
#     print(f"Batch {i}: X={X_batch}, y={y_batch}")

#GPU上训练循环
#实例化一个神经网络模型
#输入特征维度为 2，输出类别数为 2（二分类任务）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNetwork(num_inputs = 2, num_outputs = 2)
model.to(device)

# 使用 随机梯度下降（SGD） 优化器
# 传入模型的所有可训练参数
# 学习率（lr）设置为 0.5
# optimizer = torch.optim.SGD(model.parameters(), lr=0.7)
# #训练总轮数为 3 轮
# num_epochs = 3
# for epoch in range(num_epochs):
#      # 设置模型为训练模式
#     model.train()
#     # 按批次加载数据（train_dataloader）
#     # features 是输入特征，labels 是对应的标签
#     for batch_idx, (features, labels) in enumerate(train_dataloader):
#         features = features.to(device)
#         labels = labels.to(device)
#         # 前向传播，计算模型输出
#         logits = model(features)
#         # 计算损失
#         loss = F.cross_entropy(logits, labels)
#         # 清除上一轮的梯度
#         optimizer.zero_grad()
#         # 反向传播，计算梯度
#         loss.backward()
#         # 更新模型参数
#         optimizer.step()
#         print(f"Epoch {epoch+1:03d}/{num_epochs:03d}, Batch {batch_idx:03d} / {len(train_dataloader):03d}, Loss: {loss:.2f}")
#     # 评估模型在测试集上的性能
#     model.eval()

#禁用梯度计算，避免计算梯度，减少内存占用和计算时间(适用于模型评估、推理等不需要反向传播的场景)
# with torch.no_grad():
#     #前向传播获取输出
#     outputs = model(X_train)
#     # print(outputs)
#     #设置打印格式，避免科学计数法
#     torch.set_printoptions(sci_mode=False)
#     #dim=1 表示在第二个维度（类别维度）上进行归一化
#     #输出 probas 是每个样本属于各个类别的概率（总和为 1）
#     probas = torch.softmax(outputs, dim=1)
#     # print(probas)
#     #使用 argmax 获取概率最大的类别索引
#     #dim=1 表示在类别维度上取最大值
#     #predictions 是每个样本的预测类别
#     predictions = torch.argmax(probas, dim=1)
#     # print(predictions)
#     #计算准确率
#     accuracy = torch.sum(predictions == y_train) / len(y_train)
#     # print(f"Accuracy: {accuracy:.2f}")

# #一个计算预测准确率的函数
# def calculate_accuracy(model, dataloader):
#     model.eval()
#     correct = 0
#     total = 0
#     for idx, (features, labels) in enumerate(dataloader):
#         with torch.no_grad():
#                 outputs = model(features)
#         predictions = torch.argmax(outputs, dim=1)
#         compare = predictions == labels
#         correct += torch.sum(compare)
#         total += len(compare)
#     accuracy = correct / total
#     return accuracy.item()

# accuracy = calculate_accuracy(model, test_dataloader)
# # print(f"Train Accuracy: {accuracy:.2f}")

#保存和加载模型
# torch.save(model.state_dict(), "model.pth")
# model = NeuralNetwork(num_inputs = 2, num_outputs = 2)
# model.load_state_dict(torch.load("model.pth"))

# print(torch.cuda.is_available())

# 测试在 GPU 上的加法操作
# tensor_1 = torch.tensor([1,2,3])
# tensor_1 = tensor_1.to("cuda")
# tensor_2 = torch.tensor([2,3,4])
# tensor_2 = tensor_2.to("cuda")
# print(tensor_1 + tensor_2)
def create_dataloader_v1(txt, batch_size = 4, max_length = 256, stride = 128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return dataloader

# dataloader = create_dataloader_v1(raw_txet, batch_size=6, max_length=4, stride=4, shuffle=False, drop_last=False)
# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)  

# 4、嵌入层
input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3
torch.manual_seed(123)
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#print(embedding_layer.weight)
#print(embedding_layer(torch.tensor([3])))
# print(embedding_layer(input_ids))

vocab_size = 50257
output_dim = 256
token_embeding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_txet, batch_size=8, max_length=max_length, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, tragets = next(data_iter)
# print(inputs)
# print(inputs.shape)
token_embeddings = token_embeding_layer(inputs)
# print(token_embeddings.shape)

context_length = max_length
position_embedding_layer = torch.nn.Embedding(context_length, output_dim)
position_embeddings = position_embedding_layer(torch.arange(context_length))
# print(position_embeddings.shape)

input_embeddings = token_embeddings + position_embeddings
# print(input_embeddings.shape)

# 5、编码注意力机制
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 计算注意力分数
query = inputs[1]
scores = torch.empty(len(inputs))
for i , v in enumerate(inputs):
     scores[i] = torch.dot(query, v)
# print(scores)

# tmp_scores = scores / scores.sum()
# print(tmp_scores)
# print(tmp_scores.sum())

# def soft_max_native(x):
#     return torch.exp(x) / torch.exp(x).sum()

# print(soft_max_native(scores))
# print(soft_max_native(scores).sum())

attention_weights = torch.softmax(scores, dim=0)
# print(attention_weights)
# print(attention_weights.sum())

#计算第二个输入的上下文向量
res = torch.zeros(query.shape)
for i , v in enumerate(inputs):
    res += v * attention_weights[i]
print(res)

attn_scores = torch.empty(6, 6)
# for i , v in enumerate(inputs):
#     for j , w in enumerate(inputs):
#         attn_scores[i, j] = torch.dot(v, w)

attn_scores = inputs @ inputs.T
# 
attn_weights = torch.softmax(attn_scores, dim=1)
# print(attn_weights)
# print(attn_weights.sum(dim=1))
all_contxt_vecs = attn_weights @ inputs
# print(all_contxt_vecs)

#实现带可训练权重的自注意力机制
x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
# 初始化可训练权重
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad = False) #  查询权重矩阵
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad = False) #  键权重矩阵
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad = False) #  值权重矩阵

query_2 = x_2 @ W_query #  查询向量
key_2 = x_2 @ W_key #  键向量
value_2 = x_2 @ W_value #  值向量
# print(query_2)

keys = inputs @ W_key #  键向量矩阵
values = inputs @ W_value #  值向量矩阵
# print(keys.shape)
# print(values.shape)

attn_score_22 = query_2.dot(keys[1])
# print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
# print(attn_scores_2)

d_k= keys.shape[-1]
#缩放点积注意力
atten_weight_2 = torch.softmax(attn_scores_2 / (d_k**0.5), dim = -1)
# print(atten_weight_2.sum(dim=-1))

atten_context = atten_weight_2 @ values
# print(atten_context)

selfAttention_v1 = SelfAttention_v1(d_in, d_out)
context_vec = selfAttention_v1(inputs)
# print(context_vec)

#因果注意力机制(也称为掩码注意力)
queries = selfAttention_v1.W_query(inputs)
keys = selfAttention_v1.W_key(inputs)
dk = keys.shape[-1]
attn_scores = queries @ keys.T
attn_scores_weight = torch.softmax(attn_scores / dk **0.5, dim=-1)
# print(attn_scores_weight)
length = attn_scores_weight.shape[0]
mask = torch.tril(torch.ones(length, length))#提取矩阵的下三角部分(包括对角线),其他位置填充为0
# print(mask)
mask = attn_scores_weight * mask
# print(mask)

#计算每行的和,用于后续归一化。
# dim=-1: 沿着最后一个维度(列)求和,即对每一行求和
# keepdim=True: 保持维度不变,结果仍然是二维矩阵(而不是变成一维向量)
mask_sum = mask.sum(dim=-1, keepdim=True)
mask = mask / mask_sum
# print(mask)

#用softmax函数的数学特性，以更少的步骤更高效地计算掩码后的注意力权重
context_length = attn_scores.shape[0]
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = torch.masked_fill(attn_scores, mask.bool(), -torch.inf)
# print(masked)

scores_weight = torch.softmax(masked / (dk**0.5), dim=-1)
# print(scores_weight)

#利用dropout掩码额外的注意力权重(dropout仅在训练期间使用，训练结束后会被取消)
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
#在对注意力权重矩阵应用50%的dropout率时，矩阵中有一半的元素会随机被置为0。
# 为了补偿减少的活跃元素，矩阵中剩余元素的值会按1/0.5=2的比例进行放大
# print(dropout(example))

# print(dropout(scores_weight))

#实现一个简化的因果注意力类  
# 将多个相同形状的张量沿着指定维度堆叠,创建一个新的维度。
# [inputs, inputs]: 要堆叠的张量列表(这里是两个相同的 inputs)
# dim=0: 在第0维(最外层)进行堆叠,创建批次维度
batch = torch.stack([inputs, inputs], dim=0)
# print(batch.shape)
context_length = batch.shape[1]
causal_attention = CausalAttention(d_in, d_out, context_length, 0.0)
context_vec = causal_attention(batch)
# print(context_vec.shape)

#一个实现多头注意力的封装类
context_length = batch.shape[1]
d_in = 3
d_out = 1
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vec = mha(batch)
# print(context_vec)
# print(context_vec.shape)

torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vec = mha(batch)
print(context_vec)
print(context_vec.shape)

