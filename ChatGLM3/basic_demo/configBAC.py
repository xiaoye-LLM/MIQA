#  configBAC.py 定义模型参数
import os

import torch

# =====添加配置项 拆分训练集和数据集===== #
TRAIN_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', R'BAC\motorData\train.txt')
TEST_SAMPLE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', R'BAC\motorData\test.txt')
all_text_PATH =os.path.join(os.path.dirname(__file__), '..', '..', R'BAC\motorData\all.txt')
# ======词表路径 和 标签表路径===== #
VOCAB_PATH = os.path.join(os.path.dirname(__file__), '..', '..',r'BAC\motorData\vocabe.txt')
LABEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..',r'BAC\motorData\tag.txt')
# ==========weight_decay：权重衰减（L2 正则化）系数，用于防止过拟合。weight_decay=1e-5 表示每次更新参数时会对参数施加一个 L2 正则化。
weight_decay=1e-4
# =======优化器每隔100 学习率调整0.5====== #
step_size = 30
gamma = 0.3
# =======当验证指标（如验证损失）不再改善时10，降低学习率0.1====== #
patience = 10
factor = 0.3
# ======= utils ======== #
WORD_UNK = '<UNK>'  # 没有见过的统称为UNK
WORD_PAD = '<PAD>'  # 不等长时填充空白
WORD_PAD_ID = 0
WORD_UNK_ID = 1
LABEL_O_ID = 0
base_len = 30
# ======= model ======= #
num_heads = 2  # 多头自注意力机制 头数
linear_SIZE = 256  #
batch_size = 64  # 80   64
num_layers = 2  # bilstm 层数
VOCAB_SIZE = 2300  # 词表的长度
EMBEDDING_DIM = 300  # 转换的词向量的维度
HIDDEN_SIZE = 256  # LSTM输出的隐层大小#600 700
EPOCH = 20000
LR = 1e-4
TARGET_SIZE = 5  # 经过全连接层输出的向量维度


# ======model预测=====#

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

