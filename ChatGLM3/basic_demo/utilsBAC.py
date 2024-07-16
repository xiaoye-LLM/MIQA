# 定义dataset数据集和等长切分文本 utilsBAC.py
from collections import defaultdict

import torch
from torch.utils import data

import pandas as pd

from configBAC import VOCAB_PATH, LABEL_PATH, base_len, TRAIN_SAMPLE_PATH, TEST_SAMPLE_PATH, \
    WORD_UNK, WORD_PAD_ID, LABEL_O_ID, WORD_UNK_ID, DEVICE, batch_size

# 把文字转换成数字
# 加载词表和标签表

def get_vocab():
    df = pd.read_csv(VOCAB_PATH, names=['word', 'id'])
    return list(df['word']), dict(df.values)


def get_label():
    df = pd.read_csv(LABEL_PATH, names=['label', 'id'])
    return list(df['label']), dict(df.values)



class Dataset(data.Dataset):  # base_len（句子长度） 表示一个句子取几个字
    def __init__(self, type='train', base_len=base_len):
        super().__init__()
        self.base_len = base_len
        sample_path = TRAIN_SAMPLE_PATH if type == 'train' else TEST_SAMPLE_PATH

        self.df = pd.read_csv(sample_path, names=['word', 'label'])
        _, self.word2id = get_vocab()
        _, self.label2id = get_label()
        self.get_points()
        # print(self.points)

    # 计算分割点
    def get_points(self):
        self.points = [0]
        i = 0
        while True:
            if i + self.base_len >= len(self.df):
                self.points.append(len(self.df))
                break
            #  一句话正好有为O的点！
            if self.df.loc[i + self.base_len, 'label'] == 'O':
                i += self.base_len
                self.points.append(i)
            else:
                i += 1

    #  point是一个一个点[0,50,150,190]

    #  返回句子长度，是point的个数减一
    def __len__(self):
        return len(self.points) - 1

    #
    #  读取的每一段进行汉字转id，标签转id 方便做向量化
    def __getitem__(self, index):
        #  self.df是训练集
        df = self.df[self.points[index]:self.points[index + 1]]  # 得到的df是汉字，标签
        # print(df)

        word_unk_id = self.word2id[WORD_UNK]
        label_o_id = self.label2id['O']
        input = [self.word2id.get(w, word_unk_id) for w in df['word']]
        target = [self.label2id.get(l, label_o_id) for l in df['label']]
        raw_text = df['word'].tolist()
        # print(raw_text)
        return input, target  # 返回的是每个句子的id，并对应的标签的id


'''
将批次中的样本按照句子的长度从大到小进行排序，以便后续处理。
找到批次中最长的句子的长度，作为填充其他句子时所需的长度。创建三个空列表 input、target 和 mask，用于存储处理后的数据。
遍历批次中的每个样本，执行以下操作：
计算当前样本需要填充的长度 pad_len，即最长句子长度减去当前句子长度。
将当前样本的句子内容（item[0]）进行填充，使用 WORD_PAD_ID 值进行填充。
将当前样本的标签内容（item[1]）进行填充，使用 LABEL_O_ID 值进行填充。
创建一个掩码（mask），长度与当前句子相同，句子部分填充为1，填充部分填充为0。
将处理后的数据添加到相应的列表中。返回处理后的数据作为张量（tensor），分别为输入（input）、目标（target）和掩码（mask）。
目标张量（target tensor）是用来表示标签信息的张量。在序列标注任务中，每个输入序列都有对应的标签序列，目标张量存储了每个位置上的真实标签。它的形状通常与输入张量相同，并且每个位置上的值表示该位置的真实标签的索引或编码。

掩码张量（mask tensor）是用来指示输入序列中哪些位置是有效的，哪些位置是填充的。在不等长序列的处理中，为了进行批处理，需要对输入进行填充，使得所有序列具有相同的长度。掩码张量通过在有效位置上设置为1，填充位置上设置为0，来标记输入序列的有效部分。它的形状与输入张量相同，每个位置上的值表示该位置是否是填充位置。
'''


def collate_fn(batch):
    #  print("batch.type: `",type(batch))
    # 后面的x:x[0]代表字典的键（key）给sort排序，x:x[1]代表字典的值（values）给sort排序，reverse=true表示降序，reverse=false表示逆序。
    batch.sort(key=lambda x: len(x[0]), reverse=True)  # 给所有句子按照从大到小排序

    max_len = len(batch[0][0])  # 得到最大句子长度
    # print(batch[0])
    input = []
    target = []
    mask = []
    # print(max_len)
    for item in batch:
        # print(max_len)   #每个item对应的都是一个句子的id和标签的id
        pad_len = max_len - len(item[0])  # 填充长度
        input.append(item[0] + [WORD_PAD_ID] * pad_len)
        target.append(item[1] + [LABEL_O_ID] * pad_len)  # 在config.py中定义的 具体查看output中生成的txt文件
        mask.append([1] * len(item[0]) + [0] * pad_len)  # 有句子的地方全都填1  填充的填0

    return torch.tensor(input), torch.tensor(target), torch.tensor(mask).bool()



def extract(label, text):
    res = []
    extracted_entities = defaultdict(list)
    i = 0

    while i < len(label):
        if label[i].startswith('B-'):
            entity_type = label[i][2:]
            start = i
            i += 1
            entity_tokens = [text[start]]
            while i < len(label) and label[i].startswith('I-'):
                entity_tokens.append(text[i])
                i += 1
            entity = ''.join(entity_tokens)
            if entity not in extracted_entities[entity_type]:
                extracted_entities[entity_type].append(entity)
                res.append({entity_type: entity})
        else:
            i += 1

    return res


def prepare_input(question, word2id):
    input_ids = torch.tensor([[word2id.get(w, WORD_UNK_ID) for w in question]], dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor([[1] * len(question)], dtype=torch.bool, device=DEVICE)
    return input_ids, attention_mask



def extract_bio_entities(y_pred, id2label, text):
    entities = []
    current_entity = None
    current_label = None

    for idx, label_id in enumerate(y_pred[0]):
        label = id2label[label_id]
        if label.startswith("B-"):
            if current_entity is not None:
                entities.append((current_entity, current_label))
            current_label = label.split("-")[1]
            current_entity = text[idx]
        elif label.startswith("I-") and current_entity is not None:
            current_entity += text[idx]
        else:
            if current_entity is not None:
                entities.append((current_entity, current_label))
                current_entity = None
                current_label = None

    if current_entity is not None:
        entities.append((current_entity, current_label))

    return entities
# def report(y_true, y_pred):
#     return classification_report(y_true, y_pred)


if __name__ == '__main__':
    # 定义Datase类
    dataset = Dataset()
    loader = data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)  # 每次取100个句子

    print(iter(loader).__next__())  # 这里可能会报错
    # id2label,label2id = get_label()
    #  print(word2id)
