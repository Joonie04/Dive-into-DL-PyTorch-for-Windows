# 14.9 用于预训练BERT的数据集
import os
import random
import torch
import collections
from pathlib import Path


def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元
    
    参数:
        lines: 文本行列表
        token: 词元类型，'word' 表示单词，'char' 表示字符
    
    返回:
        词元列表的列表
    """
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


def count_corpus(tokens):
    """统计词元频率
    
    参数:
        tokens: 词元列表
    
    返回:
        词元频率计数器
    """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)


class Vocab:
    """文本词表类
    
    词表用于将词元映射到索引，以及将索引映射回词元。
    这是自然语言处理中的基础工具。
    
    参数:
        tokens: 词元列表
        min_freq: 最小词频，低于此频率的词元将被忽略
        reserved_tokens: 保留的词元列表，这些词元将始终包含在词表中
    """
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """初始化词表
        
        参数:
            tokens: 词元列表
            min_freq: 最小词频
            reserved_tokens: 保留的词元列表
        """
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self._unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        """返回词表大小
        
        返回:
            词表中的词元数量
        """
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        """获取词元的索引
        
        参数:
            tokens: 单个词元或词元列表
        
        返回:
            词元索引或索引列表
        """
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self._unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        """将索引转换为词元
        
        参数:
            indices: 单个索引或索引列表
        
        返回:
            词元或词元列表
        """
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """获取输入序列的词元及其片段索引
    
    为BERT准备输入序列，添加特殊的标记<cls>和<sep>。
    <cls>标记表示整个输入序列，<sep>标记用于分隔两个句子。
    
    参数:
        tokens_a: 第一个句子的词元列表
        tokens_b: 第二个句子的词元列表（可选）
    
    返回:
        (tokens, segments) 元组
        - tokens: 包含特殊标记的词元列表
        - segments: 片段索引列表，0表示片段A，1表示片段B
    """
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments


def _read_wiki(data_dir):
    """读取WikiText-2数据集
    
    参数:
        data_dir: 数据集目录路径
    
    返回:
        段落列表，每个段落是一个句子列表
    """
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    paragraphs = [line.strip().lower().split(' . ') for line in lines if len(line.split(' . ')) >= 2]
    return paragraphs


## 14.9.1 为预训练任务定义辅助函数
### 1. 生成下一句预测任务的数据
def _get_next_sentence(sentence, next_sentence, paragraphs):
    """生成下一句预测任务的数据
    
    随机决定下一句是否是真实的下一句，用于下一句预测任务。
    
    参数:
        sentence: 当前句子
        next_sentence: 下一句
        paragraphs: 所有段落的列表
    
    返回:
        (sentence, next_sentence, is_next) 元组
        - sentence: 当前句子
        - next_sentence: 下一句（可能是真实的下一句，也可能是随机句子）
        - is_next: 布尔值，表示 next_sentence 是否是真实的下一句
    """
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    """从段落中获取下一句预测任务的数据
    
    参数:
        paragraph: 当前段落
        paragraphs: 所有段落的列表
        vocab: 词表对象
        max_len: 最大序列长度
    
    返回:
        下一句预测任务的数据列表，每个元素是 (tokens, segments, is_next) 元组
    """
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


### 2. 生成掩码语言模型任务的数据
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    """替换掩码语言模型任务的词元
    
    在BERT预训练中，随机掩蔽输入序列中的一些词元。
    掩蔽策略：
    - 80% 的时间：替换为 "<mask>" 词元
    - 10% 的时间：保持词元不变
    - 10% 的时间：替换为随机词元
    
    参数:
        tokens: 词元列表
        candidate_pred_positions: 候选预测位置列表
        num_mlm_preds: 要预测的词元数量
        vocab: 词表对象
    
    返回:
        (mlm_input_tokens, pred_positions_and_labels) 元组
        - mlm_input_tokens: 掩蔽后的词元列表
        - pred_positions_and_labels: 预测位置和标签的列表
    """
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels


def _get_mlm_data_from_tokens(tokens, vocab):
    """从词元中获取掩码语言模型任务的数据
    
    参数:
        tokens: 词元列表
        vocab: 词表对象
    
    返回:
        (mlm_input_token_ids, pred_positions, mlm_pred_label_ids) 元组
        - mlm_input_token_ids: 掩蔽后的词元索引列表
        - pred_positions: 预测位置列表
        - mlm_pred_label_ids: 预测标签索引列表
    """
    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


## 14.9.2 将文本转换为预训练数据集
def _pad_bert_inputs(examples, max_len, vocab):
    """填充BERT的输入
    
    将不同长度的输入序列填充到相同的长度，以便批量处理。
    
    参数:
        examples: 示例列表，每个示例是 (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) 元组
        max_len: 最大序列长度
        vocab: 词表对象
    
    返回:
        (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels) 元组
    """
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens, = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (max_len - len(segments)), dtype=torch.long))
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        all_mlm_weights.append(torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels)


class _WikiTextDataset(torch.utils.data.Dataset):
    """用于加载WikiText-2数据集的自定义数据集
    
    参数:
        paragraphs: 段落列表
        max_len: 最大序列长度
    """
    def __init__(self, paragraphs, max_len):
        """初始化数据集
        
        参数:
            paragraphs: 段落列表
            max_len: 最大序列长度
        """
        paragraphs = [tokenize(paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocab(sentences, min_freq=5, reserved_tokens=['<pad>', '<mask>', '<cls>', '<sep>'])
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        examples = [_get_mlm_data_from_tokens(tokens, self.vocab) + (segments, is_next)
                    for tokens, segments, is_next in examples]
        (self.all_token_ids, self.all_segments, self.valid_lens, self.all_pred_positions, self.all_mlm_weights, self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(examples, max_len, self.vocab)

    def __getitem__(self, idx):
        """获取数据集中的单个样本
        
        参数:
            idx: 样本索引
        
        返回:
            (token_ids, segments, valid_lens, pred_positions, mlm_weights, mlm_labels, nsp_label) 元组
        """
        return (self.all_token_ids[idx], self.all_segments[idx], self.valid_lens[idx], self.all_pred_positions[idx], self.all_mlm_weights[idx], self.all_mlm_labels[idx], self.nsp_labels[idx])

    def __len__(self):
        """返回数据集大小
        
        返回:
            数据集中的样本数量
        """
        return len(self.all_token_ids)


def load_data_wiki(batch_size, max_len):
    """加载WikiText-2数据集
    
    参数:
        batch_size: 批量大小
        max_len: 最大序列长度
    
    返回:
        (train_iter, vocab) 元组
        - train_iter: 训练数据迭代器
        - vocab: 词表对象
    """
    num_workers = 0
    from downloader.wikitext import get_dataset_path
    data_dir = get_dataset_path()
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab


if __name__ == "__main__":
    print("=" * 60)
    print("14.9 用于预训练BERT的数据集")
    print("=" * 60)
    
    batch_size, max_len = 512, 64
    train_iter, vocab = load_data_wiki(batch_size, max_len)
    
    print(f"\n词表大小: {len(vocab)}")
    print(f"数据集大小: {len(train_iter.dataset)}")
    
    for (tokens_X, segments_X, valid_lens_X, pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y) in train_iter:
        print(f"\ntokens_X.shape: {tokens_X.shape}")
        print(f"segments_X.shape: {segments_X.shape}")
        print(f"valid_lens_X.shape: {valid_lens_X.shape}")
        print(f"pred_positions_X.shape: {pred_positions_X.shape}")
        print(f"mlm_weights_X.shape: {mlm_weights_X.shape}")
        print(f"mlm_Y.shape: {mlm_Y.shape}")
        print(f"nsp_Y.shape: {nsp_Y.shape}")
        break
