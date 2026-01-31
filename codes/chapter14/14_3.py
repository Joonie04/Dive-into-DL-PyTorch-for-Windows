# 14.3 用于预训练词嵌入的数据集
import collections
import math
import os
import random
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

## 14.3.1 读取数据集

def read_ptb():
    """将 PTB 数据集加载到文本行的列表中
    
    PTB (Penn Treebank) 是一个常用的语言建模数据集。
    该函数读取训练集文件，将每行文本分割成词元列表。
    
    返回:
        句子列表，每个句子是一个词元列表
    """
    from downloader.ptb import download_ptb
    data_dir = download_ptb()
    if data_dir is None:
        raise RuntimeError("数据集下载失败")
    
    with open(os.path.join(data_dir, 'ptb.train.txt'), 'r') as f:
        raw_text = f.read()
    return [line.strip().split() for line in raw_text.split('\n')]

sentences = read_ptb()
print(f'# sentences数: {len(sentences)}')

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

    @property
    def unk(self):
        """获取未知词元的索引
        
        返回:
            未知词元的索引
        """
        return self._unk

    @property
    def token_freqs(self):
        """获取词元频率列表
        
        返回:
            词元频率列表
        """
        return self.token_freqs

def count_corpus(tokens):
    """统计词元频率
    
    参数:
        tokens: 词元列表或词元列表的列表
    
    返回:
        词元频率计数器
    """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

vocab = Vocab(sentences, min_freq=10)
print(f'vocab size: {len(vocab)}')


## 14.3.2 下采样
def subsample(sentences, vocab):
    """下采样高频词
    
    高频词（如 "the", "a"）在文本中非常常见，但携带的信息量较少。
    下采样可以减少这些词的影响，使模型更关注有意义的词。
    
    参数:
        sentences: 句子列表
        vocab: 词表对象
    
    返回:
        (subsampled_sentences, counter) 元组
        - subsampled_sentences: 下采样后的句子列表
        - counter: 词元频率计数器
    """
    # 排除未知词元'<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]
    counter = count_corpus(sentences)
    num_tokens = sum(counter.values())

    # 如何在下采样期间保留词元, 则返回True
    # 下采样公式: P(keep) = sqrt(threshold / (freq(token) / num_tokens))
    # 其中 threshold 是一个超参数，通常设为 1e-4
    def keep(token):
        return random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens)

    return ([[token for token in line if keep(token)] for line in sentences], counter)

subsampled, counter = subsample(sentences, vocab)

def show_list_len_pair_hist(legends, xlabel, ylabel, xlist, ylist):
    """绘制两个列表长度的直方图
    
    参数:
        legends: 图例列表
        xlabel: x 轴标签
        ylabel: y 轴标签
        xlist: 第一个列表
        ylist: 第二个列表
    """
    _, _, patches = plt.hist([[len(l) for l in xlist], [len(l) for l in ylist]], bins=20)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for patch, legend in zip(patches, legends):
        patch.set_label(legend)
    plt.legend()
    plt.show()

show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence', 'count', sentences, subsampled)

def compare_counts(token):
    """比较下采样前后词元的数量
    
    参数:
        token: 要比较的词元
    
    返回:
        描述下采样前后词元数量的字符串
    """
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
compare_counts('join')
corpus = [token for line in subsampled for token in line]
print("corpus[:3]", corpus[:3])


## 14.3.3 中心词和上下文词的提取
def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中心词和上下文词
    
    跳元模型（Skip-gram）是一种词嵌入学习方法。
    对于每个中心词，我们预测其上下文窗口中的词。
    
    参数:
        corpus: 语料库，每个元素是一个句子的词元索引列表
        max_window_size: 最大上下文窗口大小
    
    返回:
        (centers, contexts) 元组
        - centers: 中心词列表
        - contexts: 上下文词列表，每个元素是一个上下文词索引列表
    """
    centers, contexts = [], []
    for line in corpus:
        # 要形成"中心词-上下文词"对, 每个句子至少需要有2个词
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # 上下文窗口中心为i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size), min(len(line), i + 1 + window_size)))
            indices.remove(i)  # 从上下文词中排除中心词
            contexts.append([line[idx] for idx in indices])
    return centers, contexts

tiny_dataset = [list(range(7)), list(range(7, 10))]
print('数据集', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('中心词', center, '的上下文词', context)

all_centers, all_contexts = get_centers_and_contexts(subsampled, 5)
print(f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}')


## 14.3.4 负采样
class RandomGenerator:
    """根据 n 个采样权重随机采样
    
    负采样是训练词嵌入时的一种技术。
    对于每个正样本（中心词-上下文词对），我们随机选择一些负样本（噪声词）。
    
    参数:
        sampling_weights: 采样权重列表
    """
    def __init__(self, sampling_weights):
        """初始化随机生成器
        
        参数:
            sampling_weights: 采样权重列表
        """
        self.population = list(range(len(sampling_weights)))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        """随机抽取一个样本
        
        返回:
            抽取的样本索引
        """
        if self.i == len(self.candidates):
            # 缓存k个随机采样索引
            self.candidates = random.choices(self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]

generator = RandomGenerator([2, 3, 4])
print([generator.draw() for _ in range(10)])

def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词
    
    负采样用于训练词嵌入模型。
    对于每个上下文词，我们随机选择 K 个噪声词。
    噪声词的选择基于词频的 0.75 次方，这样既能保持高频词的采样概率，
    又能增加低频词被选中的机会。
    
    参数:
        all_contexts: 所有上下文词列表
        vocab: 词表对象
        counter: 词元频率计数器
        K: 每个上下文词对应的负样本数量
    
    返回:
        负样本列表，每个元素是一个负样本索引列表
    """
    # 索引为0和1的词元是'UNK'和'<PAD>'
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75 for i in range(2, len(vocab))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, vocab, counter, 5)


## 14.3.5 小批量加载训练实例
def batchify(data):
    """将数据整理成小批量
    
    由于每个中心词对应的上下文词数量不同，我们需要将它们填充到相同的长度。
    我们使用掩码（mask）来区分真实的上下文词和填充的零。
    
    参数:
        data: 数据列表，每个元素是 (center, context, negative) 元组
    
    返回:
        (centers, contexts_negatives, masks, labels) 元组
        - centers: 中心词张量，形状为 (batch_size, 1)
        - contexts_negatives: 上下文词和负样本张量，形状为 (batch_size, max_len)
        - masks: 掩码张量，形状为 (batch_size, max_len)
        - labels: 标签张量，形状为 (batch_size, max_len)
    """
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(contexts_negatives),
            torch.tensor(masks), torch.tensor(labels))

x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
# 输出示例:
# centers = tensor([[1], [1]])
# contexts_negatives = tensor([[2, 2, 3, 3, 3, 3], [2, 2, 2, 3, 3, 0]])
# masks = tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
# labels = tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0]])


## 14.3.6 整合代码
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """加载 PTB 数据集, 然后将其加载到内存中
    
    该函数整合了所有数据预处理步骤：
    1. 读取数据集
    2. 构建词表
    3. 下采样高频词
    4. 提取中心词和上下文词
    5. 负采样
    6. 创建数据加载器
    
    参数:
        batch_size: 批次大小
        max_window_size: 最大上下文窗口大小
        num_noise_words: 每个上下文词对应的负样本数量
    
    返回:
        (data_iter, vocab) 元组
        - data_iter: 数据迭代器
        - vocab: 词表对象
    """
    num_workers = 0
    sentences = read_ptb()
    vocab = Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(Dataset):
        """PTB 数据集类
        
        参数:
            centers: 中心词列表
            contexts: 上下文词列表
            negatives: 负样本列表
        """
        def __init__(self, centers, contexts, negatives):
            """初始化 PTB 数据集
            
            参数:
                centers: 中心词列表
                contexts: 上下文词列表
                negatives: 负样本列表
            """
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives
        
        def __getitem__(self, index):
            """获取单个样本
            
            参数:
                index: 样本索引
            
            返回:
                (center, context, negative) 元组
            """
            return (self.centers[index], self.contexts[index], self.negatives[index])
        
        def __len__(self):
            """返回数据集大小
            
            返回:
                数据集样本数量
            """
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)
    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                            collate_fn=batchify, num_workers=num_workers)
    return data_iter, vocab

data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, '=', data.shape)
    break

if __name__ == "__main__":
    print("=" * 60)
    print("14.3 用于预训练词嵌入的数据集 (PTB)")
    print("=" * 60)
    
    print("\n数据集信息:")
    print(f"  - 句子数: {len(sentences)}")
    print(f"  - 词表大小: {len(vocab)}")
    print(f"  - 中心词-上下文词对数: {sum([len(contexts) for contexts in all_contexts])}")
    
    print("\n预处理参数:")
    print(f"  - 最小词频: 10")
    print(f"  - 最大上下文窗口大小: 5")
    print(f"  - 负样本数量: 5")
    print(f"  - 批次大小: 512")
