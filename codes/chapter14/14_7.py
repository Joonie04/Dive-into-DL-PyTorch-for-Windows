# 14.7 词的相似度和类比任务
import os
import torch
from torch import nn
from pathlib import Path

class TokenEmbedding:
    """词元嵌入类
    
    用于加载和管理预训练的词向量，如 GloVe 词向量。
    提供了词元到索引、索引到词元的映射，以及获取词向量的功能。
    
    参数:
        embedding_name: 嵌入名称，用于指定要加载的词向量数据集
    """
    def __init__(self, embedding_name):
        """初始化词元嵌入
        
        参数:
            embedding_name: 嵌入名称
        """
        self.idx_to_token, self.idx_to_vec = self._load_embedding(embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        """加载词嵌入
        
        从本地数据集目录加载预训练的词向量文件。
        词向量文件格式为：每行一个词，第一个字段是词，后面是词向量值。
        
        参数:
            embedding_name: 嵌入名称
        
        返回:
            (idx_to_token, idx_to_vec) 元组
            - idx_to_token: 索引到词元的列表
            - idx_to_vec: 索引到词向量的张量
        """
        idx_to_token, idx_to_vec = ['<unk>'], []
        
        if embedding_name == 'glove.6b.50d':
            from downloader.glove_6b_50d import get_dataset_path
            data_dir = get_dataset_path()
            vec_file = Path(data_dir) / "glove.6B.50d.txt"
        elif embedding_name == 'glove.6b.100d':
            from downloader.glove_6b_100d import get_dataset_path
            data_dir = get_dataset_path()
            vec_file = Path(data_dir) / "glove.6B.100d.txt"
        elif embedding_name == 'glove.42b.300d':
            from downloader.glove_42b_300d import get_dataset_path
            data_dir = get_dataset_path()
            vec_file = Path(data_dir) / "glove.42B.300d.txt"
        elif embedding_name == 'wiki.en':
            from downloader.wiki_en import get_dataset_path
            data_dir = get_dataset_path()
            vec_file = Path(data_dir) / "wiki.en.vec"
        else:
            raise ValueError(f"未知的嵌入名称: {embedding_name}")

        with open(vec_file, 'r', encoding='utf-8') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, torch.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        """获取词元的词向量
        
        参数:
            tokens: 单个词元或词元列表
        
        返回:
            词向量张量
        """
        indices = [self.token_to_idx.get(token, self.unknown_idx) for token in tokens]
        vecs = self.idx_to_vec[torch.tensor(indices)]
        return vecs

    def __len__(self):
        """返回词表大小
        
        返回:
            词表中的词元数量
        """
        return len(self.idx_to_token)


def knn(W, x, k):
    """k 近邻搜索
    
    使用余弦相似度找到与查询向量最相似的 k 个向量。
    余弦相似度衡量两个向量之间的夹角，值越接近 1 表示越相似。
    
    参数:
        W: 词向量矩阵，形状为 (num_words, embedding_dim)
        x: 查询向量，形状为 (embedding_dim,)
        k: 要返回的最近邻数量
    
    返回:
        (topk, cos) 元组
        - topk: k 个最近邻的索引
        - cos: k 个最近邻的余弦相似度
    """
    cos = torch.mv(W, x.reshape(-1,)) / (torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) * torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]


def get_similar_tokens(query_token, k, embed):
    """获取与查询词最相似的词
    
    使用 k 近邻搜索找到与查询词最相似的 k 个词。
    相似度使用余弦相似度衡量。
    
    参数:
        query_token: 查询词
        k: 要返回的相似词数量
        embed: 词嵌入对象
    """
    W, x = embed.idx_to_vec, embed.token_to_idx[query_token]
    topk, cos = knn(W, x, k+1)
    for i, c in zip(topk[1:], cos[1:]):
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')


def get_analogy(token_a, token_b, token_c, embed):
    """词类比推理
    
    通过词向量运算完成类比推理。
    例如：man - woman + son = father，表示"男人"与"女人"的关系，
    类似于"儿子"与"父亲"的关系。
    
    参数:
        token_a: 词 a
        token_b: 词 b
        token_c: 词 c
        embed: 词嵌入对象
    
    返回:
        类比推理的结果词
    """
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]


if __name__ == "__main__":
    print("=" * 60)
    print("14.7 词的相似度和类比任务")
    print("=" * 60)
    
    print("\n## 14.7.1 加载预训练词向量")
    print("加载 GloVe 6B 50d 词向量数据集...")
    glove_6b50d = TokenEmbedding('glove.6b.50d')
    print(f"词表大小: {len(glove_6b50d)}")
    print(f"'beautiful' 的索引: {glove_6b50d.token_to_idx['beautiful']}")
    print(f"索引 3367 对应的词: {glove_6b50d.idx_to_token[3367]}")
    
    print("\n## 14.7.2 应用预训练词向量")
    
    print("\n### 1. 词相似度")
    print("查询与 'chip' 最相似的 3 个词:")
    get_similar_tokens('chip', 3, glove_6b50d)
    
    print("\n查询与 'baby' 最相似的 3 个词:")
    get_similar_tokens('baby', 3, glove_6b50d)
    
    print("\n查询与 'beautiful' 最相似的 3 个词:")
    get_similar_tokens('beautiful', 3, glove_6b50d)
    
    print("\n### 2. 词类比")
    print("类比推理: man - woman + son =", get_analogy('man', 'woman', 'son', glove_6b50d))
    print("类比推理: bad - worst + big =", get_analogy('bad', 'worst', 'big', glove_6b50d))
    print("类比推理: do - did + go =", get_analogy('do', 'did', 'go', glove_6b50d))
