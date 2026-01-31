# 14.8 来自Transformer的双向编码器表示(BERT)

import math
import torch
from torch import nn


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽无效位置
    
    参数:
        X: 输入张量
        valid_len: 有效长度
        value: 用于填充的值
    
    返回:
        屏蔽后的张量
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作
    
    参数:
        X: 输入张量
        valid_lens: 有效长度
    
    返回:
        softmax后的张量
    """
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class DotProductAttention(nn.Module):
    """缩放点积注意力
    
    使用缩放的点积来计算查询和键之间的注意力分数。
    
    参数:
        dropout: dropout率
    """
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """前向传播
        
        参数:
            queries: 查询向量
            keys: 键向量
            values: 值向量
            valid_lens: 有效长度
        
        返回:
            注意力输出
        """
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


class MultiHeadAttention(nn.Module):
    """多头注意力
    
    将输入投影到多个子空间，并在每个子空间中独立计算注意力。
    
    参数:
        key_size: 键的维度
        query_size: 查询的维度
        value_size: 值的维度
        num_hiddens: 隐藏层维度
        num_heads: 注意力头的数量
        dropout: dropout率
        bias: 是否使用偏置
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        """前向传播
        
        参数:
            queries: 查询向量
            keys: 键向量
            values: 值向量
            valid_lens: 有效长度
        
        返回:
            多头注意力输出
        """
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)


def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状
    
    参数:
        X: 输入张量
        num_heads: 注意力头的数量
    
    返回:
        变换后的张量
    """
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作
    
    参数:
        X: 输入张量
        num_heads: 注意力头的数量
    
    返回:
        逆转后的张量
    """
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)


class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络
    
    对每个位置独立地应用相同的前馈网络。
    
    参数:
        num_hiddens: 隐藏层维度
        ffn_num_hiddens: 前馈网络隐藏层维度
        dropout: dropout率
    """
    def __init__(self, num_hiddens, ffn_num_hiddens, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(num_hiddens, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(ffn_num_hiddens, num_hiddens)
    
    def forward(self, X):
        """前向传播
        
        参数:
            X: 输入张量
        
        返回:
            前馈网络输出
        """
        return self.dense2(self.dropout(self.relu(self.dense1(X))))


class AddNorm(nn.Module):
    """残差连接和层规范化
    
    将残差连接和层规范化组合在一起。
    
    参数:
        normalized_shape: 规范化的形状
        dropout: dropout率
    """
    def __init__(self, normalized_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """前向传播
        
        参数:
            X: 输入张量
            Y: 要添加的张量
        
        返回:
            残差连接和层规范化后的输出
        """
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):
    """Transformer编码器块
    
    包含多头注意力、残差连接、层规范化和前馈网络。
    
    参数:
        key_size: 键的维度
        query_size: 查询的维度
        value_size: 值的维度
        num_hiddens: 隐藏层维度
        norm_shape: 规范化的形状
        ffn_num_input: 前馈网络输入维度
        ffn_num_hiddens: 前馈网络隐藏层维度
        num_heads: 注意力头的数量
        dropout: dropout率
        bias: 是否使用偏置
    """
    def __init__(self, key_size, query_size, value_size, num_hiddens, norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads, dropout, bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
                                            num_heads, dropout, bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        """前向传播
        
        参数:
            X: 输入张量
            valid_lens: 有效长度
        
        返回:
            编码器块输出
        """
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))


## 14.8.4 输入表示
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


class BERTEncoder(nn.Module):
    """BERT编码器
    
    BERT（Bidirectional Encoder Representations from Transformers）是一种预训练语言模型，
    使用Transformer编码器来学习双向的上下文表示。
    
    参数:
        vocab_size: 词表大小
        num_hiddens: 隐藏层维度
        norm_shape: 层规范化的形状
        ffn_num_input: 前馈网络输入维度
        ffn_num_hiddens: 前馈网络隐藏层维度
        num_heads: 注意力头的数量
        num_layers: 编码器层数
        dropout: dropout率
        max_len: 最大序列长度
        key_size: 键的维度
        query_size: 查询的维度
        value_size: 值的维度
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape, 
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens=None):
        """前向传播
        
        参数:
            tokens: 词元索引张量，形状为 (batch_size, num_steps)
            segments: 片段索引张量，形状为 (batch_size, num_steps)
            valid_lens: 有效长度（可选）
        
        返回:
            编码后的表示，形状为 (batch_size, num_steps, num_hiddens)
        """
        x = self.token_embedding(tokens) + self.segment_embedding(segments) + self.pos_embedding[:, :tokens.shape[1], :]
        for blk in self.blks:
            x = blk(x, valid_lens)
        return x


vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
print("encoded_X.shape:", encoded_X.shape)


## 14.8.5 预训练任务
### 1. 掩蔽语言模型(masked language modeling)
class MaskLM(nn.Module):
    """掩蔽语言模型任务
    
    在BERT预训练中，随机掩蔽输入序列中的一些词元，
    然后模型需要预测这些被掩蔽的词元。
    
    参数:
        vocab_size: 词表大小
        num_hiddens: 隐藏层维度
        num_inputs: 输入维度
    """
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Embedding(vocab_size, num_hiddens))
            

    def forward(self, X, pred_positions):
        """前向传播
        
        参数:
            X: 编码器输出，形状为 (batch_size, num_steps, num_hiddens)
            pred_positions: 预测位置，形状为 (batch_size, num_predictions)
        
        返回:
            预测结果，形状为 (batch_size, num_predictions, vocab_size)
        """
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
print("mlm_Y_hat.shape:", mlm_Y_hat.shape)
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
print("mlm_l.shape:", mlm_l.shape)


### 2. 下一句预测(next sentence prediction)
class NextSentencePred(nn.Module):
    """下一句预测任务
    
    在BERT预训练中，模型需要预测两个句子是否是连续的。
    
    参数:
        num_inputs: 输入维度
    """
    def __init__(self, num_inputs=768, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        """前向传播
        
        参数:
            X: 输入张量，形状为 (batch_size, num_hiddens)
        
        返回:
            预测结果，形状为 (batch_size, 2)
        """
        return self.output(X)


encoded_X = torch.flatten(encoded_X, start_dim=1)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
print("nsp_Y_hat.shape:", nsp_Y_hat.shape)
nsp_Y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_Y)
print("nsp_l.shape:", nsp_l.shape)


## 14.8.6 整合代码
class BERTModel(nn.Module):
    """BERT模型
    
    整合了BERT编码器、掩蔽语言模型和下一句预测任务。
    
    参数:
        vocab_size: 词表大小
        num_hiddens: 隐藏层维度
        norm_shape: 层规范化的形状
        ffn_num_input: 前馈网络输入维度
        ffn_num_hiddens: 前馈网络隐藏层维度
        num_heads: 注意力头的数量
        num_layers: 编码器层数
        dropout: dropout率
        max_len: 最大序列长度
        key_size: 键的维度
        query_size: 查询的维度
        value_size: 值的维度
        hid_in_features: 隐藏层特征维度
        mlm_in_features: 掩蔽语言模型输入特征维度
        nsp_in_features: 下一句预测输入特征维度
    """
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768, nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=max_len, key_size=key_size, query_size=query_size, value_size=value_size)
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        """前向传播
        
        参数:
            tokens: 词元索引张量
            segments: 片段索引张量
            valid_lens: 有效长度（可选）
            pred_positions: 预测位置（可选）
        
        返回:
            (encoded_X, mlm_Y_hat, nsp_Y_hat) 元组
            - encoded_X: 编码后的表示
            - mlm_Y_hat: 掩蔽语言模型预测结果
            - nsp_Y_hat: 下一句预测结果
        """
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(encoded_X[:, 0, :])
        return encoded_X, mlm_Y_hat, nsp_Y_hat


if __name__ == "__main__":
    print("=" * 60)
    print("14.8 来自Transformer的双向编码器表示(BERT)")
    print("=" * 60)
    
    print("\n## 14.8.4 输入表示")
    print("测试BERTEncoder...")
    vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
    norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
    encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens, num_heads, num_layers, dropout)
    tokens = torch.randint(0, vocab_size, (2, 8))
    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
    encoded_X = encoder(tokens, segments, None)
    print(f"encoded_X.shape: {encoded_X.shape}")
    
    print("\n## 14.8.5 预训练任务")
    print("\n### 1. 掩蔽语言模型(masked language modeling)")
    mlm = MaskLM(vocab_size, num_hiddens)
    mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
    mlm_Y_hat = mlm(encoded_X, mlm_positions)
    print(f"mlm_Y_hat.shape: {mlm_Y_hat.shape}")
    mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
    loss = nn.CrossEntropyLoss(reduction='none')
    mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
    print(f"mlm_l.shape: {mlm_l.shape}")
    
    print("\n### 2. 下一句预测(next sentence prediction)")
    encoded_X_flat = torch.flatten(encoded_X, start_dim=1)
    nsp = NextSentencePred(encoded_X_flat.shape[-1])
    nsp_Y_hat = nsp(encoded_X_flat)
    print(f"nsp_Y_hat.shape: {nsp_Y_hat.shape}")
    nsp_Y = torch.tensor([0, 1])
    nsp_l = loss(nsp_Y_hat, nsp_Y)
    print(f"nsp_l.shape: {nsp_l.shape}")
    
    print("\n## 14.8.6 整合代码")
    print("测试BERTModel...")
    bert = BERTModel(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                     ffn_num_hiddens, num_heads, num_layers, dropout)
    encoded_X, mlm_Y_hat, nsp_Y_hat = bert(tokens, segments, None, mlm_positions)
    print(f"encoded_X.shape: {encoded_X.shape}")
    print(f"mlm_Y_hat.shape: {mlm_Y_hat.shape}")
    print(f"nsp_Y_hat.shape: {nsp_Y_hat.shape}")
