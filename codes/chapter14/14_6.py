# 14.6 子词嵌入

## 14.6.2 字节对编码
import collections

# 定义初始符号表，包含所有小写字母和未知词标记
# 字节对编码（BPE）是一种子词分词方法，通过迭代合并最频繁的字符对来构建子词
symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']

# 原始词元频率，模拟一些词的后缀及其出现频率
# 例如：'fast_' 出现 4 次，'faster_' 出现 3 次，'tall_' 出现 5 次，'taller_' 出现 4 次
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}

# 将原始词元转换为字符列表形式
# 例如：'fast_' -> ['f', 'a', 's', 't', '_']
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
print("token_freqs:", token_freqs)

def get_max_freq_pair(token_freqs):
    """返回出现次数最多的一个符号对
    
    该函数遍历所有词元，统计每对相邻符号的出现频率，
    并返回出现次数最多的符号对。
    
    参数:
        token_freqs: 词元频率字典，键为词元（字符列表形式），值为频率
    
    返回:
        出现次数最多的符号对（元组形式）
    """
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # "pairs"的键是两个连续符号的元组
            # 例如：['f', 'a', 's', 't', '_'] 会产生 ('f', 'a'), ('a', 's'), ('s', 't'), ('t', '_')
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # 具有最大值的"pairs"键

def merge_symbols(max_freq_pair, token_freqs, symbols):
    """合并符号对以生成新符号
    
    该函数将最频繁的符号对合并为一个新的符号，
    并更新所有词元中的符号表示。
    
    参数:
        max_freq_pair: 要合并的符号对（元组形式）
        token_freqs: 词元频率字典
        symbols: 符号列表，新符号将被添加到这个列表中
    
    返回:
        更新后的词元频率字典
    """
    # 将新的合并符号添加到符号表中
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        # 在词元中替换符号对为新的合并符号
        # 例如：如果 max_freq_pair = ('t', '_')，则 't _' 会被替换为 't_'
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs

# 执行字节对编码的合并过程
# num_merges 指定要执行的合并次数
num_merges = 10
for i in range(num_merges):
    # 找到最频繁的符号对
    max_freq_pair = get_max_freq_pair(token_freqs)
    # 合并该符号对
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f"merge #{i+1}:", max_freq_pair)

print("symbols:", symbols)
print("list(token_freqs.keys()):", list(token_freqs.keys()))

def segment_BPE(tokens, symbols):
    """使用 BPE 进行分词
    
    该函数将词元分解为子词序列。
    它使用贪婪算法，尽可能匹配最长的子词。
    
    参数:
        tokens: 词元列表
        symbols: 子词符号列表（由 BPE 合并过程生成）
    
    返回:
        分词后的子词列表
    """
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # 使用滑动窗口在词元中查找匹配的子词
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                # 如果找到匹配的子词，将其添加到输出中
                outputs.append(token[start: end])
                start = end
                end = len(token)
            else:
                # 如果没有找到匹配，缩短窗口
                end -= 1
        # 如果词元中仍有未匹配的部分，标记为未知词
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs

# 测试 BPE 分词
tokens = ['tallest_', 'fatter_']
print("segment_BPE(tokens, symbols):", segment_BPE(tokens, symbols))

if __name__ == "__main__":
    print("=" * 60)
    print("14.6 子词嵌入 - 字节对编码 (BPE)")
    print("=" * 60)
    
    print("\n初始符号表:")
    print(f"  - 符号数量: {len(symbols)}")
    print(f"  - 符号: {symbols}")
    
    print("\n原始词元频率:")
    print(f"  {raw_token_freqs}")
    
    print("\nBPE 合并过程:")
    print(f"  - 合并次数: {num_merges}")
    print(f"  - 最终符号数量: {len(symbols)}")
    
    print("\nBPE 分词示例:")
    print(f"  - 输入词元: {tokens}")
    print(f"  - 分词结果: {segment_BPE(tokens, symbols)}")
