# 9.6 编码器-解码器架构（Encoder-Decoder Architecture）

from torch import nn  # 导入PyTorch神经网络模块

# 9.6.1 编码器
# 定义编码器类
class Encoder(nn.Module):  # 定义编码器类，继承自nn.Module
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):  # 初始化函数
        super(Encoder, self).__init__(**kwargs)  # 调用父类的初始化函数

    def forward(self, X, *args):  # 定义前向传播函数
        """前向传播函数"""
        raise NotImplementedError  # 抛出未实现错误，要求子类必须实现此方法

# 9.6.2 解码器
# 定义解码器类
class Decoder(nn.Module):  # 定义解码器类，继承自nn.Module
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):  # 初始化函数
        super(Decoder, self).__init__(**kwargs)  # 调用父类的初始化函数

    def init_state(self, enc_outputs, *args):  # 定义初始化状态的函数
        """初始化解码器的隐藏状态"""
        raise NotImplementedError  # 抛出未实现错误，要求子类必须实现此方法

    def forward(self, X, state):  # 定义前向传播函数
        """前向传播函数"""
        raise NotImplementedError  # 抛出未实现错误，要求子类必须实现此方法

# 9.6.3 合并编码器和解码器
# 定义编码器-解码器类
class EncoderDecoder(nn.Module):  # 定义编码器-解码器类，继承自nn.Module
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):  # 初始化函数
        super(EncoderDecoder, self).__init__(**kwargs)  # 调用父类的初始化函数
        self.encoder = encoder  # 保存编码器
        self.decoder = decoder  # 保存解码器

    def forward(self, enc_X, dec_X, *args):  # 定义前向传播函数
        """前向传播函数"""
        # 编码器处理输入序列
        enc_outputs = self.encoder(enc_X, *args)  # 将编码器输入传递给编码器，得到编码器输出
        # 初始化解码器的状态
        dec_state = self.decoder.init_state(enc_outputs, *args)  # 使用编码器输出初始化解码器状态
        # 解码器处理输出序列
        return self.decoder(dec_X, dec_state)  # 将解码器输入和状态传递给解码器，得到最终输出
