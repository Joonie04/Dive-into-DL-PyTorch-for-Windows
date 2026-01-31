import os  # 导入os模块，用于文件路径操作
import requests  # 导入requests模块，用于下载文件
import zipfile  # 导入zipfile模块，用于解压文件
import hashlib  # 导入hashlib模块，用于计算文件哈希值

# 定义数据集URL和哈希值
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'  # d2l数据集的基础URL
DATA_HUB = {  # 数据集信息字典
    'fra-eng': (DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')  # 法语-英语数据集：URL和SHA1哈希值
}

# 定义数据集保存目录
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')  # 数据集目录：downloader/../dataset

def download(name):  # 定义下载函数
    """下载并解压数据集"""
    if name not in DATA_HUB:  # 如果数据集名称不在DATA_HUB中
        raise ValueError(f'未知的数据集：{name}')  # 抛出错误
    
    url, sha1 = DATA_HUB[name]  # 获取数据集的URL和哈希值
    filename = os.path.join(DATASET_DIR, f'{name}.zip')  # 设置下载文件的保存路径
    extract_dir = os.path.join(DATASET_DIR, name)  # 设置解压目录
    
    # 如果解压目录已存在，直接返回
    if os.path.exists(extract_dir):  # 如果解压目录已存在
        return extract_dir  # 直接返回解压目录
    
    # 确保数据集目录存在
    os.makedirs(DATASET_DIR, exist_ok=True)  # 创建数据集目录（如果不存在）
    
    # 下载文件
    print(f'正在下载 {name}...')  # 打印下载信息
    response = requests.get(url, stream=True)  # 发送GET请求，使用流式下载
    response.raise_for_status()  # 检查请求是否成功
    
    # 保存文件
    with open(filename, 'wb') as f:  # 以二进制写入模式打开文件
        for chunk in response.iter_content(chunk_size=8192):  # 分块下载
            if chunk:  # 如果数据块不为空
                f.write(chunk)  # 写入数据块
    
    # 验证文件哈希值
    print(f'正在验证 {name}...')  # 打印验证信息
    with open(filename, 'rb') as f:  # 以二进制读取模式打开文件
        file_hash = hashlib.sha1(f.read()).hexdigest()  # 计算文件的SHA1哈希值
    if file_hash != sha1:  # 如果哈希值不匹配
        os.remove(filename)  # 删除下载的文件
        raise ValueError(f'文件哈希值不匹配：{file_hash} != {sha1}')  # 抛出错误
    
    # 解压文件
    print(f'正在解压 {name}...')  # 打印解压信息
    with zipfile.ZipFile(filename, 'r') as zip_ref:  # 以读取模式打开zip文件
        zip_ref.extractall(DATASET_DIR)  # 解压到数据集目录
    
    print(f'{name} 下载完成！')  # 打印完成信息
    return extract_dir  # 返回解压目录
