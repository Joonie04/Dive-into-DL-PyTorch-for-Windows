import os  # 导入操作系统模块
import requests  # 导入requests库，用于下载文件
import hashlib  # 导入hashlib库，用于计算文件的MD5值

# 定义数据集URL和MD5校验值
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'  # d2l数据集的基础URL
DATA_HUB = {  # 数据集字典，包含数据集名称和对应的URL、MD5校验值
    'time_machine': (DATA_URL + 'timemachine.txt', '090b5e7e70c295757f55df93cb0a180b9691891a')  # 时间机器数据集
}

# 定义数据集存储路径
DATA_DIR = 'dataset'  # 数据集存储目录

def download(name, cache_dir=os.path.join(DATA_DIR, 'time_machine')):  # 定义下载数据集的函数
    assert name in DATA_HUB, f'{name} 不存在于 {DATA_HUB}'  # 检查数据集名称是否存在于DATA_HUB中
    url, sha1_hash = DATA_HUB[name]  # 获取数据集的URL和MD5校验值
    os.makedirs(cache_dir, exist_ok=True)  # 创建数据集存储目录（如果不存在）
    fname = os.path.join(cache_dir, name + '.txt')  # 定义数据集文件路径
    
    if os.path.exists(fname):  # 如果文件已存在
        sha1 = hashlib.sha1()  # 创建SHA1哈希对象
        with open(fname, 'rb') as f:  # 以二进制读取模式打开文件
            while True:  # 循环读取文件
                data = f.read(1048576)  # 每次读取1MB数据
                if not data:  # 如果读取到文件末尾
                    break  # 退出循环
                sha1.update(data)  # 更新哈希值
        if sha1.hexdigest() == sha1_hash:  # 如果计算得到的哈希值与预期哈希值相同
            return fname  # 返回文件路径
    
    print(f'正在从 {url} 下载 {name}...')  # 打印下载信息
    r = requests.get(url, stream=True)  # 发送HTTP GET请求，启用流式下载
    r.raise_for_status()  # 检查请求是否成功
    
    total_size = int(r.headers.get('content-length', 0))  # 获取文件总大小
    block_size = 1024  # 设置块大小为1KB
    downloaded = 0  # 初始化已下载大小
    
    with open(fname, 'wb') as f:  # 以二进制写入模式打开文件
        for chunk in r.iter_content(chunk_size=block_size):  # 分块下载文件
            if chunk:  # 如果块不为空
                f.write(chunk)  # 写入块到文件
                downloaded += len(chunk)  # 更新已下载大小
                if total_size > 0:  # 如果知道文件总大小
                    percent = downloaded * 100 / total_size  # 计算下载进度百分比
                    print(f'\r下载进度: {percent:.1f}%', end='')  # 打印下载进度
    print()  # 打印换行
    
    sha1 = hashlib.sha1()  # 创建SHA1哈希对象
    with open(fname, 'rb') as f:  # 以二进制读取模式打开文件
        while True:  # 循环读取文件
            data = f.read(1048576)  # 每次读取1MB数据
            if not data:  # 如果读取到文件末尾
                break  # 退出循环
            sha1.update(data)  # 更新哈希值
    assert sha1.hexdigest() == sha1_hash, f'{name} 的哈希值不匹配'  # 检查哈希值是否匹配
    return fname  # 返回文件路径

if __name__ == '__main__':  # 如果作为主程序运行
    file_path = download('time_machine')  # 下载时间机器数据集
    print(f'数据集已下载到: {file_path}')  # 打印数据集路径
