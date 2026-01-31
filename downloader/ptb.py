import os
import requests
import zipfile
from pathlib import Path

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATASET_NAME = "ptb"
EXPECTED_HASH = "319d85e578af0cdc590547f26231e4e31cdf1e42"

def download_ptb():
    """下载 PTB (Penn Treebank) 数据集
    
    PTB 是一个常用的语言建模数据集，包含训练集、验证集和测试集。
    数据集用于预训练词嵌入和语言模型训练。
    
    返回:
        数据集目录路径
    """
    dataset_dir = Path("dataset/ptb")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = dataset_dir / f"{DATASET_NAME}.zip"
    
    if zip_path.exists():
        print(f"数据集已存在于 {zip_path}")
    else:
        url = DATA_URL + f"{DATASET_NAME}.zip"
        print(f"正在下载数据集: {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r下载进度: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end='')
            
            print(f"\n下载完成: {zip_path}")
        except Exception as e:
            print(f"\n下载失败: {e}")
            return None
    
    train_file = dataset_dir / "ptb.train.txt"
    valid_file = dataset_dir / "ptb.valid.txt"
    test_file = dataset_dir / "ptb.test.txt"
    
    if not train_file.exists() or not valid_file.exists() or not test_file.exists():
        print(f"正在解压数据集: {zip_path}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_dir)
            print(f"解压完成: {dataset_dir}")
        except Exception as e:
            print(f"解压失败: {e}")
            return None
    
    return str(dataset_dir)

if __name__ == "__main__":
    data_dir = download_ptb()
    if data_dir:
        print(f"\n数据集路径: {data_dir}")
        print("数据集结构:")
        print(f"  - ptb.train.txt: {os.path.join(data_dir, 'ptb.train.txt')}")
        print(f"  - ptb.valid.txt: {os.path.join(data_dir, 'ptb.valid.txt')}")
        print(f"  - ptb.test.txt: {os.path.join(data_dir, 'ptb.test.txt')}")
