import os
import zipfile
import requests
from pathlib import Path

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATASET_NAME = "wikitext-2-v1"
DATASET_HASH = "3c914d17d80b1459be871a5039ac23e752a53cbe"

def download_wikitext():
    """下载 WikiText-2 数据集
    
    WikiText-2 是一个用于语言建模和预训练的维基百科文本数据集。
    数据集包含训练集、验证集和测试集，用于预训练 BERT 等语言模型。
    
    返回:
        数据集目录路径
    """
    dataset_dir = Path("dataset/wikitext")
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
    
    train_file = dataset_dir / "wiki.train.tokens"
    valid_file = dataset_dir / "wiki.valid.tokens"
    test_file = dataset_dir / "wiki.test.tokens"
    
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

def get_dataset_path():
    """获取数据集路径
    
    返回:
        数据集目录路径
    """
    dataset_dir = Path("dataset/wikitext")
    if dataset_dir.exists():
        return str(dataset_dir)
    return download_wikitext()

if __name__ == "__main__":
    path = download_wikitext()
    if path:
        print(f"数据集路径: {path}")
