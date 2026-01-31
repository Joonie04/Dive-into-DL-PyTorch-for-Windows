import os
import zipfile
import requests
from pathlib import Path

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATASET_NAME = "glove.42B.300d"
DATASET_HASH = "b5116e234e9eb9076672cfeabf5469f3eec904fa"

def download_glove_42b_300d():
    """下载 GloVe 42B 300d 词向量数据集
    
    GloVe (Global Vectors for Word Representation) 是一种无监督学习算法，
    用于获取词的向量表示。42B 表示在 420 亿词的语料库上训练，300d 表示词向量维度为 300。
    这是一个更大的数据集，词向量维度更高，可以捕捉更丰富的语义信息。
    
    返回:
        数据集目录路径
    """
    dataset_dir = Path("dataset/glove_42b_300d")
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
    
    vec_file = dataset_dir / "glove.42B.300d.txt"
    
    if not vec_file.exists():
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
    dataset_dir = Path("dataset/glove_42b_300d")
    if dataset_dir.exists():
        return str(dataset_dir)
    return download_glove_42b_300d()

if __name__ == "__main__":
    path = download_glove_42b_300d()
    if path:
        print(f"数据集路径: {path}")
