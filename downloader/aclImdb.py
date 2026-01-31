import os
import tarfile
import requests
from pathlib import Path


DATA_URL = "https://ai.stanford.edu/~amaas/data/sentiment/"
DATASET_NAME = "aclImdb_v1"


def download_aclimdb():
    """下载 ACL IMDB 数据集
    
    ACL IMDB 是一个用于情感分析的大型电影评论数据集。
    数据集包含 50,000 条评论，分为训练集和测试集，每集包含 25,000 条评论。
    每个评论都标记为正面（pos）或负面（neg）。
    
    返回:
        数据集目录路径
    """
    dataset_dir = Path("dataset/aclImdb")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = dataset_dir / f"{DATASET_NAME}.tar.gz"
    
    if tar_path.exists():
        print(f"数据集已存在于 {tar_path}")
    else:
        url = DATA_URL + f"{DATASET_NAME}.tar.gz"
        print(f"正在下载数据集: {url}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            downloaded = 0
            
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r下载进度: {percent:.1f}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)", end='')
            
            print(f"\n下载完成: {tar_path}")
        except Exception as e:
            print(f"\n下载失败: {e}")
            return None
    
    train_pos_dir = dataset_dir / "aclImdb" / "train" / "pos"
    train_neg_dir = dataset_dir / "aclImdb" / "train" / "neg"
    test_pos_dir = dataset_dir / "aclImdb" / "test" / "pos"
    test_neg_dir = dataset_dir / "aclImdb" / "test" / "neg"
    
    if not (train_pos_dir.exists() and train_neg_dir.exists() and 
            test_pos_dir.exists() and test_neg_dir.exists()):
        print(f"正在解压数据集: {tar_path}")
        try:
            with tarfile.open(tar_path, 'r:gz') as tar_ref:
                tar_ref.extractall(dataset_dir)
            print(f"解压完成: {dataset_dir}")
        except Exception as e:
            print(f"解压失败: {e}")
            return None
    
    return str(dataset_dir / "aclImdb")


def get_dataset_path():
    """获取数据集路径
    
    如果数据集不存在，则自动下载。
    
    返回:
        数据集目录路径
    """
    dataset_path = download_aclimdb()
    if dataset_path is None:
        raise FileNotFoundError("无法获取数据集路径")
    return dataset_path


if __name__ == "__main__":
    print("=" * 60)
    print("ACL IMDB 数据集下载工具")
    print("=" * 60)
    
    dataset_path = get_dataset_path()
    print(f"\n数据集路径: {dataset_path}")
