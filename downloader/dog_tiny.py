import os
import requests
import zipfile
from pathlib import Path

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATASET_NAME = "kaggle_dog_tiny"
EXPECTED_HASH = "0cb91d09b814ecdc07b50f31f8dcad3e81d6a86d"

def download_dog_tiny():
    """下载 Dog Tiny 数据集
    
    这是一个用于 Kaggle 竞赛的狗的品种识别数据集。
    数据集包含训练集、验证集和测试集，用于图像分类任务。
    
    返回:
        数据集目录路径
    """
    dataset_dir = Path("dataset/dog_tiny")
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
    
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"
    labels_file = dataset_dir / "labels.csv"
    
    if not train_dir.exists() or not test_dir.exists() or not labels_file.exists():
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
    data_dir = download_dog_tiny()
    if data_dir:
        print(f"\n数据集路径: {data_dir}")
        print("数据集结构:")
        print(f"  - train: {os.path.join(data_dir, 'train')}")
        print(f"  - test: {os.path.join(data_dir, 'test')}")
        print(f"  - labels.csv: {os.path.join(data_dir, 'labels.csv')}")
