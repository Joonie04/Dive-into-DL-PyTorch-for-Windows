import os
import sys
import tarfile
import requests
from pathlib import Path

DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATASET_NAME = 'VOCtrainval_11-May-2012'
DATASET_HASH = '4e443f8a2eca6b1dac8a6c57641b67dd40621a49'

def download_voc2012():
    """下载 Pascal VOC2012 数据集
    
    返回:
        数据集目录路径
    """
    dataset_dir = Path("dataset/voc2012")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    tar_path = dataset_dir / f"{DATASET_NAME}.tar"
    
    if tar_path.exists():
        print(f"数据集已存在于 {tar_path}")
    else:
        url = DATA_URL + f"{DATASET_NAME}.tar"
        print(f"正在下载数据集: {url}")
        print(f"数据集大小约为 2GB，请耐心等待...")
        
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
    
    voc_dir = dataset_dir / "VOCdevkit" / "VOC2012"
    
    if not voc_dir.exists():
        print(f"正在解压数据集: {tar_path}")
        try:
            with tarfile.open(tar_path, 'r') as tar_ref:
                tar_ref.extractall(dataset_dir)
            print(f"解压完成: {dataset_dir}")
        except Exception as e:
            print(f"解压失败: {e}")
            return None
    
    return str(voc_dir)

if __name__ == "__main__":
    voc_dir = download_voc2012()
    if voc_dir:
        print(f"\n数据集路径: {voc_dir}")
        print("数据集结构:")
        print(f"  - JPEGImages: {os.path.join(voc_dir, 'JPEGImages')}")
        print(f"  - SegmentationClass: {os.path.join(voc_dir, 'SegmentationClass')}")
        print(f"  - ImageSets/Segmentation: {os.path.join(voc_dir, 'ImageSets', 'Segmentation')}")
