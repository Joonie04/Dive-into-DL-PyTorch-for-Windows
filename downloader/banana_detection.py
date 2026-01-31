import os
import requests
import zipfile
from pathlib import Path

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATASET_NAME = "banana-detection"
EXPECTED_HASH = "5de26c8fce5ccdea9f91267273464dc968d20d72"

def download_banana_detection():
    dataset_dir = Path("dataset/banana-detection")
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
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"下载完成: {zip_path}")
        except Exception as e:
            print(f"下载失败: {e}")
            return None
    
    train_dir = dataset_dir / "bananas_train"
    val_dir = dataset_dir / "bananas_val"
    
    if not train_dir.exists() or not val_dir.exists():
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
    data_dir = download_banana_detection()
    if data_dir:
        print(f"数据集路径: {data_dir}")
