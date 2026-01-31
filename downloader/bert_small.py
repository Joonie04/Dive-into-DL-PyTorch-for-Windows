import os
import zipfile
import requests
from pathlib import Path


def get_dataset_path():
    """获取BERT small数据集路径
    
    返回:
        BERT small数据集目录路径
    """
    dataset_dir = Path(__file__).parent.parent / 'dataset' / 'bert.small'
    return str(dataset_dir)


def download_extract():
    """下载并解压BERT small预训练模型
    
    下载BERT small模型到dataset文件夹并解压。
    
    返回:
        解压后的数据集目录路径
    """
    data_dir = get_dataset_path()
    os.makedirs(data_dir, exist_ok=True)
    
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/bert.small.torch.zip'
    zip_path = os.path.join(data_dir, 'bert.small.torch.zip')
    
    if not os.path.exists(zip_path):
        print(f"正在下载BERT small模型到 {zip_path}...")
        response = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("下载完成")
    
    if not os.path.exists(os.path.join(data_dir, 'vocab.json')):
        print(f"正在解压 {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("解压完成")
    
    return data_dir


if __name__ == "__main__":
    download_extract()
