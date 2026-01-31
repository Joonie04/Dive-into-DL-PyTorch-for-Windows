import os
import zipfile
import requests
from pathlib import Path

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATASET_NAME = "wiki.en"
DATASET_HASH = "c1816da3821ae9f43899be655002f6c723e91b88"

def download_wiki_en():
    """下载 Wiki.en 数据集
    
    Wiki.en 是从维基百科提取的英文文本数据集，用于预训练词嵌入和语言模型训练。
    数据集包含大量的英文文本，可以用于学习词的语义表示。
    
    返回:
        数据集目录路径
    """
    dataset_dir = Path("dataset/wiki_en")
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
    
    vec_file = dataset_dir / "wiki.en.vec"
    
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
    dataset_dir = Path("dataset/wiki_en")
    if dataset_dir.exists():
        return str(dataset_dir)
    return download_wiki_en()

if __name__ == "__main__":
    path = download_wiki_en()
    if path:
        print(f"数据集路径: {path}")
