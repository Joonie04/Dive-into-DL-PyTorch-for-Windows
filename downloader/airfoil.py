import os
import urllib.request
import hashlib

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATASET_DIR = r"d:\NEXT_LEVEL\Projects\DeepLearning\dataset"
AIRFOIL_FILENAME = "airfoil_self_noise.dat"
AIRFOIL_URL = DATA_URL + AIRFOIL_FILENAME
AIRFOIL_HASH = "76e5be1548fd8222e5074cf0faae75edff8cf93f"

def download_file(url, dest_path, expected_hash=None):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if os.path.exists(dest_path):
        if expected_hash:
            with open(dest_path, 'rb') as f:
                file_hash = hashlib.sha1(f.read()).hexdigest()
                if file_hash == expected_hash:
                    print(f"File already exists and verified: {dest_path}")
                    return dest_path
        else:
            print(f"File already exists: {dest_path}")
            return dest_path
    
    print(f"Downloading {url} to {dest_path}...")
    urllib.request.urlretrieve(url, dest_path)
    
    if expected_hash:
        with open(dest_path, 'rb') as f:
            file_hash = hashlib.sha1(f.read()).hexdigest()
            if file_hash != expected_hash:
                os.remove(dest_path)
                raise ValueError(f"File hash mismatch. Expected: {expected_hash}, Got: {file_hash}")
    
    print(f"Download completed: {dest_path}")
    return dest_path

def get_airfoil_dataset_path():
    dest_path = os.path.join(DATASET_DIR, AIRFOIL_FILENAME)
    return download_file(AIRFOIL_URL, dest_path, AIRFOIL_HASH)

if __name__ == "__main__":
    get_airfoil_dataset_path()
