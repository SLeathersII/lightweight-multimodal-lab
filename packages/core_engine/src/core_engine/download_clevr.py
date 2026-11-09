# packages/core_engine/src/core_engine/download_clevr.py
import os
import zipfile
import urllib.request

CLEVR_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "../../../data")

def download_and_extract_clevr():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "CLEVR_v1.0.zip")
    extract_path = os.path.join(DATA_DIR, "CLEVR_v1.0")

    if not os.path.exists(extract_path):
        print(f"Downloading official CLEVR dataset (~18 GB) to {zip_path}...")
        urllib.request.urlretrieve(CLEVR_URL, zip_path)
        print("Extracting CLEVR dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print("Dataset extracted successfully!")
    else:
        print(f"CLEVR dataset already exists at {extract_path}")

if __name__ == "__main__":
    download_and_extract_clevr()