import os
import urllib.request
import zipfile

def download_cmapss():
    url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    zip_path = "CMAPSSData.zip"
    
    if not os.path.exists("train_FD001.txt"):
        print("Downloading NASA C-MAPSS dataset...")
        urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
            
        print("Dataset ready: train_FD001.txt is now in your working directory.")
    else:
        print("train_FD001.txt already exists locally.")

# Run it to fetch the file
download_cmapss()