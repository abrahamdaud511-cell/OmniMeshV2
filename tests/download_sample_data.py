import os
import pandas as pd

def download_indonlu_smsa():
    url = "https://huggingface.co/datasets/indonlu/resolve/main/smsa/train.csv"
    os.makedirs("./data/raw", exist_ok=True)
    df = pd.read_csv(url)
    df = df[['text', 'label']]
    df.to_csv("./data/raw/smsa_train.csv", index=False)
    print("Sample data downloaded to ./data/raw/smsa_train.csv")

if __name__ == "__main__":
    download_indonlu_smsa()
