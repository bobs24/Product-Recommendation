import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from model.feature_extractor import FeatureExtractor
from utils.image_utils import load_image_from_url

def main():
    df = pd.read_csv("data/product_data.csv")
    fe = FeatureExtractor()
    embeddings = []
    valid_urls = []

    # tqdm wraps the iterable and shows progress bar with description
    for url in tqdm(df['IMAGE_URL'], desc="Extracting embeddings"):
        img = load_image_from_url(url)
        if img is not None:
            emb = fe.extract(img)
            embeddings.append(emb)
            valid_urls.append(url)

    embeddings = np.array(embeddings)
    np.save("data/embeddings.npy", embeddings)

    with open("data/image_urls.pkl", "wb") as f:
        pickle.dump(valid_urls, f)

    print(f"Saved {len(valid_urls)} embeddings and URLs.")

if __name__ == "__main__":
    main()