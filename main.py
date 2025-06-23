import uvicorn
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO
from model.feature_extractor import FeatureExtractor
from utils.faiss_index import FaissIndex

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

# Load model and data
embeddings = np.load("data/embeddings.npy")
with open("data/image_urls.pkl", "rb") as f:
    image_urls = pickle.load(f)
product_data = pd.read_csv("data/product_data.csv")

fe = FeatureExtractor()
index = FaissIndex(dim=embeddings.shape[1])
index.build(embeddings, image_urls)

@app.post("/recommend")
async def recommend(file: UploadFile = File(...), threshold: float = 0.8, k: int = 100):
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        user_emb = fe.extract(image)
        results = index.search(user_emb, threshold=threshold, k=k)

        if not results:
            return JSONResponse({"message": "No similar products found"}, status_code=404)

        input_url = results[0][0]
        input_row = product_data[product_data['IMAGE'] == input_url]

        input_group_id = input_row['GROUP_ID'].values[0] if not input_row.empty else None
        input_product_name = input_row['PRODUCT_NAME'].values[0] if not input_row.empty else None

        # Filtering logic
        filtered = []
        for url, sim in results:
            row = product_data[product_data['IMAGE'] == url]
            group_id = row['GROUP_ID'].values[0] if not row.empty else None
            product_name = row['PRODUCT_NAME'].values[0] if not row.empty else None

            if (input_group_id is None or input_group_id == 0):
                if product_name != input_product_name:
                    filtered.append((url, sim))
            else:
                if group_id != input_group_id:
                    filtered.append((url, sim))

        # De-duplicate by product name
        seen = set()
        final = []
        for url, sim in filtered:
            row = product_data[product_data['IMAGE'] == url]
            product_name = row['PRODUCT_NAME'].values[0] if not row.empty else None
            if product_name and product_name not in seen:
                seen.add(product_name)
                brand_name = row['BRAND_NAME'].values[0] if 'BRAND_NAME' in row else "Unknown"
                final.append({
                    "brand_name": brand_name,
                    "product_name": product_name,
                    "image_url": url,
                    "similarity_score": float(f"{sim:.4f}")
                })

        return {"recommendations": final[:15]}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)