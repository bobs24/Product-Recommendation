import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from model.feature_extractor import FeatureExtractor
from utils.faiss_index import FaissIndex
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import streamlit.components.v1 as components

st.set_page_config(page_title="🛍️ Product Recommender", layout="wide")

@st.cache_resource
def load_resources():
    embeddings = np.load("data/embeddings.npy")
    with open("data/image_urls.pkl", "rb") as f:
        image_urls = pickle.load(f)
    product_data = pd.read_csv("data/product_data.csv")
    fe = FeatureExtractor()
    index = FaissIndex(dim=embeddings.shape[1])
    index.build(embeddings, image_urls)
    return fe, index, image_urls, product_data

fe, index, image_urls, product_data = load_resources()

st.title("🛍️ Product Image Recommender")

uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    user_img = Image.open(uploaded_file).convert("RGB")
    st.image(user_img, caption="Uploaded Image", width=250)

    user_emb = fe.extract(user_img)
    results = index.search(user_emb, threshold=0.8, k=100)

    if len(results) > 0:
        input_image_url = results[0][0]

        # Get GROUP_ID of uploaded image
        input_group_id_series = product_data.loc[product_data['IMAGE'] == input_image_url, 'GROUP_ID']
        input_group_id = input_group_id_series.values[0] if not input_group_id_series.empty else None

        # Get PRODUCT_NAME of uploaded image
        input_product_name_series = product_data.loc[product_data['IMAGE'] == input_image_url, 'PRODUCT_NAME']
        input_product_name = input_product_name_series.values[0] if not input_product_name_series.empty else None

        # st.markdown(f"**GROUP_ID of uploaded image:** `{input_group_id}`")

        filtered_results = []
        for url, sim in results:
            group_id_series = product_data.loc[product_data['IMAGE'] == url, 'GROUP_ID']
            group_id = group_id_series.values[0] if not group_id_series.empty else None

            product_series = product_data.loc[product_data['IMAGE'] == url, 'PRODUCT_NAME']
            product_name = product_series.values[0] if not product_series.empty else None

            # Rule: if GROUP_ID is None or 0, exclude same product name
            if (input_group_id is None or input_group_id == 0):
                if product_name != input_product_name:
                    filtered_results.append((url, sim))
            else:
                if group_id != input_group_id:
                    filtered_results.append((url, sim))

        seen_products = set()
        deduped_results = []
        for url, sim in filtered_results:
            product_series = product_data.loc[product_data['IMAGE'] == url, 'PRODUCT_NAME']
            product_name = product_series.values[0] if not product_series.empty else None
            if product_name and product_name not in seen_products:
                seen_products.add(product_name)
                deduped_results.append((url, sim))

        top_results = deduped_results[:15]

        cards_html = ""
        for url, sim in top_results:
            brand = product_data.loc[product_data['IMAGE'] == url, 'BRAND_NAME'].values
            product = product_data.loc[product_data['IMAGE'] == url, 'PRODUCT_NAME'].values
            brand_name = brand[0] if len(brand) > 0 else "Unknown Brand"
            product_name = product[0] if len(product) > 0 else "Unknown Product"
            cards_html += f"""
                <div class="card">
                    <img src="{url}" alt="Product Image"/>
                    <div class="info">
                        <h4>{brand_name}</h4>
                        <p>{product_name}</p>
                        <span>Similarity: {sim:.2f}</span>
                    </div>
                </div>
            """

        full_html = f"""
        <style>
            .carousel-wrapper {{
                overflow-x: auto;
                overflow-y: visible;  /* allow vertical overflow if any */
                white-space: nowrap;
                padding: 20px 16px 40px 16px;
                height: auto;
                scroll-behavior: smooth;
            }}
            .carousel {{
                display: flex;
                gap: 10px;
                align-items: stretch;  /* all cards same height */
            }}
            .card {{
                flex: 0 0 auto;
                width: 280px;  /* 1.5x wider */
                /* no fixed height */
                border: 1px solid #ddd;
                border-radius: 14px;
                padding: 14px;
                background: #fff;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                text-align: center;
                box-sizing: border-box;
                transition: transform 0.2s ease-in-out;
                font-family: "Segoe UI", sans-serif;
            }}
            .card:hover {{
                transform: scale(1.04);
                box-shadow: 0 6px 16px rgba(0,0,0,0.12);
            }}
            .card img {{
                width: 100%;
                height: 300px;  /* 1.5x taller */
                object-fit: cover;
                border-radius: 8px;
            }}
            .info h4 {{
                font-size: 20px;
                margin: 12px 0 6px;
                color: #222;
                white-space: normal;
            }}
            .info p {{
                font-size: 16px;
                margin: 0 0 8px;
                color: #555;
                white-space: normal;
            }}
            .info span {{
                font-size: 13px;
                color: #888;
            }}
        </style>

        <div class="carousel-wrapper">
            <div class="carousel">
                {cards_html}
            </div>
        </div>
        """

        st.subheader("🔍 Recommended Products")
        components.html(full_html, height=600, scrolling=False)

    else:
        st.info("✨ No visually similar items found — this might be a one-of-a-kind product!")