import requests
from io import BytesIO
from PIL import Image

def load_image_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        return image
    except Exception as e:
        print(f"Failed to load image from {url}: {e}")
        return None