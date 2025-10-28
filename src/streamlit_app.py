"""
Web demo với Streamlit - Alternative cho Gradio
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from config import IMG_SIZE, BREED_NAMES, FINAL_MODEL_DIR
from utils import load_class_names


@st.cache_resource
def load_model():
    """Load model với caching"""
    model_dir = Path(FINAL_MODEL_DIR)
    models = list(model_dir.glob('*.keras'))
    
    if not models:
        st.error("Không tìm thấy model. Hãy train trước!")
        st.stop()
    
    latest_model = max(models, key=lambda p: p.stat().st_mtime)
    
    try:
        model = tf.keras.models.load_model(str(latest_model))
    except:
        model = tf.keras.models.load_model(str(latest_model), compile=False)
    
    return model, latest_model.name


def predict(image, model):
    """Predict giống chó"""
    # Resize và preprocess
    image = image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)[0]
    return predictions


# Page config
st.set_page_config(
    page_title="Dog Breed Classifier",
    page_icon="🐕",
    layout="centered"
)

# Title
st.title("🐕 Dog Breed Classifier")
st.markdown("Upload ảnh chó để nhận diện giống")

# Load model
model, model_name = load_model()
st.sidebar.success(f"Model: {model_name}")

# File uploader
uploaded_file = st.file_uploader(
    "Chọn ảnh...",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Ảnh đã upload", use_container_width=True)
    
    # Predict
    with st.spinner("Đang phân tích..."):
        predictions = predict(image, model)
    
    # Results
    st.subheader("Kết quả dự đoán")
    
    # Top-5
    top_indices = np.argsort(predictions)[-5:][::-1]
    
    for idx in top_indices:
        breed = BREED_NAMES[idx].replace('_', ' ')
        conf = predictions[idx]
        
        st.progress(float(conf))
        st.write(f"**{breed}**: {conf*100:.2f}%")
        st.write("")

st.markdown("---")
st.caption("Model: EfficientNetB3 | 10 dog breeds")
