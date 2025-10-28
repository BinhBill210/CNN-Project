"""
Gradio app: Phân loại giống chó từ ảnh
"""
from pathlib import Path
from typing import Optional
import json
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Import cấu hình và utils nội bộ
from config import FINAL_MODEL_DIR, IMG_SIZE, get_class_names_path
from utils import save_class_names, load_class_names


def get_latest_model_path() -> Path:
    """Lấy đường dẫn model .keras mới nhất trong models/final."""
    final_dir = FINAL_MODEL_DIR
    final_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(final_dir.glob("*.keras"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not models:
        raise FileNotFoundError("Không tìm thấy file .keras trong models/final")
    return models[0]


def ensure_class_names() -> Path:
    """Đảm bảo có file class_names.json, nếu chưa có thì tạo mới."""
    class_path = get_class_names_path()
    if not class_path.exists():
        save_class_names(class_path)
    return class_path


class Predictor:
    def __init__(self):
        model_path = get_latest_model_path()
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = load_class_names(ensure_class_names())

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img, dtype=np.float32)
        arr = tf.keras.applications.efficientnet.preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)
        return arr

    def predict(self, image: Image.Image, top_k: int = 3):
        x = self.preprocess(image)
        probs = self.model.predict(x, verbose=0)[0]
        top_idx = np.argsort(probs)[-top_k:][::-1]
        labels = [self.class_names[i] for i in top_idx]
        scores = [float(probs[i]) for i in top_idx]
        return {label: score for label, score in zip(labels, scores)}


predictor: Optional[Predictor] = None


def infer(image: Image.Image):
    global predictor
    if predictor is None:
        predictor = Predictor()
    return predictor.predict(image)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Dog Breed Classification") as demo:
        gr.Markdown("""
        **Dog Breed Classification** — Tải ảnh chó của bạn để dự đoán giống.
        """)
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Ảnh đầu vào")
                btn = gr.Button("Dự đoán")
            with gr.Column():
                out = gr.Label(num_top_classes=3, label="Top-3 kết quả")

        btn.click(infer, inputs=inp, outputs=out)
        inp.change(infer, inputs=inp, outputs=out)
    return demo


if __name__ == "__main__":
    app = build_app()
    # host=0.0.0.0 để có thể truy cập từ máy khác nếu cần
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)


