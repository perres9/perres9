import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


IMG_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predicao de imagem com modelo treinado.")
    parser.add_argument("--image", type=str, required=True, help="Caminho da imagem.")
    parser.add_argument("--model", type=str, required=True, help="Caminho do modelo .keras.")
    parser.add_argument("--labels", type=str, required=True, help="Arquivo de labels.")
    return parser.parse_args()


def load_labels(labels_path: str):
    return [line.strip() for line in Path(labels_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def preprocess_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def main():
    args = parse_args()

    model = tf.keras.models.load_model(args.model)
    labels = load_labels(args.labels)

    image_batch = preprocess_image(args.image)
    probs = model.predict(image_batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
    confidence = float(probs[pred_idx])

    print(f"Imagem: {args.image}")
    print(f"Predicao: {pred_label}")
    print(f"Confianca: {confidence:.4f}")


if __name__ == "__main__":
    main()
