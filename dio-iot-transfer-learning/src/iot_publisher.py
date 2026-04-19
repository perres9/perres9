import argparse
import json
from pathlib import Path

import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf
from PIL import Image


IMG_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inferencia de imagem e publicacao MQTT.")
    parser.add_argument("--image", type=str, required=True, help="Caminho da imagem.")
    parser.add_argument("--model", type=str, default="models/image_classifier.keras", help="Modelo .keras")
    parser.add_argument("--labels", type=str, default="models/labels.txt", help="Arquivo de labels")
    parser.add_argument("--broker", type=str, default="test.mosquitto.org", help="Broker MQTT")
    parser.add_argument("--port", type=int, default=1883, help="Porta MQTT")
    parser.add_argument("--topic", type=str, default="dio/iot/predictions", help="Topico MQTT")
    return parser.parse_args()


def load_labels(labels_path: str):
    return [line.strip() for line in Path(labels_path).read_text(encoding="utf-8").splitlines() if line.strip()]


def preprocess_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path).convert("RGB").resize(IMG_SIZE)
    array = np.array(image, dtype=np.float32)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


def classify_image(model_path: str, labels_path: str, image_path: str):
    model = tf.keras.models.load_model(model_path)
    labels = load_labels(labels_path)

    image_batch = preprocess_image(image_path)
    probs = model.predict(image_batch, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx] if pred_idx < len(labels) else str(pred_idx)
    confidence = float(probs[pred_idx])

    return pred_label, confidence


def publish_prediction(broker: str, port: int, topic: str, payload: dict):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(broker, port, keepalive=30)
    client.loop_start()

    result = client.publish(topic, json.dumps(payload), qos=0, retain=False)
    result.wait_for_publish(timeout=5)

    client.loop_stop()
    client.disconnect()


def main():
    args = parse_args()

    prediction, confidence = classify_image(args.model, args.labels, args.image)

    payload = {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "source": Path(args.image).name,
    }

    publish_prediction(args.broker, args.port, args.topic, payload)

    print(f"Broker: {args.broker}:{args.port}")
    print(f"Topico: {args.topic}")
    print(f"Payload publicado: {json.dumps(payload, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
