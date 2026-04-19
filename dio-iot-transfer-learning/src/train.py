import argparse
from pathlib import Path

import tensorflow as tf


IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treino com Transfer Learning (MobileNetV2).")
    parser.add_argument("--data-dir", type=str, required=True, help="Diretorio com subpastas por classe.")
    parser.add_argument("--epochs", type=int, default=10, help="Numero de epocas.")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamanho do batch.")
    parser.add_argument(
        "--output-model",
        type=str,
        default="models/image_classifier.keras",
        help="Caminho de saida do modelo.",
    )
    parser.add_argument(
        "--output-labels",
        type=str,
        default="models/labels.txt",
        help="Arquivo para salvar nomes das classes.",
    )
    return parser.parse_args()


def build_datasets(data_dir: str, batch_size: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names


def build_model(num_classes: int) -> tf.keras.Model:
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ]
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels(labels_path: Path, class_names):
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text("\n".join(class_names), encoding="utf-8")


def main():
    args = parse_args()

    train_ds, val_ds, class_names = build_datasets(args.data_dir, args.batch_size)
    model = build_model(len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model)

    save_labels(Path(args.output_labels), class_names)

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    print(f"Treino concluido. Melhor val_accuracy: {best_val_acc:.4f}")
    print(f"Modelo salvo em: {output_model}")
    print(f"Labels salvos em: {args.output_labels}")


if __name__ == "__main__":
    main()
