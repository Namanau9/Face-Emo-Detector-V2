import argparse
import json
import os
from datetime import datetime
from typing import Dict, Tuple

import tensorflow as tf

from dataset_loader import build_test_dataset, build_train_validation_datasets


SEED = 42
tf.keras.utils.set_random_seed(SEED)


@tf.keras.utils.register_keras_serializable()
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
            name="attention_map",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        attention = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(attention)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@tf.keras.utils.register_keras_serializable()
class ModelPreprocessor(tf.keras.layers.Layer):
    def __init__(self, architecture: str = "mobilenetv2", **kwargs):
        super().__init__(**kwargs)
        self.architecture = architecture

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        if self.architecture == "efficientnetb0":
            return tf.keras.applications.efficientnet.preprocess_input(inputs)
        return tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"architecture": self.architecture})
        return config


BACKBONES = {
    "mobilenetv2": {
        "builder": tf.keras.applications.MobileNetV2,
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
        "default_image_size": 96,
        "recommended_fine_tune_layers": 20,
        "recommended_fine_tune_lr_ratio": 0.02,
    },
    "efficientnetb0": {
        "builder": tf.keras.applications.EfficientNetB0,
        "preprocess": tf.keras.applications.efficientnet.preprocess_input,
        "default_image_size": 128,
        "recommended_fine_tune_layers": 35,
        "recommended_fine_tune_lr_ratio": 0.01,
    },
}


def build_model(
    image_size: int,
    num_classes: int,
    architecture: str,
    base_trainable: bool = False,
) -> tf.keras.Model:
    backbone_config = BACKBONES[architecture]
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
            tf.keras.layers.RandomTranslation(0.06, 0.06),
        ],
        name="augmentation",
    )

    x = augmentation(inputs)
    x = ModelPreprocessor(architecture=architecture, name="preprocess")(x)

    base_model = backbone_config["builder"](
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base_model.trainable = base_trainable
    x = base_model(x, training=False)
    x = SpatialAttention(name="spatial_attention")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.20)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="emotion")(x)
    model = tf.keras.Model(inputs, outputs, name=f"emotion_{architecture}_attention")
    return model


def get_backbone_layer(model: tf.keras.Model, architecture: str) -> tf.keras.Model:
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and architecture in layer.name.lower():
            return layer
    raise ValueError(f"{architecture} backbone layer not found in the assembled model.")


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
        ],
    )


def callbacks(output_dir: str) -> list[tf.keras.callbacks.Callback]:
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoints_dir, "best_model.keras"),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "training_log.csv")),
    ]


def compute_class_weights(class_names: list[str], class_counts: Dict[str, int]) -> Dict[int, float]:
    total = sum(class_counts.values())
    num_classes = len(class_names)
    raw_weights = {
        idx: total / (num_classes * max(class_counts[class_name], 1))
        for idx, class_name in enumerate(class_names)
    }
    return {idx: min(weight, 2.0) for idx, weight in raw_weights.items()}


def set_backbone_trainable_range(base_model: tf.keras.Model, trainable_from: int) -> None:
    threshold = max(0, len(base_model.layers) - trainable_from)
    for index, layer in enumerate(base_model.layers):
        should_train = index >= threshold
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = should_train


def evaluate_and_export(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    output_dir: str,
    history: tf.keras.callbacks.History,
    dataset_stats: dict,
    export_metadata: dict,
) -> None:
    metrics = model.evaluate(test_ds, verbose=1, return_dict=True)
    model_path = os.path.join(output_dir, "emotion_model.keras")
    model.save(model_path)

    with open(os.path.join(output_dir, "labels.json"), "w", encoding="utf-8") as handle:
        json.dump({"labels": class_names, **export_metadata}, handle, indent=2)

    with open(os.path.join(output_dir, "training_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "saved_at": datetime.utcnow().isoformat() + "Z",
                "model_path": model_path,
                "labels": class_names,
                "architecture": export_metadata["architecture"],
                "image_size": export_metadata["image_size"],
                "metrics": {key: float(value) for key, value in metrics.items()},
                "history": {key: [float(v) for v in values] for key, values in history.history.items()},
                "dataset": dataset_stats,
            },
            handle,
            indent=2,
        )


def train(args: argparse.Namespace) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    train_ds, val_ds, class_names, class_counts, train_count, val_count = build_train_validation_datasets(
        train_dir=args.train_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        seed=SEED,
    )
    test_ds = build_test_dataset(
        test_dir=args.test_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=SEED,
    )

    model = build_model(
        args.image_size,
        num_classes=len(class_names),
        architecture=args.architecture,
        base_trainable=False,
    )
    class_weights = compute_class_weights(class_names, class_counts)
    compile_model(model, learning_rate=args.learning_rate)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks(args.output_dir),
        class_weight=class_weights if args.use_class_weights else None,
        verbose=1,
    )

    if args.fine_tune_epochs > 0:
        base_model = get_backbone_layer(model, args.architecture)
        base_model.trainable = True
        set_backbone_trainable_range(base_model, trainable_from=args.fine_tune_layers)
        compile_model(model, learning_rate=args.learning_rate * args.fine_tune_lr_ratio)
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs + args.fine_tune_epochs,
            initial_epoch=len(history.history["loss"]),
            callbacks=callbacks(args.output_dir),
            class_weight=class_weights if args.use_class_weights_in_finetune else None,
            verbose=1,
        )
        for key, values in fine_tune_history.history.items():
            if key in history.history:
                history.history[key].extend(values)
            else:
                history.history[key] = values

    dataset_stats = {
        "train_dir": args.train_dir,
        "test_dir": args.test_dir,
        "class_counts": class_counts,
        "class_weights": class_weights,
        "train_samples": train_count,
        "validation_samples": val_count,
    }
    export_metadata = {
        "architecture": args.architecture,
        "image_size": args.image_size,
        "preprocess": args.architecture,
    }
    evaluate_and_export(model, test_ds, class_names, args.output_dir, history, dataset_stats, export_metadata)
    return model, history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a face emotion recognition model.")
    parser.add_argument("--train-dir", default=os.path.join("..", "dataset", "train"))
    parser.add_argument("--test-dir", default=os.path.join("..", "dataset", "test"))
    parser.add_argument("--output-dir", default=os.path.join("..", "saved_model"))
    parser.add_argument("--architecture", choices=sorted(BACKBONES.keys()), default="efficientnetb0")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-layers", type=int, default=None)
    parser.add_argument("--fine-tune-lr-ratio", type=float, default=None)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--use-class-weights-in-finetune", action="store_true")
    args = parser.parse_args()
    backbone_config = BACKBONES[args.architecture]
    if args.image_size is None:
        args.image_size = backbone_config["default_image_size"]
    if args.fine_tune_layers is None:
        args.fine_tune_layers = backbone_config["recommended_fine_tune_layers"]
    if args.fine_tune_lr_ratio is None:
        args.fine_tune_lr_ratio = backbone_config["recommended_fine_tune_lr_ratio"]
    return args


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
