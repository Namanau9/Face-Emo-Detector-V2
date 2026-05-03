import math
import os
from typing import Dict, Tuple

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def count_images_by_class(directory: str) -> Dict[str, int]:
    counts = {}
    for class_name in sorted(os.listdir(directory)):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
        counts[class_name] = sum(
            1 for name in os.listdir(class_dir) if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
    return counts


def build_train_validation_datasets(
    train_dir: str,
    image_size: int,
    batch_size: int,
    validation_split: float,
    seed: int,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, list[str], Dict[str, int], int, int]:
    common_args = dict(
        directory=train_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        seed=seed,
        validation_split=validation_split,
    )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common_args,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=True,
        **common_args,
    )

    class_names = train_ds.class_names
    class_counts = count_images_by_class(train_dir)
    total_images = sum(class_counts.values())
    train_count = math.floor(total_images * (1 - validation_split))
    val_count = total_images - train_count

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names, class_counts, train_count, val_count


def build_test_dataset(test_dir: str, image_size: int, batch_size: int, seed: int) -> tf.data.Dataset:
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=(image_size, image_size),
        shuffle=False,
        seed=seed,
    )
    return test_ds.prefetch(AUTOTUNE)
