from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

from config import DataConfig


AUTOTUNE = tf.data.AUTOTUNE


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def normalize_relative_path(value: object) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def resolve_image_path(dataset_root: Path, relative_path: object) -> Path:
    normalized = normalize_relative_path(relative_path)
    return dataset_root.joinpath(*normalized.split("/"))


def load_manifest(
    csv_path: Path,
    dataset_root: Path,
    split_name: str,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Не найден manifest/CSV для {split_name}: {csv_path}"
        )

    frame = pd.read_csv(csv_path)

    required_columns = {"Path", "ClassId"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path.name}: отсутствуют колонки "
            f"{sorted(missing_columns)}"
        )

    frame = frame.copy()
    frame["ClassId"] = pd.to_numeric(
        frame["ClassId"], errors="raise"
    ).astype(int)
    frame["Split"] = split_name
    frame["AbsolutePath"] = frame["Path"].map(
        lambda value: str(resolve_image_path(dataset_root, value))
    )

    bad_labels = frame[
        ~frame["ClassId"].between(0, 42, inclusive="both")
    ]
    if not bad_labels.empty:
        raise ValueError(
            f"{csv_path.name}: найдены ClassId вне диапазона 0..42"
        )

    missing_files = frame[
        ~frame["AbsolutePath"].map(lambda value: Path(value).exists())
    ]
    if not missing_files.empty:
        examples = "\n".join(
            missing_files["AbsolutePath"].head(10).tolist()
        )
        raise FileNotFoundError(
            f"{csv_path.name}: отсутствуют {len(missing_files)} файлов.\n"
            f"Первые примеры:\n{examples}"
        )

    duplicate_paths = frame["AbsolutePath"].str.lower().duplicated(
        keep=False
    )
    if duplicate_paths.any():
        examples = frame.loc[
            duplicate_paths, ["Path", "ClassId"]
        ].head(10)
        raise ValueError(
            f"{csv_path.name}: повторяющиеся пути внутри одного split.\n"
            f"{examples.to_string(index=False)}"
        )

    return frame


def load_all_manifests(
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = load_manifest(
        config.train_manifest_path,
        config.dataset_root,
        "train",
    )
    validation = load_manifest(
        config.validation_manifest_path,
        config.dataset_root,
        "validation",
    )
    test = load_manifest(
        config.test_manifest_path,
        config.dataset_root,
        "test",
    )

    expected_classes = set(range(config.number_of_classes))
    for split_name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        actual_classes = set(frame["ClassId"].unique())
        if actual_classes != expected_classes:
            missing = sorted(expected_classes - actual_classes)
            extra = sorted(actual_classes - expected_classes)
            raise ValueError(
                f"{split_name}: неправильный набор классов. "
                f"Отсутствуют: {missing}; лишние: {extra}"
            )

    return train, validation, test


def calculate_class_weights(
    train_frame: pd.DataFrame,
    number_of_classes: int,
) -> dict[int, float]:
    counts = (
        train_frame["ClassId"]
        .value_counts()
        .sort_index()
        .reindex(range(number_of_classes), fill_value=0)
    )

    if (counts == 0).any():
        empty_classes = counts[counts == 0].index.tolist()
        raise ValueError(
            f"Невозможно рассчитать веса: пустые классы {empty_classes}"
        )

    total = float(counts.sum())
    weights = total / (number_of_classes * counts.astype(float))
    return {
        int(class_id): float(weight)
        for class_id, weight in weights.items()
    }


def save_class_weights(
    class_weights: dict[int, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        str(class_id): round(weight, 8)
        for class_id, weight in class_weights.items()
    }
    output_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def create_augmentation_model(
    config: DataConfig,
) -> tf.keras.Sequential:
    """
    Геометрические и цветовые преобразования.

    Горизонтальный/вертикальный flip намеренно не используется:
    направление стрелки и положение символа могут менять смысл знака.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(
                    config.augmentation_resize,
                    config.augmentation_resize,
                    config.channels,
                )
            ),
            tf.keras.layers.RandomCrop(
                config.image_height,
                config.image_width,
                seed=config.random_seed + 1,
            ),
            tf.keras.layers.RandomRotation(
                factor=config.rotation_factor,
                fill_mode="reflect",
                interpolation="bilinear",
                seed=config.random_seed + 2,
            ),
            tf.keras.layers.RandomTranslation(
                height_factor=config.translation_factor,
                width_factor=config.translation_factor,
                fill_mode="reflect",
                interpolation="bilinear",
                seed=config.random_seed + 3,
            ),
            tf.keras.layers.RandomZoom(
                height_factor=(
                    -config.zoom_factor,
                    config.zoom_factor,
                ),
                width_factor=(
                    -config.zoom_factor,
                    config.zoom_factor,
                ),
                fill_mode="reflect",
                interpolation="bilinear",
                seed=config.random_seed + 4,
            ),
            tf.keras.layers.RandomBrightness(
                factor=config.brightness_factor,
                value_range=(0.0, 255.0),
                seed=config.random_seed + 5,
            ),
            tf.keras.layers.RandomContrast(
                factor=config.contrast_factor,
                seed=config.random_seed + 6,
            ),
        ],
        name="traffic_sign_augmentation",
    )


def _decode_rgb_image(path: tf.Tensor) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])
    return tf.cast(image, tf.float32)


def _maybe_add_noise(
    image: tf.Tensor,
    config: DataConfig,
) -> tf.Tensor:
    probability = tf.random.uniform(
        shape=[],
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    def add_noise() -> tf.Tensor:
        noise = tf.random.normal(
            shape=tf.shape(image),
            mean=0.0,
            stddev=config.noise_stddev_255,
            dtype=tf.float32,
        )
        return tf.clip_by_value(image + noise, 0.0, 255.0)

    return tf.cond(
        probability < config.noise_probability,
        add_noise,
        lambda: image,
    )


def _maybe_blur(
    image: tf.Tensor,
    config: DataConfig,
) -> tf.Tensor:
    probability = tf.random.uniform(
        shape=[],
        minval=0.0,
        maxval=1.0,
        dtype=tf.float32,
    )

    def blur() -> tf.Tensor:
        batched = tf.expand_dims(image, axis=0)
        blurred = tf.nn.avg_pool2d(
            batched,
            ksize=3,
            strides=1,
            padding="SAME",
        )
        return tf.squeeze(blurred, axis=0)

    return tf.cond(
        probability < config.blur_probability,
        blur,
        lambda: image,
    )


def normalize_image(
    image: tf.Tensor,
    mode: str,
) -> tf.Tensor:
    image = tf.cast(image, tf.float32)

    if mode == "zero_one":
        return image / 255.0

    if mode == "minus_one_one":
        return image / 127.5 - 1.0

    if mode == "none":
        return image

    raise ValueError(
        "normalization_mode должен быть одним из: "
        "'zero_one', 'minus_one_one', 'none'"
    )


def denormalize_for_display(
    image: tf.Tensor | np.ndarray,
    mode: str,
) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)

    if mode == "zero_one":
        array = array * 255.0
    elif mode == "minus_one_one":
        array = (array + 1.0) * 127.5
    elif mode == "none":
        pass
    else:
        raise ValueError(f"Неизвестный режим нормализации: {mode}")

    return np.clip(array, 0.0, 255.0).astype(np.uint8)


def make_preprocess_function(
    config: DataConfig,
    training: bool,
    augmenter: tf.keras.Model | None,
):
    def preprocess(
        image_path: tf.Tensor,
        label: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        image = _decode_rgb_image(image_path)

        if training:
            image = tf.image.resize(
                image,
                size=(
                    config.augmentation_resize,
                    config.augmentation_resize,
                ),
                method="bilinear",
                antialias=True,
            )

            if augmenter is None:
                raise RuntimeError(
                    "Для training требуется augmentation model."
                )

            image = augmenter(
                tf.expand_dims(image, axis=0),
                training=True,
            )
            image = tf.squeeze(image, axis=0)
            image = _maybe_blur(image, config)
            image = _maybe_add_noise(image, config)
        else:
            image = tf.image.resize(
                image,
                size=(config.image_height, config.image_width),
                method="bilinear",
                antialias=True,
            )

        image = tf.clip_by_value(image, 0.0, 255.0)
        image = normalize_image(image, config.normalization_mode)
        image = tf.ensure_shape(
            image,
            [
                config.image_height,
                config.image_width,
                config.channels,
            ],
        )
        label = tf.cast(label, tf.int32)
        return image, label

    return preprocess


def build_dataset(
    frame: pd.DataFrame,
    config: DataConfig,
    training: bool,
    augmenter: tf.keras.Model | None = None,
) -> tf.data.Dataset:
    paths = frame["AbsolutePath"].astype(str).to_numpy()
    labels = frame["ClassId"].astype(np.int32).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = (
        config.deterministic_pipeline
    )
    dataset = dataset.with_options(options)

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(config.shuffle_buffer, len(frame)),
            seed=config.random_seed,
            reshuffle_each_iteration=True,
        )

    preprocess = make_preprocess_function(
        config=config,
        training=training,
        augmenter=augmenter,
    )

    dataset = dataset.map(
        preprocess,
        num_parallel_calls=AUTOTUNE,
        deterministic=config.deterministic_pipeline,
    )
    dataset = dataset.batch(
        config.batch_size,
        drop_remainder=False,
    )
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def build_all_datasets(
    config: DataConfig,
) -> dict[str, Any]:
    set_global_seed(config.random_seed)

    train_frame, validation_frame, test_frame = load_all_manifests(
        config
    )
    augmenter = create_augmentation_model(config)

    train_dataset = build_dataset(
        train_frame,
        config,
        training=True,
        augmenter=augmenter,
    )
    validation_dataset = build_dataset(
        validation_frame,
        config,
        training=False,
    )
    test_dataset = build_dataset(
        test_frame,
        config,
        training=False,
    )

    class_weights = calculate_class_weights(
        train_frame,
        config.number_of_classes,
    )

    return {
        "train_frame": train_frame,
        "validation_frame": validation_frame,
        "test_frame": test_frame,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "test_dataset": test_dataset,
        "augmentation_model": augmenter,
        "class_weights": class_weights,
    }
