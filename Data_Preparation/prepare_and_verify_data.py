from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from config import DataConfig
from data_pipeline import (
    build_all_datasets,
    denormalize_for_display,
    save_class_weights,
)


def save_split_counts(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    pieces = []
    for split_name, frame in (
        ("train", train),
        ("validation", validation),
        ("test", test),
    ):
        counts = (
            frame.groupby("ClassId")
            .size()
            .rename("image_count")
            .reset_index()
        )
        counts["split"] = split_name
        pieces.append(counts)

    result = pd.concat(pieces, ignore_index=True)
    result = result[
        ["split", "ClassId", "image_count"]
    ].sort_values(["split", "ClassId"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def assert_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    config: DataConfig,
    split_name: str,
) -> list[str]:
    messages = []

    if images.shape.rank != 4:
        raise AssertionError(
            f"{split_name}: images rank={images.shape.rank}, ожидался 4"
        )

    if tuple(images.shape[1:]) != (
        config.image_height,
        config.image_width,
        config.channels,
    ):
        raise AssertionError(
            f"{split_name}: неправильная форма {images.shape}"
        )

    if labels.shape.rank != 1:
        raise AssertionError(
            f"{split_name}: labels должны быть одномерными, "
            f"получено {labels.shape}"
        )

    minimum = float(tf.reduce_min(images).numpy())
    maximum = float(tf.reduce_max(images).numpy())

    if config.normalization_mode == "zero_one":
        valid_range = -1e-6 <= minimum and maximum <= 1.0 + 1e-6
        expected_range = "[0, 1]"
    elif config.normalization_mode == "minus_one_one":
        valid_range = -1.0 - 1e-6 <= minimum and maximum <= 1.0 + 1e-6
        expected_range = "[-1, 1]"
    else:
        valid_range = -1e-6 <= minimum and maximum <= 255.0 + 1e-6
        expected_range = "[0, 255]"

    if not valid_range:
        raise AssertionError(
            f"{split_name}: диапазон пикселей {minimum:.4f}.."
            f"{maximum:.4f}, ожидался {expected_range}"
        )

    label_min = int(tf.reduce_min(labels).numpy())
    label_max = int(tf.reduce_max(labels).numpy())
    if label_min < 0 or label_max >= config.number_of_classes:
        raise AssertionError(
            f"{split_name}: метки {label_min}..{label_max}"
        )

    messages.extend(
        [
            f"{split_name} batch shape: {tuple(images.shape)}",
            f"{split_name} pixel range: {minimum:.6f} .. {maximum:.6f}",
            f"{split_name} label range in batch: {label_min}..{label_max}",
        ]
    )
    return messages


def save_batch_figure(
    dataset: tf.data.Dataset,
    config: DataConfig,
    output_pdf: Path,
    output_png: Path,
    title: str,
    max_images: int = 16,
) -> None:
    images, labels = next(iter(dataset))
    count = min(max_images, int(images.shape[0]))

    columns = 4
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(11.7, 8.3))
    axes = np.asarray(axes).reshape(-1)

    for axis in axes:
        axis.axis("off")

    for index in range(count):
        display_image = denormalize_for_display(
            images[index].numpy(),
            config.normalization_mode,
        )
        axes[index].imshow(display_image)
        axes[index].set_title(
            f"ClassId: {int(labels[index].numpy())}",
            fontsize=9,
        )
        axes[index].axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(figure)


def save_augmentation_comparison(
    train_frame: pd.DataFrame,
    augmenter: tf.keras.Model,
    config: DataConfig,
    output_pdf: Path,
    output_png: Path,
) -> None:
    selected_classes = [0, 2, 12, 14]
    selected_rows = []

    for class_id in selected_classes:
        rows = train_frame[train_frame["ClassId"] == class_id]
        if not rows.empty:
            selected_rows.append(rows.iloc[0])

    columns = 4
    rows_count = len(selected_rows)
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(11.7, 2.5 * rows_count),
    )
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    for row_index, row in enumerate(selected_rows):
        image_bytes = tf.io.read_file(row["AbsolutePath"])
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(
            image,
            (config.augmentation_resize, config.augmentation_resize),
            antialias=True,
        )

        original = tf.image.resize(
            image,
            (config.image_height, config.image_width),
            antialias=True,
        )
        axes[row_index, 0].imshow(
            np.clip(original.numpy(), 0, 255).astype(np.uint8)
        )
        axes[row_index, 0].set_title(
            f"Original\nClassId {int(row['ClassId'])}"
        )
        axes[row_index, 0].axis("off")

        for column_index in range(1, columns):
            augmented = augmenter(
                tf.expand_dims(image, axis=0),
                training=True,
            )
            augmented = tf.squeeze(augmented, axis=0)
            axes[row_index, column_index].imshow(
                np.clip(augmented.numpy(), 0, 255).astype(np.uint8)
            )
            axes[row_index, column_index].set_title(
                f"Augmented {column_index}"
            )
            axes[row_index, column_index].axis("off")

    figure.suptitle(
        "Traffic-sign augmentation examples "
        "(crop, rotation, shift, zoom, brightness, contrast)"
    )
    figure.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(figure)


def benchmark_dataset(
    dataset: tf.data.Dataset,
    number_of_batches: int = 10,
) -> tuple[int, float, float]:
    start = time.perf_counter()
    images_seen = 0
    batches_seen = 0

    for images, _ in dataset.take(number_of_batches):
        _ = float(tf.reduce_mean(images).numpy())
        images_seen += int(images.shape[0])
        batches_seen += 1

    elapsed = time.perf_counter() - start
    speed = images_seen / elapsed if elapsed > 0 else 0.0
    return batches_seen, elapsed, speed


def save_report(
    config: DataConfig,
    data: dict,
    batch_messages: list[str],
    benchmark_messages: list[str],
    output_path: Path,
) -> None:
    train = data["train_frame"]
    validation = data["validation_frame"]
    test = data["test_frame"]
    class_weights = data["class_weights"]

    train_paths = set(train["AbsolutePath"].str.lower())
    validation_paths = set(validation["AbsolutePath"].str.lower())
    test_paths = set(test["AbsolutePath"].str.lower())

    lines = [
        "GTSRB DATA PREPARATION REPORT",
        "=" * 42,
        f"Python: {sys.version.split()[0]}",
        f"Platform: {platform.platform()}",
        f"TensorFlow: {tf.__version__}",
        f"Physical devices: {tf.config.list_physical_devices()}",
        "",
        f"Image shape: {config.image_height}x{config.image_width}x{config.channels}",
        f"Batch size: {config.batch_size}",
        f"Number of classes: {config.number_of_classes}",
        f"Normalization: {config.normalization_mode}",
        f"Random seed: {config.random_seed}",
        "",
        f"Training images: {len(train):,}",
        f"Validation images: {len(validation):,}",
        f"Test images: {len(test):,}",
        "",
        f"Train/validation path overlap: {len(train_paths & validation_paths)}",
        f"Train/test path overlap: {len(train_paths & test_paths)}",
        f"Validation/test path overlap: {len(validation_paths & test_paths)}",
        "",
        "Batch checks:",
        *[f"  {message}" for message in batch_messages],
        "",
        "Pipeline benchmark:",
        *[f"  {message}" for message in benchmark_messages],
        "",
        "Class weights:",
    ]

    for class_id, weight in sorted(class_weights.items()):
        lines.append(f"  {class_id}: {weight:.6f}")

    lines.extend(
        [
            "",
            "Decisions:",
            "1. Train receives augmentation; validation and test do not.",
            "2. Images are resized to 224x224 RGB.",
            "3. No horizontal or vertical flips are used.",
            "4. The official test set remains untouched.",
            "5. Images are processed on the fly; no enlarged copy is saved.",
            "6. Labels remain integer ClassId values for sparse loss.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    config = DataConfig()
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.figure_root.mkdir(parents=True, exist_ok=True)
    config.report_root.mkdir(parents=True, exist_ok=True)

    print("Загрузка manifest-файлов и создание tf.data pipeline...")
    data = build_all_datasets(config)

    config_path = config.report_root / "data_config.json"
    config_path.write_text(
        json.dumps(
            config.to_serializable_dict(),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    save_class_weights(
        data["class_weights"],
        config.report_root / "class_weights.json",
    )

    save_split_counts(
        data["train_frame"],
        data["validation_frame"],
        data["test_frame"],
        config.report_root / "class_counts_by_split.csv",
    )

    batch_messages: list[str] = []
    for split_name, dataset in (
        ("train", data["train_dataset"]),
        ("validation", data["validation_dataset"]),
        ("test", data["test_dataset"]),
    ):
        images, labels = next(iter(dataset))
        batch_messages.extend(
            assert_batch(images, labels, config, split_name)
        )

    save_batch_figure(
        data["train_dataset"],
        config,
        config.figure_root / "training_batch_after_augmentation.pdf",
        config.figure_root / "training_batch_after_augmentation.png",
        "Training batch after preprocessing and augmentation",
    )

    save_batch_figure(
        data["validation_dataset"],
        config,
        config.figure_root / "validation_batch_preprocessed.pdf",
        config.figure_root / "validation_batch_preprocessed.png",
        "Validation batch after deterministic preprocessing",
    )

    save_augmentation_comparison(
        data["train_frame"],
        data["augmentation_model"],
        config,
        config.figure_root / "augmentation_comparison.pdf",
        config.figure_root / "augmentation_comparison.png",
    )

    benchmark_messages: list[str] = []
    for split_name, dataset in (
        ("train", data["train_dataset"]),
        ("validation", data["validation_dataset"]),
        ("test", data["test_dataset"]),
    ):
        batches, seconds, images_per_second = benchmark_dataset(
            dataset,
            number_of_batches=10,
        )
        benchmark_messages.append(
            f"{split_name}: {batches} batches in {seconds:.3f} s; "
            f"{images_per_second:.1f} images/s"
        )

    save_report(
        config=config,
        data=data,
        batch_messages=batch_messages,
        benchmark_messages=benchmark_messages,
        output_path=config.report_root / "data_pipeline_report.txt",
    )

    print("\nПроверка завершена.")
    print(f"Отчёты: {config.report_root}")
    print(f"Фигуры: {config.figure_root}")
    print("\nОсновные файлы:")
    print(config.report_root / "data_pipeline_report.txt")
    print(config.report_root / "class_weights.json")
    print(config.figure_root / "augmentation_comparison.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
