from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_PROJECT_ROOT = Path(
    r"D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project"
)
DEFAULT_DATASET_ROOT = DEFAULT_PROJECT_ROOT / "Datasets" / "traffic_signs"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT / "Codes" / "Data_Cleaning" / "outputs"
)

EXPECTED_CLASS_IDS = set(range(43))
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".tif", ".tiff"}

# Безопасный режим: скрипт ничего не удаляет и не перемещает.
# Сначала изучите отчёты. Только затем, при необходимости, включайте карантин.
QUARANTINE_CORRUPT_FILES = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Проверка и первичная очистка датасета GTSRB."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Папка с Meta.csv, Test.csv, Train.csv и папками Meta/Test/Train.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для CSV-отчётов и PDF-графиков.",
    )
    parser.add_argument(
        "--skip-hashes",
        action="store_true",
        help="Не считать SHA-256. Выполняется быстрее, но дубликаты не будут найдены.",
    )
    return parser.parse_args()


def normalize_relative_path(value: Any) -> str:
    """Приводит путь из CSV к единому виду с прямыми слешами."""
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/")


def resolve_dataset_path(dataset_root: Path, relative_value: Any) -> Path:
    relative = normalize_relative_path(relative_value)
    if not relative:
        return dataset_root
    return dataset_root.joinpath(*relative.split("/"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def image_quality_scores(image: Image.Image) -> tuple[float, float]:
    """
    Возвращает:
    - среднюю яркость 0..255;
    - простую оценку резкости: чем меньше число, тем вероятнее размытие.

    Это только автоматическая эвристика, а не окончательное решение.
    """
    gray = ImageOps.grayscale(image).resize((64, 64))
    array = np.asarray(gray, dtype=np.float32)
    brightness = float(array.mean())

    dx = np.diff(array, axis=1)
    dy = np.diff(array, axis=0)
    sharpness = float(dx.var() + dy.var())
    return brightness, sharpness


def read_image_metadata(path: Path, compute_hash: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": "ok",
        "error": "",
        "width": np.nan,
        "height": np.nan,
        "mode": "",
        "format": "",
        "size_bytes": path.stat().st_size if path.exists() else np.nan,
        "brightness_mean": np.nan,
        "sharpness_score": np.nan,
        "sha256": "",
    }

    try:
        # verify() проверяет структуру файла.
        with Image.open(path) as image:
            image.verify()

        # После verify() изображение нужно открыть повторно.
        with Image.open(path) as image:
            image.load()
            row["width"], row["height"] = image.size
            row["mode"] = image.mode
            row["format"] = image.format or path.suffix.lstrip(".").upper()
            brightness, sharpness = image_quality_scores(image)
            row["brightness_mean"] = brightness
            row["sharpness_score"] = sharpness

        if compute_hash:
            row["sha256"] = sha256_file(path)

    except (
        FileNotFoundError,
        PermissionError,
        OSError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        row["status"] = "corrupt_or_unreadable"
        row["error"] = f"{type(exc).__name__}: {exc}"

    return row


def load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Не найден CSV-файл: {csv_path}")
    return pd.read_csv(csv_path)


def validate_dataset_structure(dataset_root: Path) -> pd.DataFrame:
    checks = []

    required_items = {
        "Meta.csv": dataset_root / "Meta.csv",
        "Test.csv": dataset_root / "Test.csv",
        "Train.csv": dataset_root / "Train.csv",
        "Meta folder": dataset_root / "Meta",
        "Test folder": dataset_root / "Test",
        "Train folder": dataset_root / "Train",
    }

    for name, path in required_items.items():
        checks.append(
            {
                "item": name,
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file(),
                "is_directory": path.is_dir(),
            }
        )

    train_root = dataset_root / "Train"
    actual_class_folders = {
        int(folder.name)
        for folder in train_root.iterdir()
        if folder.is_dir() and folder.name.isdigit()
    } if train_root.exists() else set()

    checks.append(
        {
            "item": "Train class folders 0..42",
            "path": str(train_root),
            "exists": actual_class_folders == EXPECTED_CLASS_IDS,
            "is_file": False,
            "is_directory": True,
        }
    )

    return pd.DataFrame(checks)


def build_csv_class_maps(
    dataset_root: Path,
    train_csv: pd.DataFrame,
    test_csv: pd.DataFrame,
    meta_csv: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    def build_map(frame: pd.DataFrame) -> dict[str, int]:
        if "Path" not in frame.columns or "ClassId" not in frame.columns:
            return {}
        result: dict[str, int] = {}
        for _, row in frame[["Path", "ClassId"]].dropna().iterrows():
            key = normalize_relative_path(row["Path"]).lower()
            try:
                result[key] = int(row["ClassId"])
            except (TypeError, ValueError):
                continue
        return result

    return build_map(train_csv), build_map(test_csv), build_map(meta_csv)


def infer_class_id(
    split: str,
    relative_path: str,
    train_map: dict[str, int],
    test_map: dict[str, int],
    meta_map: dict[str, int],
) -> int | None:
    key = relative_path.lower()

    csv_map = {
        "Train": train_map,
        "Test": test_map,
        "Meta": meta_map,
    }.get(split, {})

    if key in csv_map:
        return csv_map[key]

    path = Path(relative_path)
    if split == "Train" and len(path.parts) >= 2:
        parent = path.parent.name
        if parent.isdigit():
            return int(parent)

    if split == "Meta" and path.stem.isdigit():
        return int(path.stem)

    return None


def scan_images(
    dataset_root: Path,
    output_dir: Path,
    train_map: dict[str, int],
    test_map: dict[str, int],
    meta_map: dict[str, int],
    compute_hashes: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ignored_rows: list[dict[str, str]] = []

    for split in ("Meta", "Train", "Test"):
        split_root = dataset_root / split
        if not split_root.exists():
            continue

        all_files = sorted(path for path in split_root.rglob("*") if path.is_file())

        for index, path in enumerate(all_files, start=1):
            relative_path = path.relative_to(dataset_root).as_posix()

            # LibreOffice создаёт временные lock-файлы вида .~lock...#.
            if path.name.startswith(".~lock.") and path.name.endswith("#"):
                ignored_rows.append(
                    {
                        "split": split,
                        "relative_path": relative_path,
                        "reason": "LibreOffice temporary lock file",
                    }
                )
                continue

            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                ignored_rows.append(
                    {
                        "split": split,
                        "relative_path": relative_path,
                        "reason": "Unsupported or non-image file",
                    }
                )
                continue

            metadata = read_image_metadata(path, compute_hash=compute_hashes)
            class_id = infer_class_id(
                split,
                relative_path,
                train_map,
                test_map,
                meta_map,
            )

            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "relative_path": relative_path,
                    "absolute_path": str(path),
                    **metadata,
                }
            )

            if index % 2000 == 0:
                print(f"[{split}] Проверено файлов: {index}/{len(all_files)}")

    ignored_df = pd.DataFrame(ignored_rows)
    ignored_path = output_dir / "reports" / "ignored_non_image_files.csv"
    ignored_path.parent.mkdir(parents=True, exist_ok=True)
    ignored_df.to_csv(ignored_path, index=False, encoding="utf-8-sig")

    return pd.DataFrame(rows)


def validate_csv_records(
    dataset_root: Path,
    csv_name: str,
    frame: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    issues: list[dict[str, Any]] = []

    required_columns = {"Path", "ClassId"}
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        issues.append(
            {
                "csv_file": csv_name,
                "row_number": "",
                "relative_path": "",
                "issue": f"Missing required columns: {missing_columns}",
            }
        )
        return pd.DataFrame(issues)

    inventory_by_path = {
        path.lower(): row
        for path, row in inventory.set_index("relative_path").iterrows()
    }

    normalized_paths = frame["Path"].map(normalize_relative_path)
    duplicate_mask = normalized_paths.str.lower().duplicated(keep=False)

    for row_index, row in frame.iterrows():
        relative_path = normalize_relative_path(row.get("Path"))
        absolute_path = resolve_dataset_path(dataset_root, relative_path)

        if not relative_path:
            issues.append(
                {
                    "csv_file": csv_name,
                    "row_number": row_index + 2,
                    "relative_path": "",
                    "issue": "Empty Path value",
                }
            )
            continue

        if not absolute_path.exists():
            issues.append(
                {
                    "csv_file": csv_name,
                    "row_number": row_index + 2,
                    "relative_path": relative_path,
                    "issue": "File listed in CSV does not exist",
                }
            )
            continue

        if bool(duplicate_mask.iloc[row_index]):
            issues.append(
                {
                    "csv_file": csv_name,
                    "row_number": row_index + 2,
                    "relative_path": relative_path,
                    "issue": "Duplicate Path entry in CSV",
                }
            )

        try:
            class_id = int(row["ClassId"])
            if class_id not in EXPECTED_CLASS_IDS:
                issues.append(
                    {
                        "csv_file": csv_name,
                        "row_number": row_index + 2,
                        "relative_path": relative_path,
                        "issue": f"ClassId outside 0..42: {class_id}",
                    }
                )
        except (TypeError, ValueError):
            issues.append(
                {
                    "csv_file": csv_name,
                    "row_number": row_index + 2,
                    "relative_path": relative_path,
                    "issue": f"Invalid ClassId: {row['ClassId']}",
                }
            )

        inventory_row = inventory_by_path.get(relative_path.lower())
        if inventory_row is not None:
            actual_width = inventory_row.get("width")
            actual_height = inventory_row.get("height")

            if "Width" in frame.columns and pd.notna(row.get("Width")):
                if int(row["Width"]) != int(actual_width):
                    issues.append(
                        {
                            "csv_file": csv_name,
                            "row_number": row_index + 2,
                            "relative_path": relative_path,
                            "issue": (
                                f"Width mismatch: CSV={row['Width']}, "
                                f"actual={actual_width}"
                            ),
                        }
                    )

            if "Height" in frame.columns and pd.notna(row.get("Height")):
                if int(row["Height"]) != int(actual_height):
                    issues.append(
                        {
                            "csv_file": csv_name,
                            "row_number": row_index + 2,
                            "relative_path": relative_path,
                            "issue": (
                                f"Height mismatch: CSV={row['Height']}, "
                                f"actual={actual_height}"
                            ),
                        }
                    )

            roi_columns = {"Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"}
            if roi_columns.issubset(frame.columns):
                try:
                    x1 = int(row["Roi.X1"])
                    y1 = int(row["Roi.Y1"])
                    x2 = int(row["Roi.X2"])
                    y2 = int(row["Roi.Y2"])
                    width = int(actual_width)
                    height = int(actual_height)

                    valid_roi = (
                        0 <= x1 < x2 <= width
                        and 0 <= y1 < y2 <= height
                    )
                    if not valid_roi:
                        issues.append(
                            {
                                "csv_file": csv_name,
                                "row_number": row_index + 2,
                                "relative_path": relative_path,
                                "issue": (
                                    "Invalid ROI coordinates: "
                                    f"({x1}, {y1}, {x2}, {y2}) "
                                    f"for image {width}x{height}"
                                ),
                            }
                        )
                except (TypeError, ValueError):
                    issues.append(
                        {
                            "csv_file": csv_name,
                            "row_number": row_index + 2,
                            "relative_path": relative_path,
                            "issue": "ROI contains non-integer values",
                        }
                    )

    return pd.DataFrame(issues)


def compare_train_folder_and_csv(
    train_csv: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    folder_counts = (
        inventory.query("split == 'Train' and status == 'ok'")
        .groupby("class_id", dropna=False)
        .size()
        .rename("folder_image_count")
    )

    csv_counts = (
        train_csv.groupby("ClassId")
        .size()
        .rename("csv_row_count")
        if "ClassId" in train_csv.columns
        else pd.Series(dtype=int, name="csv_row_count")
    )

    comparison = pd.concat([folder_counts, csv_counts], axis=1).fillna(0)
    comparison.index.name = "class_id"
    comparison = comparison.reset_index()
    comparison["difference_folder_minus_csv"] = (
        comparison["folder_image_count"] - comparison["csv_row_count"]
    )
    return comparison.sort_values("class_id")


def create_duplicate_reports(
    inventory: pd.DataFrame,
    reports_dir: Path,
    hashes_enabled: bool,
) -> None:
    if not hashes_enabled:
        pd.DataFrame(
            [{"message": "Hash calculation was skipped."}]
        ).to_csv(
            reports_dir / "duplicate_groups.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return

    valid = inventory[
        (inventory["status"] == "ok")
        & inventory["sha256"].astype(bool)
    ].copy()

    duplicate_hashes = (
        valid.groupby("sha256")
        .size()
        .loc[lambda series: series > 1]
        .index
    )

    duplicates = valid[valid["sha256"].isin(duplicate_hashes)].copy()
    if duplicates.empty:
        duplicates.to_csv(
            reports_dir / "duplicate_groups.csv",
            index=False,
            encoding="utf-8-sig",
        )
        duplicates.to_csv(
            reports_dir / "cross_split_duplicates.csv",
            index=False,
            encoding="utf-8-sig",
        )
        return

    duplicates["duplicate_group_size"] = duplicates.groupby("sha256")[
        "sha256"
    ].transform("size")
    duplicates["splits_in_group"] = duplicates.groupby("sha256")[
        "split"
    ].transform(lambda values: ",".join(sorted(set(values))))

    duplicates = duplicates.sort_values(["sha256", "split", "relative_path"])
    duplicates.to_csv(
        reports_dir / "duplicate_groups.csv",
        index=False,
        encoding="utf-8-sig",
    )

    cross_split = duplicates[
        duplicates["splits_in_group"].str.contains(",")
    ].copy()
    cross_split.to_csv(
        reports_dir / "cross_split_duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )


def save_class_distribution_plot(
    train_distribution: pd.DataFrame,
    figure_path: Path,
) -> None:
    if train_distribution.empty:
        return

    plt.figure(figsize=(12, 6))
    plt.bar(
        train_distribution["class_id"].astype(int),
        train_distribution["folder_image_count"],
    )
    plt.xlabel("Class ID")
    plt.ylabel("Number of training images")
    plt.title("GTSRB training-class distribution")
    plt.xticks(range(43))
    plt.tight_layout()
    plt.savefig(figure_path, format="pdf", bbox_inches="tight")
    plt.close()


def save_image_size_scatter(inventory: pd.DataFrame, figure_path: Path) -> None:
    valid = inventory[
        (inventory["status"] == "ok")
        & inventory["width"].notna()
        & inventory["height"].notna()
    ]
    if valid.empty:
        return

    plt.figure(figsize=(8, 6))
    plt.scatter(
        valid["width"],
        valid["height"],
        s=8,
        alpha=0.35,
    )
    plt.xlabel("Image width, pixels")
    plt.ylabel("Image height, pixels")
    plt.title("Image-size distribution")
    plt.tight_layout()
    plt.savefig(figure_path, format="pdf", bbox_inches="tight")
    plt.close()


def safe_open_for_plot(path: str | Path) -> Image.Image | None:
    try:
        with Image.open(path) as image:
            return image.convert("RGB").copy()
    except Exception:
        return None


def save_class_samples_pdf(
    inventory: pd.DataFrame,
    figure_path: Path,
) -> None:
    train_valid = inventory[
        (inventory["split"] == "Train")
        & (inventory["status"] == "ok")
        & inventory["class_id"].notna()
    ].copy()

    samples = (
        train_valid.sort_values("relative_path")
        .groupby("class_id", as_index=False)
        .first()
        .sort_values("class_id")
    )

    if samples.empty:
        return

    per_page = 20
    with PdfPages(figure_path) as pdf:
        for start in range(0, len(samples), per_page):
            page = samples.iloc[start : start + per_page]
            columns = 5
            rows = math.ceil(len(page) / columns)
            figure, axes = plt.subplots(rows, columns, figsize=(11.7, 8.3))
            axes_array = np.atleast_1d(axes).ravel()

            for axis in axes_array:
                axis.axis("off")

            for axis, (_, row) in zip(axes_array, page.iterrows()):
                image = safe_open_for_plot(row["absolute_path"])
                if image is not None:
                    axis.imshow(image)
                axis.set_title(f"Class {int(row['class_id'])}", fontsize=9)
                axis.axis("off")

            figure.suptitle("One representative training image per class")
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def save_challenging_samples_pdf(
    inventory: pd.DataFrame,
    figure_path: Path,
) -> None:
    valid = inventory[
        (inventory["split"].isin(["Train", "Test"]))
        & (inventory["status"] == "ok")
    ].copy()

    if valid.empty:
        return

    valid["area"] = valid["width"] * valid["height"]

    groups = [
        ("Smallest images", valid.nsmallest(12, "area")),
        ("Darkest images (automatic estimate)", valid.nsmallest(12, "brightness_mean")),
        ("Lowest sharpness score (possible blur)", valid.nsmallest(12, "sharpness_score")),
    ]

    with PdfPages(figure_path) as pdf:
        for title, frame in groups:
            figure, axes = plt.subplots(3, 4, figsize=(11.7, 8.3))
            for axis, (_, row) in zip(axes.ravel(), frame.iterrows()):
                image = safe_open_for_plot(row["absolute_path"])
                if image is not None:
                    axis.imshow(image)
                class_text = (
                    "unknown"
                    if pd.isna(row["class_id"])
                    else str(int(row["class_id"]))
                )
                axis.set_title(
                    f"{row['split']} | class {class_text}\n"
                    f"{int(row['width'])}x{int(row['height'])}",
                    fontsize=8,
                )
                axis.axis("off")

            for axis in axes.ravel()[len(frame) :]:
                axis.axis("off")

            figure.suptitle(title)
            figure.tight_layout()
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def quarantine_corrupt_files(
    inventory: pd.DataFrame,
    output_dir: Path,
) -> None:
    if not QUARANTINE_CORRUPT_FILES:
        return

    quarantine_root = output_dir / "quarantine"
    corrupt = inventory[inventory["status"] != "ok"]

    for _, row in corrupt.iterrows():
        source = Path(row["absolute_path"])
        if not source.exists():
            continue

        destination = quarantine_root / row["relative_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))


def save_summary(
    dataset_root: Path,
    inventory: pd.DataFrame,
    structure: pd.DataFrame,
    csv_issues: pd.DataFrame,
    output_path: Path,
    hashes_enabled: bool,
) -> None:
    good = inventory[inventory["status"] == "ok"]
    bad = inventory[inventory["status"] != "ok"]

    lines = [
        "GTSRB DATASET INSPECTION SUMMARY",
        "=" * 40,
        f"Dataset root: {dataset_root}",
        f"Images found: {len(inventory):,}",
        f"Readable images: {len(good):,}",
        f"Corrupt/unreadable images: {len(bad):,}",
        f"CSV issues: {len(csv_issues):,}",
        f"SHA-256 duplicate check enabled: {hashes_enabled}",
        "",
        "Images by split:",
    ]

    split_counts = good.groupby("split").size().to_dict()
    for split in ("Meta", "Train", "Test"):
        lines.append(f"  {split}: {split_counts.get(split, 0):,}")

    lines.extend(
        [
            "",
            "Structure checks:",
        ]
    )
    for _, row in structure.iterrows():
        status = "OK" if bool(row["exists"]) else "PROBLEM"
        lines.append(f"  {status}: {row['item']}")

    lines.extend(
        [
            "",
            "Important:",
            "1. The script does not delete files in safe mode.",
            "2. LibreOffice .~lock...# files are ignored.",
            "3. Review CSV reports before changing the dataset.",
            "4. Cross-split duplicates may indicate data leakage.",
            "5. Low sharpness/brightness flags are only heuristics.",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset_root}")
    print(f"Output:  {output_dir}")

    structure = validate_dataset_structure(dataset_root)
    structure.to_csv(
        reports_dir / "dataset_structure.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not structure["exists"].all():
        print(
            "Ошибка: структура датасета неполная. "
            "Смотрите outputs/reports/dataset_structure.csv"
        )
        return 1

    train_csv = load_csv(dataset_root / "Train.csv")
    test_csv = load_csv(dataset_root / "Test.csv")
    meta_csv = load_csv(dataset_root / "Meta.csv")

    train_map, test_map, meta_map = build_csv_class_maps(
        dataset_root,
        train_csv,
        test_csv,
        meta_csv,
    )

    inventory = scan_images(
        dataset_root=dataset_root,
        output_dir=output_dir,
        train_map=train_map,
        test_map=test_map,
        meta_map=meta_map,
        compute_hashes=not args.skip_hashes,
    )
    inventory.to_csv(
        reports_dir / "image_inventory.csv",
        index=False,
        encoding="utf-8-sig",
    )

    corrupt = inventory[inventory["status"] != "ok"].copy()
    corrupt.to_csv(
        reports_dir / "corrupt_or_unreadable_images.csv",
        index=False,
        encoding="utf-8-sig",
    )

    csv_issue_frames = []
    for csv_name, frame in [
        ("Train.csv", train_csv),
        ("Test.csv", test_csv),
        ("Meta.csv", meta_csv),
    ]:
        csv_issue_frames.append(
            validate_csv_records(
                dataset_root=dataset_root,
                csv_name=csv_name,
                frame=frame,
                inventory=inventory,
            )
        )

    csv_issues = pd.concat(csv_issue_frames, ignore_index=True)
    csv_issues.to_csv(
        reports_dir / "csv_consistency_issues.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_distribution = compare_train_folder_and_csv(train_csv, inventory)
    train_distribution.to_csv(
        reports_dir / "train_class_distribution.csv",
        index=False,
        encoding="utf-8-sig",
    )

    create_duplicate_reports(
        inventory=inventory,
        reports_dir=reports_dir,
        hashes_enabled=not args.skip_hashes,
    )

    save_class_distribution_plot(
        train_distribution,
        figures_dir / "train_class_distribution.pdf",
    )
    save_image_size_scatter(
        inventory,
        figures_dir / "image_size_distribution.pdf",
    )
    save_class_samples_pdf(
        inventory,
        figures_dir / "sample_images_by_class.pdf",
    )
    save_challenging_samples_pdf(
        inventory,
        figures_dir / "challenging_image_examples.pdf",
    )

    quarantine_corrupt_files(inventory, output_dir)

    save_summary(
        dataset_root=dataset_root,
        inventory=inventory,
        structure=structure,
        csv_issues=csv_issues,
        output_path=output_dir / "SUMMARY.txt",
        hashes_enabled=not args.skip_hashes,
    )

    print("\nПроверка завершена.")
    print(f"Главный итог: {output_dir / 'SUMMARY.txt'}")
    print(f"CSV-отчёты:  {reports_dir}")
    print(f"PDF-фигуры:  {figures_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
