from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(
    r"D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project"
)
DATASET_ROOT = PROJECT_ROOT / "Datasets" / "traffic_signs"
DATA_CLEANING_ROOT = PROJECT_ROOT / "Codes" / "Data_Cleaning"

TRAIN_CSV = DATASET_ROOT / "Train.csv"
DUPLICATE_REPORT = (
    DATA_CLEANING_ROOT
    / "outputs"
    / "reports"
    / "cross_split_duplicates.csv"
)
OUTPUT_DIR = DATA_CLEANING_ROOT / "prepared_data"

VALIDATION_RATIO = 0.20
RANDOM_SEED = 42
EXPECTED_CLASSES = set(range(43))


def normalize_path(value: object) -> str:
    """Единый вид путей для надёжного сравнения."""
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.lstrip("/").lower()


def make_group_id(path_value: object, class_id: int) -> str:
    """
    GTSRB обычно использует имя:
    00014_00023_00010.png

    00014 = класс
    00023 = одна серия/один физический знак
    00010 = кадр внутри серии

    Все кадры одной серии должны попасть только в train
    или только в validation.
    """
    filename = Path(str(path_value).replace("\\", "/")).stem
    parts = filename.split("_")

    if len(parts) >= 3:
        sequence_id = parts[1]
        return f"class_{class_id:02d}_sequence_{sequence_id}"

    # Безопасный запасной вариант для нестандартного имени.
    return f"class_{class_id:02d}_file_{filename}"


def choose_validation_groups(
    class_frame: pd.DataFrame,
    validation_ratio: float,
    rng: np.random.Generator,
) -> set[str]:
    groups = np.array(sorted(class_frame["GroupId"].unique()), dtype=object)

    if len(groups) < 2:
        raise ValueError(
            f"Для класса {class_frame['ClassId'].iloc[0]} найдено "
            "меньше двух независимых групп."
        )

    rng.shuffle(groups)

    number_for_validation = round(len(groups) * validation_ratio)
    number_for_validation = max(1, number_for_validation)
    number_for_validation = min(len(groups) - 1, number_for_validation)

    return set(groups[:number_for_validation])


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_CSV.exists():
        print(f"Не найден файл: {TRAIN_CSV}")
        return 1

    if not DUPLICATE_REPORT.exists():
        print(f"Не найден отчёт о дубликатах: {DUPLICATE_REPORT}")
        print("Сначала запустите inspect_and_clean_dataset.py")
        return 1

    train = pd.read_csv(TRAIN_CSV)
    duplicates = pd.read_csv(DUPLICATE_REPORT)

    required_train_columns = {"Path", "ClassId"}
    missing = required_train_columns - set(train.columns)
    if missing:
        print(f"В Train.csv отсутствуют колонки: {sorted(missing)}")
        return 1

    train["_NormalizedPath"] = train["Path"].map(normalize_path)

    duplicate_train_paths = set(
        duplicates.loc[
            duplicates["split"].astype(str).str.lower() == "train",
            "relative_path",
        ].map(normalize_path)
    )

    excluded = train[
        train["_NormalizedPath"].isin(duplicate_train_paths)
    ].copy()

    clean = train[
        ~train["_NormalizedPath"].isin(duplicate_train_paths)
    ].copy()

    found_duplicate_paths = set(excluded["_NormalizedPath"])
    missing_from_train = duplicate_train_paths - found_duplicate_paths

    if missing_from_train:
        print("Не все пути из отчёта найдены в Train.csv:")
        for path in sorted(missing_from_train):
            print(f"  {path}")
        return 1

    # Сохраняем список исключённых файлов. Исходные изображения не удаляются.
    excluded.drop(columns=["_NormalizedPath"]).to_csv(
        OUTPUT_DIR / "excluded_train_duplicates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    clean_without_helper = clean.drop(columns=["_NormalizedPath"])
    clean_without_helper.to_csv(
        OUTPUT_DIR / "Train_clean.csv",
        index=False,
        encoding="utf-8-sig",
    )

    clean["ClassId"] = clean["ClassId"].astype(int)
    clean["GroupId"] = clean.apply(
        lambda row: make_group_id(row["Path"], int(row["ClassId"])),
        axis=1,
    )

    actual_classes = set(clean["ClassId"].unique())
    if actual_classes != EXPECTED_CLASSES:
        print(
            "После очистки набор классов не равен 0..42.\n"
            f"Найдены классы: {sorted(actual_classes)}"
        )
        return 1

    rng = np.random.default_rng(RANDOM_SEED)
    split_parts: list[pd.DataFrame] = []

    for class_id, class_frame in clean.groupby("ClassId", sort=True):
        validation_groups = choose_validation_groups(
            class_frame=class_frame,
            validation_ratio=VALIDATION_RATIO,
            rng=rng,
        )

        class_part = class_frame.copy()
        class_part["Split"] = np.where(
            class_part["GroupId"].isin(validation_groups),
            "validation",
            "train",
        )
        split_parts.append(class_part)

    split_data = pd.concat(split_parts, ignore_index=True)

    train_manifest = split_data[
        split_data["Split"] == "train"
    ].copy()
    validation_manifest = split_data[
        split_data["Split"] == "validation"
    ].copy()

    output_columns = [
        column
        for column in train.columns
        if column != "_NormalizedPath"
    ] + ["GroupId"]

    train_manifest[output_columns].to_csv(
        OUTPUT_DIR / "train_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation_manifest[output_columns].to_csv(
        OUTPUT_DIR / "validation_manifest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        split_data.groupby(["ClassId", "Split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for column in ("train", "validation"):
        if column not in summary.columns:
            summary[column] = 0

    summary["total"] = summary["train"] + summary["validation"]
    summary["validation_percent"] = (
        summary["validation"] / summary["total"] * 100
    ).round(2)

    summary.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    train_paths = set(train_manifest["_NormalizedPath"])
    validation_paths = set(validation_manifest["_NormalizedPath"])
    train_groups = set(train_manifest["GroupId"])
    validation_groups = set(validation_manifest["GroupId"])

    path_overlap = train_paths & validation_paths
    group_overlap = train_groups & validation_groups

    train_classes = set(train_manifest["ClassId"].unique())
    validation_classes = set(validation_manifest["ClassId"].unique())

    checks = [
        f"Original Train.csv rows: {len(train):,}",
        f"Excluded exact Train/Test duplicates: {len(excluded):,}",
        f"Clean rows: {len(clean):,}",
        f"Training rows: {len(train_manifest):,}",
        f"Validation rows: {len(validation_manifest):,}",
        (
            "Validation share: "
            f"{len(validation_manifest) / len(clean) * 100:.2f}%"
        ),
        f"Training groups: {len(train_groups):,}",
        f"Validation groups: {len(validation_groups):,}",
        f"Path overlap: {len(path_overlap)}",
        f"Group overlap: {len(group_overlap)}",
        f"All 43 classes in training: {train_classes == EXPECTED_CLASSES}",
        (
            "All 43 classes in validation: "
            f"{validation_classes == EXPECTED_CLASSES}"
        ),
        "Official Test.csv was not modified.",
        "Original image files were not deleted or moved.",
    ]

    (OUTPUT_DIR / "split_integrity_checks.txt").write_text(
        "\n".join(checks),
        encoding="utf-8",
    )

    if path_overlap or group_overlap:
        print("Ошибка: обнаружено пересечение между train и validation.")
        return 1

    print("\n".join(checks))
    print(f"\nРезультаты сохранены в:\n{OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
