from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


NormalizationMode = Literal["zero_one", "minus_one_one", "none"]


@dataclass(frozen=True)
class DataConfig:
    project_root: Path = Path(
        r"D:\SoftwareEngineering\Machine Learning\Traffic_Sign_Project"
    )

    image_height: int = 224
    image_width: int = 224
    augmentation_resize: int = 240
    channels: int = 3
    number_of_classes: int = 43

    batch_size: int = 32
    shuffle_buffer: int = 10_000
    random_seed: int = 42

    # Для Custom CNN удобно получать пиксели в диапазоне 0..1.
    # Для transfer learning позже можно установить "none", если
    # нормализация будет встроена непосредственно в модель.
    normalization_mode: NormalizationMode = "zero_one"

    # Аугментация применяется только к training.
    rotation_factor: float = 0.03       # примерно ±10.8 градусов
    translation_factor: float = 0.05
    zoom_factor: float = 0.08
    brightness_factor: float = 0.12
    contrast_factor: float = 0.12

    noise_probability: float = 0.15
    noise_stddev_255: float = 4.0
    blur_probability: float = 0.15

    deterministic_pipeline: bool = True

    @property
    def dataset_root(self) -> Path:
        return self.project_root / "Datasets" / "traffic_signs"

    @property
    def data_cleaning_root(self) -> Path:
        return self.project_root / "Codes" / "Data_Cleaning"

    @property
    def prepared_data_root(self) -> Path:
        return self.data_cleaning_root / "prepared_data"

    @property
    def train_manifest_path(self) -> Path:
        return self.prepared_data_root / "train_manifest.csv"

    @property
    def validation_manifest_path(self) -> Path:
        return self.prepared_data_root / "validation_manifest.csv"

    @property
    def test_manifest_path(self) -> Path:
        return self.dataset_root / "Test.csv"

    @property
    def code_root(self) -> Path:
        return self.project_root / "Codes" / "Data_Preparation"

    @property
    def output_root(self) -> Path:
        return self.code_root / "outputs"

    @property
    def figure_root(self) -> Path:
        return self.output_root / "figures"

    @property
    def report_root(self) -> Path:
        return self.output_root / "reports"

    def to_serializable_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["project_root"] = str(self.project_root)
        result["dataset_root"] = str(self.dataset_root)
        result["prepared_data_root"] = str(self.prepared_data_root)
        result["train_manifest_path"] = str(self.train_manifest_path)
        result["validation_manifest_path"] = str(
            self.validation_manifest_path
        )
        result["test_manifest_path"] = str(self.test_manifest_path)
        result["output_root"] = str(self.output_root)
        return result
