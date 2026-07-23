from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent
MODEL_DIR = APP_ROOT / "models"
RESULTS_DIR = APP_ROOT / "results"
SAMPLES_DIR = APP_ROOT / "samples"

IMAGE_SIZE = (224, 224)
NUMBER_OF_CLASSES = 43
LOW_CONFIDENCE_THRESHOLD = 0.60


@dataclass(frozen=True)
class ModelConfig:
    display_name: str
    filename: str
    input_range: str
    description: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.filename


MODEL_CONFIGS: dict[str, ModelConfig] = {
    "ConvNeXtTiny": ModelConfig(
        display_name="ConvNeXtTiny",
        filename="best_convnexttiny.keras",
        input_range="zero_255",
        description=(
            "Best overall test accuracy and robustness in the completed experiments. "
            "It is the largest and slowest model in the comparison."
        ),
    ),
    "Custom CNN": ModelConfig(
        display_name="Custom CNN",
        filename="best_custom_cnn.keras",
        input_range="zero_one",
        description=(
            "Smallest model with the lowest storage requirement. "
            "It is useful when deployment size matters more than maximum accuracy."
        ),
    ),
    "EfficientNetV2B0": ModelConfig(
        display_name="EfficientNetV2B0",
        filename="best_efficientnetv2b0.keras",
        input_range="zero_255",
        description=(
            "Transfer-learning model with moderate size and fast recorded inference. "
            "Its test accuracy was lower than ConvNeXtTiny in this project."
        ),
    ),
}

MODEL_ORDER = [
    "ConvNeXtTiny",
    "Custom CNN",
    "EfficientNetV2B0",
]

RESULT_FILES = {
    "comparison": RESULTS_DIR / "all_models_comparison_rounded.csv",
    "robustness": RESULTS_DIR / "robustness_model_summary.csv",
    "gradcam": RESULTS_DIR / "gradcam_model_summary.csv",
    "shap": RESULTS_DIR / "shap_model_summary.csv",
}
