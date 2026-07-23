from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit
import tensorflow as tf
from PIL import Image

from config import MODEL_CONFIGS, RESULT_FILES, SAMPLES_DIR
from image_processing import image_to_array_255, prepare_model_input
from model_loader import load_model_bundle_uncached
from prediction import predict_top_k


def main() -> int:
    print("Python:", sys.version)
    print("TensorFlow:", tf.__version__)
    print("Streamlit:", streamlit.__version__)
    print("Pandas:", pd.__version__)
    print("NumPy:", np.__version__)

    missing_files = []

    for model_config in MODEL_CONFIGS.values():
        if not model_config.path.exists():
            missing_files.append(str(model_config.path))

    for result_path in RESULT_FILES.values():
        if not result_path.exists():
            missing_files.append(str(result_path))

    if missing_files:
        print("Missing required files:")
        for path in missing_files:
            print(" -", path)
        return 1

    sample_paths = sorted(SAMPLES_DIR.glob("*"))

    if not sample_paths:
        print("No sample images were found.")
        return 1

    with Image.open(sample_paths[0]) as image:
        image = image.convert("RGB")
        image_255 = image_to_array_255(image)

    print("Validation sample:", sample_paths[0].name)

    for model_name in MODEL_CONFIGS:
        print("\nLoading:", model_name)
        bundle = load_model_bundle_uncached(model_name)
        model_input = prepare_model_input(
            image_255,
            bundle.input_range,
        )
        prediction = predict_top_k(bundle, model_input)

        print(" Input shape:", bundle.model.input_shape)
        print(" Output shape:", bundle.model.output_shape)
        print(" Parameters:", bundle.parameter_count)
        print(
            " Prediction:",
            prediction.predicted_class_id,
            prediction.predicted_class_name,
            f"({prediction.confidence:.4f})",
        )

    print("\nInstallation and model validation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
