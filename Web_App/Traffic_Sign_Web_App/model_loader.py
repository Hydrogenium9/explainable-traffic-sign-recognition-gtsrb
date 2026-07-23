from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf
from huggingface_hub import hf_hub_download

from config import MODEL_CONFIGS


HF_MODEL_REPOSITORY = (
    "Hydrogenium9/gtsrb-traffic-sign-models"
)


@dataclass
class ModelBundle:
    model_name: str
    model: tf.keras.Model
    input_range: str
    output_is_logits: bool
    parameter_count: int
    model_size_mb: float


def set_linear_output(
    model: tf.keras.Model,
) -> bool:
    # Replace a final softmax activation so Grad-CAM uses pre-softmax scores.
    final_layer = model.layers[-1]

    if not hasattr(
        final_layer,
        "activation",
    ):
        return False

    activation_name = (
        tf.keras.activations.serialize(
            final_layer.activation
        )
    )

    if activation_name != "softmax":
        return False

    final_layer.activation = (
        tf.keras.activations.linear
    )

    return True


def resolve_model_path(
    model_name: str,
) -> Path:
    # Use a local model file when it is available.
    if model_name not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model: {model_name}"
        )

    model_config = MODEL_CONFIGS[
        model_name
    ]

    local_model_path = (
        model_config.path
    )

    if local_model_path.exists():
        print(
            f"Using local model file: "
            f"{local_model_path}"
        )

        return local_model_path

    # Download the selected model from Hugging Face when it is not local.
    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_MODEL_REPOSITORY,
            filename=model_config.filename,
            repo_type="model",
        )

    except Exception as error:
        raise RuntimeError(
            f"Could not download model "
            f"'{model_name}' from Hugging Face.\n"
            f"Repository: {HF_MODEL_REPOSITORY}\n"
            f"Filename: {model_config.filename}\n"
            f"Original error: {error}"
        ) from error

    resolved_path = Path(
        downloaded_path
    )

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"The downloaded model file "
            f"was not found: {resolved_path}"
        )

    print(
        f"Using Hugging Face model file: "
        f"{resolved_path}"
    )

    return resolved_path


def validate_model_shape(
    model_name: str,
    model: tf.keras.Model,
) -> None:
    # Validate the expected image input and classification output shapes.
    input_shape = tuple(
        model.input_shape[1:]
    )

    expected_input_shape = (
        224,
        224,
        3,
    )

    if input_shape != expected_input_shape:
        raise ValueError(
            f"{model_name} has an unexpected "
            f"input shape: {model.input_shape}. "
            f"Expected: "
            f"(None, 224, 224, 3)."
        )

    output_classes = int(
        model.output_shape[-1]
    )

    if output_classes != 43:
        raise ValueError(
            f"{model_name} has "
            f"{output_classes} output classes. "
            f"Expected 43."
        )


def load_model_bundle_uncached(
    model_name: str,
) -> ModelBundle:
    # Load one selected Keras model for prediction and Grad-CAM.
    if model_name not in MODEL_CONFIGS:
        raise KeyError(
            f"Unknown model: {model_name}"
        )

    model_config = MODEL_CONFIGS[
        model_name
    ]

    model_path = resolve_model_path(
        model_name
    )

    try:
        model = tf.keras.models.load_model(
            model_path,
            compile=False,
        )

    except Exception as error:
        raise RuntimeError(
            f"TensorFlow could not load "
            f"the model '{model_name}'.\n"
            f"Model path: {model_path}\n"
            f"Original error: {error}"
        ) from error

    validate_model_shape(
        model_name,
        model,
    )

    output_is_logits = set_linear_output(
        model
    )

    parameter_count = int(
        model.count_params()
    )

    model_size_mb = float(
        model_path.stat().st_size
        / 1024
        / 1024
    )

    return ModelBundle(
        model_name=model_name,
        model=model,
        input_range=(
            model_config.input_range
        ),
        output_is_logits=(
            output_is_logits
        ),
        parameter_count=(
            parameter_count
        ),
        model_size_mb=(
            model_size_mb
        ),
    )