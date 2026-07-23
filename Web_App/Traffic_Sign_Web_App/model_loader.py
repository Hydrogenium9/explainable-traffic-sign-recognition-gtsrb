from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from config import MODEL_CONFIGS


@dataclass
class ModelBundle:
    model_name: str
    model: tf.keras.Model
    input_range: str
    output_is_logits: bool
    parameter_count: int
    model_size_mb: float


def set_linear_output(model: tf.keras.Model) -> bool:
    # Replace a final softmax activation so Grad-CAM uses pre-softmax scores.
    final_layer = model.layers[-1]

    if not hasattr(final_layer, "activation"):
        return False

    activation_name = tf.keras.activations.serialize(
        final_layer.activation
    )

    if activation_name != "softmax":
        return False

    final_layer.activation = tf.keras.activations.linear
    return True


def load_model_bundle_uncached(model_name: str) -> ModelBundle:
    # Load one Keras model and prepare it for prediction and Grad-CAM.
    if model_name not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model: {model_name}")

    model_config = MODEL_CONFIGS[model_name]

    if not model_config.path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_config.path}"
        )

    model = tf.keras.models.load_model(
        model_config.path,
        compile=False,
    )

    output_is_logits = set_linear_output(model)

    return ModelBundle(
        model_name=model_name,
        model=model,
        input_range=model_config.input_range,
        output_is_logits=output_is_logits,
        parameter_count=int(model.count_params()),
        model_size_mb=float(
            model_config.path.stat().st_size / 1024 / 1024
        ),
    )
