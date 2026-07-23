from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
import tensorflow as tf


@dataclass
class GradCAMModel:
    target_layer_name: str
    model: tf.keras.Model


@dataclass(frozen=True)
class GradCAMResult:
    heatmap: np.ndarray
    overlay: np.ndarray
    target_layer_name: str


def output_rank(layer: tf.keras.layers.Layer) -> int | None:
    # Return the rank of a layer output when it is available.
    try:
        return len(layer.output.shape)
    except Exception:
        return None


def call_layer_for_inference(
    layer: tf.keras.layers.Layer,
    inputs: tf.Tensor,
) -> tf.Tensor:
    # Call a Keras layer in inference mode when it accepts a training argument.
    try:
        return layer(inputs, training=False)
    except TypeError:
        return layer(inputs)


def direct_spatial_layers(
    model: tf.keras.Model,
) -> list[tf.keras.layers.Layer]:
    # Find top-level spatial layers in a non-nested classifier.
    candidates = []

    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if isinstance(layer, tf.keras.Model):
            continue
        if output_rank(layer) == 4:
            candidates.append(layer)

    return candidates


def nested_spatial_layers(
    nested_model: tf.keras.Model,
) -> list[tf.keras.layers.Layer]:
    # Find spatial layers inside an application backbone.
    candidates = []

    for layer in reversed(nested_model.layers):
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        if output_rank(layer) == 4:
            candidates.append(layer)

    return candidates


def build_nested_grad_model(
    outer_model: tf.keras.Model,
    nested_model_index: int,
    target_layer: tf.keras.layers.Layer,
) -> tf.keras.Model:
    # Rebuild the outer forward path while exposing an internal backbone feature map.
    nested_model = outer_model.layers[nested_model_index]

    if not isinstance(nested_model, tf.keras.Model):
        raise TypeError("The selected layer is not a nested Keras model.")

    nested_feature_model = tf.keras.Model(
        inputs=nested_model.inputs,
        outputs=[target_layer.output, nested_model.output],
        name=f"{nested_model.name}_{target_layer.name}_gradcam",
    )

    outer_input = outer_model.input
    x = outer_input

    for layer in outer_model.layers[1:nested_model_index]:
        x = call_layer_for_inference(layer, x)

    feature_maps, x = nested_feature_model(x, training=False)

    for layer in outer_model.layers[nested_model_index + 1:]:
        x = call_layer_for_inference(layer, x)

    return tf.keras.Model(
        inputs=outer_input,
        outputs=[feature_maps, x],
        name=f"{outer_model.name}_{target_layer.name}_gradcam",
    )


def validate_grad_model(
    grad_model: tf.keras.Model,
    sample_input: np.ndarray,
    class_index: int,
) -> tuple[bool, str | None]:
    # Verify that the candidate feature tensor is connected to the class score.
    try:
        with tf.GradientTape() as tape:
            feature_maps, outputs = grad_model(
                sample_input,
                training=False,
            )
            class_score = outputs[:, class_index]

        gradients = tape.gradient(class_score, feature_maps)

        if gradients is None:
            return False, "gradients were None"
        if feature_maps.shape.rank != 4:
            return False, "feature-map rank was not four"
        if not bool(
            tf.reduce_all(tf.math.is_finite(gradients)).numpy()
        ):
            return False, "gradients contained non-finite values"

        return True, None
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def find_working_gradcam_model(
    model: tf.keras.Model,
    sample_input: np.ndarray,
    class_index: int,
) -> GradCAMModel:
    # Test direct layers first, then inspect nested application backbones.
    errors: list[str] = []

    for layer in direct_spatial_layers(model):
        try:
            grad_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[layer.output, model.output],
            )
            valid, error_message = validate_grad_model(
                grad_model,
                sample_input,
                class_index,
            )
            if valid:
                return GradCAMModel(
                    target_layer_name=layer.name,
                    model=grad_model,
                )
            errors.append(f"Direct {layer.name}: {error_message}")
        except Exception as error:
            errors.append(
                f"Direct {layer.name}: {type(error).__name__}: {error}"
            )

    for nested_index in reversed(range(len(model.layers))):
        nested_model = model.layers[nested_index]

        if not isinstance(nested_model, tf.keras.Model):
            continue

        for target_layer in nested_spatial_layers(nested_model):
            try:
                grad_model = build_nested_grad_model(
                    model,
                    nested_index,
                    target_layer,
                )
                valid, error_message = validate_grad_model(
                    grad_model,
                    sample_input,
                    class_index,
                )
                if valid:
                    return GradCAMModel(
                        target_layer_name=(
                            f"{nested_model.name}/{target_layer.name}"
                        ),
                        model=grad_model,
                    )
                errors.append(
                    f"Nested {nested_model.name}/{target_layer.name}: "
                    f"{error_message}"
                )
            except Exception as error:
                errors.append(
                    f"Nested {nested_model.name}/{target_layer.name}: "
                    f"{type(error).__name__}: {error}"
                )

    error_preview = "\n".join(errors[-15:])
    raise RuntimeError(
        "No spatial layer produced valid Grad-CAM gradients.\n"
        f"Last tested layers:\n{error_preview}"
    )


def create_heatmap(
    gradcam_model: GradCAMModel,
    model_input: np.ndarray,
    class_index: int,
) -> np.ndarray:
    # Compute a normalized Grad-CAM heatmap for one predicted class.
    with tf.GradientTape() as tape:
        feature_maps, outputs = gradcam_model.model(
            model_input,
            training=False,
        )
        class_score = outputs[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            "Gradients were not available for the selected feature layer."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    feature_maps = feature_maps[0]
    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1,
    )
    heatmap = tf.nn.relu(heatmap)

    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(
        maximum > 0,
        heatmap / maximum,
        heatmap,
    )

    return heatmap.numpy().astype(np.float32)


def resize_heatmap(
    heatmap: np.ndarray,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    # Resize a low-resolution activation map to the displayed image size.
    resized = tf.image.resize(
        heatmap[..., np.newaxis],
        [output_height, output_width],
        method="bilinear",
    )[..., 0]

    return tf.clip_by_value(
        resized,
        0.0,
        1.0,
    ).numpy().astype(np.float32)


def create_overlay(
    image_255: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    # Colorize the heatmap and blend it with the resized RGB image.
    color_map = matplotlib.colormaps["jet"]
    colored = color_map(heatmap)[..., :3] * 255.0

    overlay = (
        image_255.astype(np.float32) * (1.0 - alpha)
        + colored.astype(np.float32) * alpha
    )

    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def generate_gradcam(
    gradcam_model: GradCAMModel,
    model_input: np.ndarray,
    image_255: np.ndarray,
    class_index: int,
) -> GradCAMResult:
    # Create both the activation map and the visual overlay.
    small_heatmap = create_heatmap(
        gradcam_model,
        model_input,
        class_index,
    )

    height, width = image_255.shape[:2]
    heatmap = resize_heatmap(
        small_heatmap,
        height,
        width,
    )
    overlay = create_overlay(
        image_255,
        heatmap,
    )

    return GradCAMResult(
        heatmap=heatmap,
        overlay=overlay,
        target_layer_name=gradcam_model.target_layer_name,
    )
