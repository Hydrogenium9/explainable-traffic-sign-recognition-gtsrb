from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import tensorflow as tf

from class_names import class_label
from model_loader import ModelBundle


@dataclass(frozen=True)
class PredictionItem:
    rank: int
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    predicted_class_id: int
    predicted_class_name: str
    confidence: float
    top_predictions: tuple[PredictionItem, ...]
    inference_ms: float
    probabilities: np.ndarray


def to_probabilities(
    raw_output: np.ndarray,
    output_is_logits: bool,
) -> np.ndarray:
    # Convert model outputs to a normalized probability distribution.
    values = np.asarray(raw_output, dtype=np.float32).reshape(-1)

    looks_like_probabilities = (
        np.all(values >= 0.0)
        and np.all(values <= 1.0)
        and np.isclose(values.sum(), 1.0, atol=1e-3)
    )

    if output_is_logits or not looks_like_probabilities:
        values = tf.nn.softmax(values).numpy()

    return values.astype(np.float32)


def predict_top_k(
    bundle: ModelBundle,
    model_input: np.ndarray,
    top_k: int = 3,
) -> PredictionResult:
    # Run one forward pass and return ranked class probabilities.
    start_time = time.perf_counter()

    raw_output = bundle.model(
        model_input,
        training=False,
    ).numpy()[0]

    inference_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    probabilities = to_probabilities(
        raw_output,
        bundle.output_is_logits,
    )

    top_indices = np.argsort(
        probabilities
    )[-top_k:][::-1]

    top_predictions = tuple(
        PredictionItem(
            rank=rank,
            class_id=int(class_id),
            class_name=class_label(int(class_id)),
            confidence=float(probabilities[class_id]),
        )
        for rank, class_id in enumerate(top_indices, start=1)
    )

    best = top_predictions[0]

    return PredictionResult(
        predicted_class_id=best.class_id,
        predicted_class_name=best.class_name,
        confidence=best.confidence,
        top_predictions=top_predictions,
        inference_ms=float(inference_ms),
        probabilities=probabilities,
    )
