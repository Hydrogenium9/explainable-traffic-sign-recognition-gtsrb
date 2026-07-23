from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import IMAGE_SIZE


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".ppm"}


class ImageValidationError(ValueError):
    pass


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    # Decode the uploaded image and normalize its orientation and color mode.
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = ImageOps.exif_transpose(image)
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ImageValidationError(
            "The selected file could not be decoded as an image."
        ) from error


def load_image_from_path(path: Path) -> tuple[Image.Image, bytes]:
    # Read a bundled sample image and return both its pixels and original bytes.
    image_bytes = path.read_bytes()
    return load_image_from_bytes(image_bytes), image_bytes


def resize_for_model(image: Image.Image) -> Image.Image:
    # Use the same direct 224 x 224 resizing strategy as the evaluation pipeline.
    return image.resize(
        IMAGE_SIZE,
        resample=Image.Resampling.LANCZOS,
    )


def image_to_array_255(image: Image.Image) -> np.ndarray:
    # Convert the resized RGB image to a float32 array in the [0, 255] range.
    resized = resize_for_model(image)
    array = np.asarray(resized, dtype=np.float32)

    if array.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ImageValidationError(
            f"Unexpected image shape after resizing: {array.shape}"
        )

    return np.clip(array, 0.0, 255.0)


def prepare_model_input(
    image_255: np.ndarray,
    input_range: str,
) -> np.ndarray:
    # Match the preprocessing range used during model training.
    image = np.asarray(image_255, dtype=np.float32)

    if input_range == "zero_one":
        image = image / 255.0
    elif input_range != "zero_255":
        raise ValueError(f"Unsupported input range: {input_range}")

    return image[np.newaxis, ...]


def image_fingerprint(image_bytes: bytes) -> str:
    # Produce a stable key for session-level prediction and explanation caches.
    return hashlib.sha256(image_bytes).hexdigest()
