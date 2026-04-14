from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from tensorflow import keras

DEFAULT_LABELS = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def _crop_retinal_boundary(image: Image.Image) -> Image.Image:
    """
    Approximate retinal boundary detection by removing near-black margins.
    """
    gray = np.asarray(image.convert("L"))
    mask = gray > 12
    if not np.any(mask):
        return image
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1
    return image.crop((left, top, right, bottom))


def _prepare_image(image_path: Path, image_size: int = 224) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = _crop_retinal_boundary(image)
    image = image.filter(ImageFilter.MedianFilter(size=3))
    image = ImageOps.fit(image, (image_size, image_size), Image.LANCZOS)
    image_array = np.asarray(image, dtype=np.float32)
    normalized_image_array = (image_array / 127.0) - 1.0
    return np.expand_dims(normalized_image_array, axis=0)


def predict_image(
    image_path: str,
    model_path: str,
    labels: List[str] | None = None,
) -> Dict[str, object]:
    np.set_printoptions(suppress=True)
    model = keras.models.load_model(model_path)
    data = _prepare_image(Path(image_path))
    prediction = model.predict(data, verbose=0)[0]

    class_labels = labels or DEFAULT_LABELS
    top_index = int(np.argmax(prediction))
    confidence = float(prediction[top_index]) * 100.0

    probabilities = {
        class_labels[i] if i < len(class_labels) else f"class_{i}": float(score) * 100.0
        for i, score in enumerate(prediction)
    }

    return {
        "predicted_class": class_labels[top_index]
        if top_index < len(class_labels)
        else f"class_{top_index}",
        "confidence": round(confidence, 2),
        "probabilities": {k: round(v, 2) for k, v in probabilities.items()},
        "preprocessing": [
            "retinal boundary crop",
            "median noise reduction",
            "resize to 224x224",
            "normalization to [-1, 1]",
        ],
    }
