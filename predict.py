import argparse
import json
from pathlib import Path

from diab_retina_app.process import DEFAULT_LABELS, predict_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diabetic retinopathy prediction on a single image."
    )
    parser.add_argument("--image", required=True, help="Path to input retina image.")
    parser.add_argument(
        "--model",
        default="diab_retina_app/keras_model.h5",
        help="Path to Keras model file (.h5 or SavedModel directory).",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Optional labels file, one class per line.",
    )
    return parser.parse_args()


def load_labels(labels_path: str) -> list[str]:
    if not labels_path:
        return DEFAULT_LABELS
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels)
    result = predict_image(args.image, args.model, labels)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
