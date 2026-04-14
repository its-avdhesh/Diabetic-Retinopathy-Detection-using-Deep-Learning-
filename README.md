# Diabetic Retinopathy Detection (CNN, Python)

Standalone Python/TensorFlow inference project for classifying diabetic retinopathy from retina images.

## Overview
- CNN inference logic: `diab_retina_app/process.py`
- CLI entrypoint: `predict.py`
- Expected classes: `No DR`, `Mild`, `Moderate`, `Severe`, `Proliferative`

## Requirements
- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pillow

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install tensorflow numpy pillow
```

## Run Inference
```bash
python predict.py --image path/to/retina_image.jpg --model diab_retina_app/keras_model.h5
```

Optional labels file:
```bash
python predict.py --image path/to/retina_image.jpg --model diab_retina_app/keras_model.h5 --labels model/converted_keras/labels.txt
```

## Output Format
The script prints JSON:
- `predicted_class`: most likely class
- `confidence`: top-class confidence (%)
- `probabilities`: class-wise probabilities (%)
- `preprocessing`: applied preprocessing pipeline

## Paper-Aligned Behavior
The inference pipeline is aligned with the paper methodology:
- Retinal boundary-based cropping to remove dark/noisy margins
- Basic noise reduction (median filtering)
- Image resizing and normalization before CNN inference
- 5 DR severity classes: `No DR`, `Mild`, `Moderate`, `Severe`, `Proliferative`

## Notes
- Model files (`*.h5`, SavedModel exports) and generated artifacts should stay out of git.
- If your model file is not in `diab_retina_app/keras_model.h5`, pass a custom `--model` path.

