# Diabetic Retinopathy — Django project
Django project for diabetic retinopathy model inference and related utilities.

## Overview
- Django app: `diab_retina_app`
- Project configuration: `diabetic_retinopathy`
- Contains converted model artifacts under `model/converted_keras` and `model/converted_savedmodel` and a Keras model at `diab_retina_app/keras_model.h5`.

## Requirements
- Python 3.8+ (project currently uses a virtual environment `.venv` / `myenv`).
- Django (project contains Django files).
- Keras / TensorFlow for model loading (if you run inference).

## Quick setup
1. (Optional) Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies (if you have `requirements.txt`):

```bash
pip install -r requirements.txt
```

3. Apply migrations and run server:

```bash
python manage.py migrate
python manage.py runserver
```

4. Run tests:

```bash
python manage.py test diab_retina_app
```

## Project structure (high level)
- `manage.py` — Django entrypoint
- `diabetic_retinopathy/` — Django project settings and wsgi/asgi
- `diab_retina_app/` — main app (models, views, process utilities, stored `keras_model.h5`)
- `model/converted_keras/` — converted keras model and `labels.txt`
- `model/converted_savedmodel/` — savedmodel export
- `output/` — generated outputs (ignored)
- `db.sqlite3` — local sqlite DB (excluded from git)

## Models & large artifacts
Model files (e.g., `*.h5`, savedmodel folders, `output/`) are large and should not be committed. Use the files under `model/` for inference. If you update/replace models, add them to your deployment storage or dataset release rather than committing to git.

## Usage notes
- To run inference, inspect `diab_retina_app/process.py` and `diab_retina_app/views.py` for examples of model loading and prediction.
- Labels for predictions are in `model/converted_keras/labels.txt` (or `model/converted_savedmodel/labels.txt`).

## Contact / Next steps
- If you want, I can:
  - add a `requirements.txt` based on the environment,
  - update `.gitignore` in-place and untrack files,
  - or create a small script to load and test the model.

