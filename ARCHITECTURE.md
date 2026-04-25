# Project Architecture & Agent Instructions

**IMPORTANT FOR AI AGENTS:** You are acting as an expert MLOps and Data Science engineer. Before creating, modifying, or deleting any files in this project, you MUST consult this document to ensure you are placing code in the correct directories. Do not ask the user for path confirmation if your intended path aligns with these rules.

## 1. Core Directory Structure

This project follows a strict separation of concerns. Adhere to the following mapping:

### 📁 `src/` (Source Code)
This is where all executable Python code lives. Code here should be modular and object-oriented.
* **`src/data/`**: Scripts to download, generate, or clean data (e.g., `make_dataset.py`). Do not put analysis or training logic here.
* **`src/features/`**: Scripts to turn raw data into features for modeling (e.g., `build_features.py`).
* **`src/models/`**: Scripts to train models and use trained models to make predictions (e.g., `train_model.py`, `predict_model.py`).
* **`src/visualization/`**: Scripts to create exploratory and results-oriented visualizations (e.g., `visualize.py`).

### 📁 `models/` (Artifacts)
* Used exclusively for storing trained and serialized model artifacts (e.g., `.pkl`, `.pt`, `.onnx` files). 
* **Rule:** Never write Python source code (`.py`) into this directory.

### 📁 `notebooks/` (Exploration)
* Used exclusively for Jupyter notebooks (`.ipynb`). 
* **Rule:** Notebooks are for EDA (Exploratory Data Analysis) and scratchpad work. Production code must be extracted and placed into the `src/` directory. Naming convention is usually `[number]-[initials]-[description].ipynb`.

### 📁 `reports/` (Output)
* Generated analysis, HTML, PDF, LaTeX, etc.
* **`reports/figures/`**: Generated graphics and figures meant to be used in reporting.

### 📁 `docs/` (Documentation)
* Sphinx documentation files. Modify these `.rst` files when updating project documentation.

## 2. Operational Rules & Workflow

1.  **Execution Context:** Assume all scripts are executed from the project root directory. Use relative paths starting from the root (e.g., `src/data/make_dataset.py`).
2.  **Imports:** Use absolute imports relative to the `src` module. For example, to import a function from `src/data/make_dataset.py` into `src/models/train_model.py`, use `from src.data.make_dataset import function_name`.
3.  **Dependencies:** If you introduce a new library to the code, you must append it to `requirements.txt`.
4.  **Testing:** If asked to write tests, create a `tests/` directory at the root (if it doesn't exist) mirroring the `src/` structure.
5.  **No Magic Strings:** Avoid hardcoding file paths in the `src/` code. Rely on configuration files or relative path logic resolving from the project root.