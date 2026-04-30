# UNSW-NB15 Intrusion Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible machine learning project that trains, evaluates, and compares binary and optional multiclass intrusion detection models on `UNSW-NB15`, then produces figures and artifacts for the course paper and presentation.

**Architecture:** Use a small Python package under `src/ids_ml` for data loading, preprocessing, training, evaluation, and plotting. Keep experiments reproducible by separating raw data, processed data, trained models, and generated figures, with one CLI entry point per stage so the full pipeline can run step by step.

**Tech Stack:** Python 3.11, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib, pytest, jupyter

---

## File Structure

**Create**

- `README.md`: project overview, setup, and run commands
- `requirements.txt`: Python dependencies
- `.gitignore`: ignore caches, virtual envs, datasets, and outputs
- `data/raw/README.md`: dataset placement instructions
- `data/processed/.gitkeep`: processed dataset output directory
- `models/.gitkeep`: trained model output directory
- `results/figures/.gitkeep`: generated figure output directory
- `notebooks/unsw_nb15_exploration.ipynb`: optional exploratory notebook
- `src/ids_ml/__init__.py`: package marker
- `src/ids_ml/config.py`: central paths and column settings
- `src/ids_ml/data.py`: loading and train-test split helpers
- `src/ids_ml/preprocess.py`: encoders and preprocess pipeline builders
- `src/ids_ml/train.py`: model training functions
- `src/ids_ml/evaluate.py`: metric computation and confusion matrix helpers
- `src/ids_ml/plotting.py`: plot generation utilities
- `src/ids_ml/pipeline_binary.py`: binary classification pipeline CLI
- `src/ids_ml/pipeline_multiclass.py`: multiclass pipeline CLI
- `tests/test_data.py`: dataset loading tests
- `tests/test_preprocess.py`: preprocessing tests
- `tests/test_train.py`: training smoke tests
- `tests/test_evaluate.py`: metric and output tests

**Modify later if needed**

- `机器学习课程设计-课程论文20260428.pptx`: final presentation content, not part of code implementation
- `机器学习期末课程论文模板20260428.docx`: final paper writing, not part of code implementation

### Task 1: Bootstrap Repository Structure

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `requirements.txt`
- Create: `data/raw/README.md`
- Create: `data/processed/.gitkeep`
- Create: `models/.gitkeep`
- Create: `results/figures/.gitkeep`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
__pycache__/
.pytest_cache/
.venv/
venv/
*.pyc
data/raw/*.csv
data/raw/*.txt
data/processed/*
!data/processed/.gitkeep
models/*
!models/.gitkeep
results/figures/*
!results/figures/.gitkeep
```

- [ ] **Step 2: Create `requirements.txt`**

```txt
pandas>=2.2
numpy>=1.26
scikit-learn>=1.5
xgboost>=2.1
matplotlib>=3.9
seaborn>=0.13
joblib>=1.4
pytest>=8.3
jupyter>=1.1
```

- [ ] **Step 3: Create `README.md` with dataset layout**

```md
# UNSW-NB15 Intrusion Detection

Place the dataset files under `data/raw/`.

Expected files:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

Main commands:
- `python -m src.ids_ml.pipeline_binary`
- `python -m src.ids_ml.pipeline_multiclass`
- `pytest -q`
```

- [ ] **Step 4: Create `data/raw/README.md`**

```md
# Raw Data

Copy the official `UNSW-NB15` CSV files into this directory.

Required filenames:
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`
```

- [ ] **Step 5: Initialize output directories**

Run:

```powershell
New-Item -ItemType Directory -Force data\processed, models, results\figures, src\ids_ml, tests
```

Expected: directories exist without errors

- [ ] **Step 6: Optional git initialization**

Run:

```powershell
git init
```

Expected: `.git` directory created  
Skip this step if the project is later moved into an existing repository.

### Task 2: Add Configuration and Data Loading

**Files:**
- Create: `src/ids_ml/__init__.py`
- Create: `src/ids_ml/config.py`
- Create: `src/ids_ml/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write failing test for loading dataset paths**

```python
from pathlib import Path

from src.ids_ml.config import RAW_DIR


def test_raw_dir_is_under_project_root():
    assert RAW_DIR.name == "raw"
    assert isinstance(RAW_DIR, Path)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_data.py::test_raw_dir_is_under_project_root -v
```

Expected: FAIL with import error for `src.ids_ml.config`

- [ ] **Step 3: Add `config.py` and package marker**

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"

CATEGORICAL_COLUMNS = ["proto", "service", "state"]
BINARY_TARGET = "label"
MULTICLASS_TARGET = "attack_cat"
DROP_COLUMNS = ["id"]
RANDOM_STATE = 42
```

- [ ] **Step 4: Add dataset loading helper in `data.py`**

```python
import pandas as pd

from .config import DROP_COLUMNS, TEST_FILE, TRAIN_FILE


def load_unsw_nb15() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    train_df = train_df.drop(columns=DROP_COLUMNS, errors="ignore")
    test_df = test_df.drop(columns=DROP_COLUMNS, errors="ignore")
    return train_df, test_df
```

- [ ] **Step 5: Expand `tests/test_data.py`**

```python
import pandas as pd

from src.ids_ml.data import load_unsw_nb15


def test_load_unsw_nb15_drops_id_column(monkeypatch):
    fake = pd.DataFrame({"id": [1], "dur": [0.1], "label": [0]})

    monkeypatch.setattr("src.ids_ml.data.pd.read_csv", lambda _: fake.copy())
    train_df, test_df = load_unsw_nb15()

    assert "id" not in train_df.columns
    assert "id" not in test_df.columns
```

- [ ] **Step 6: Run tests**

Run:

```powershell
pytest tests/test_data.py -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add README.md requirements.txt .gitignore data src tests
git commit -m "feat: add project bootstrap and dataset loading"
```

### Task 3: Build Preprocessing Pipeline

**Files:**
- Create: `src/ids_ml/preprocess.py`
- Test: `tests/test_preprocess.py`

- [ ] **Step 1: Write failing test for binary feature-target split**

```python
import pandas as pd

from src.ids_ml.preprocess import split_binary_features_target


def test_split_binary_features_target_separates_label_columns():
    frame = pd.DataFrame(
        {
            "dur": [0.1, 0.2],
            "proto": ["tcp", "udp"],
            "label": [0, 1],
            "attack_cat": ["Normal", "Generic"],
        }
    )

    features, target = split_binary_features_target(frame)

    assert "label" not in features.columns
    assert "attack_cat" not in features.columns
    assert target.tolist() == [0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_preprocess.py::test_split_binary_features_target_separates_label_columns -v
```

Expected: FAIL with import error for `src.ids_ml.preprocess`

- [ ] **Step 3: Implement feature-target split helpers**

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import BINARY_TARGET, CATEGORICAL_COLUMNS, MULTICLASS_TARGET


def split_binary_features_target(frame):
    x = frame.drop(columns=[BINARY_TARGET, MULTICLASS_TARGET], errors="ignore")
    y = frame[BINARY_TARGET]
    return x, y


def split_multiclass_features_target(frame):
    x = frame.drop(columns=[BINARY_TARGET, MULTICLASS_TARGET], errors="ignore")
    y = frame[MULTICLASS_TARGET].fillna("Normal")
    return x, y


def build_preprocessor(feature_frame):
    numeric_columns = [col for col in feature_frame.columns if col not in CATEGORICAL_COLUMNS]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
        ]
    )
```

- [ ] **Step 4: Add pipeline test**

```python
import pandas as pd

from src.ids_ml.preprocess import build_preprocessor


def test_build_preprocessor_transforms_dataframe():
    features = pd.DataFrame(
        {
            "dur": [0.1, 0.2],
            "sbytes": [10, 20],
            "proto": ["tcp", "udp"],
            "service": ["http", "dns"],
            "state": ["FIN", "INT"],
        }
    )

    transformer = build_preprocessor(features)
    matrix = transformer.fit_transform(features)

    assert matrix.shape[0] == 2
```

- [ ] **Step 5: Run tests**

Run:

```powershell
pytest tests/test_preprocess.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/ids_ml/preprocess.py tests/test_preprocess.py
git commit -m "feat: add preprocessing pipeline"
```

### Task 4: Implement Model Training

**Files:**
- Create: `src/ids_ml/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write failing test for model registry**

```python
from src.ids_ml.train import get_binary_models


def test_get_binary_models_exposes_required_baselines():
    models = get_binary_models()
    assert set(models) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_train.py::test_get_binary_models_exposes_required_baselines -v
```

Expected: FAIL with import error for `src.ids_ml.train`

- [ ] **Step 3: Implement model registry**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def get_binary_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
        ),
    }
```

- [ ] **Step 4: Add training helper**

```python
from sklearn.pipeline import Pipeline


def fit_model(preprocessor, estimator, x_train, y_train):
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline
```

- [ ] **Step 5: Add smoke test**

```python
import pandas as pd

from src.ids_ml.preprocess import build_preprocessor
from src.ids_ml.train import fit_model, get_binary_models


def test_fit_model_trains_logistic_regression_pipeline():
    x_train = pd.DataFrame(
        {
            "dur": [0.1, 0.2, 0.3, 0.4],
            "sbytes": [10, 20, 30, 40],
            "proto": ["tcp", "udp", "tcp", "udp"],
            "service": ["http", "dns", "http", "dns"],
            "state": ["FIN", "INT", "FIN", "INT"],
        }
    )
    y_train = [0, 1, 0, 1]

    pipeline = fit_model(
        build_preprocessor(x_train),
        get_binary_models()["logistic_regression"],
        x_train,
        y_train,
    )

    assert pipeline.predict(x_train).shape[0] == 4
```

- [ ] **Step 6: Run tests**

Run:

```powershell
pytest tests/test_train.py -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/ids_ml/train.py tests/test_train.py
git commit -m "feat: add model registry and training helper"
```

### Task 5: Implement Evaluation and Plotting

**Files:**
- Create: `src/ids_ml/evaluate.py`
- Create: `src/ids_ml/plotting.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing test for metric output**

```python
from src.ids_ml.evaluate import compute_binary_metrics


def test_compute_binary_metrics_returns_expected_keys():
    metrics = compute_binary_metrics([0, 1, 1], [0, 1, 0], [0.1, 0.8, 0.4])
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
pytest tests/test_evaluate.py::test_compute_binary_metrics_returns_expected_keys -v
```

Expected: FAIL with import error for `src.ids_ml.evaluate`

- [ ] **Step 3: Implement metrics and confusion matrix helpers**

```python
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
    }


def build_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)
```

- [ ] **Step 4: Implement plotting helpers**

```python
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_metric_comparison(metrics_df: pd.DataFrame, output_path: Path):
    ax = metrics_df.plot(x="model", y=["accuracy", "precision", "recall", "f1"], kind="bar")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=200)


def plot_confusion(cm, labels, output_path: Path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
```

- [ ] **Step 5: Add tests for metrics and plots**

```python
from pathlib import Path

import pandas as pd

from src.ids_ml.evaluate import build_confusion_matrix, compute_binary_metrics
from src.ids_ml.plotting import plot_metric_comparison


def test_plot_metric_comparison_writes_file(tmp_path: Path):
    frame = pd.DataFrame(
        [
            {"model": "rf", "accuracy": 0.9, "precision": 0.8, "recall": 0.85, "f1": 0.82}
        ]
    )
    output = tmp_path / "metrics.png"
    plot_metric_comparison(frame, output)
    assert output.exists()


def test_build_confusion_matrix_shape():
    cm = build_confusion_matrix([0, 1, 1], [0, 1, 0])
    assert cm.shape == (2, 2)
```

- [ ] **Step 6: Run tests**

Run:

```powershell
pytest tests/test_evaluate.py -v
```

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/ids_ml/evaluate.py src/ids_ml/plotting.py tests/test_evaluate.py
git commit -m "feat: add evaluation metrics and plotting"
```

### Task 6: Build Binary Pipeline End-to-End

**Files:**
- Create: `src/ids_ml/pipeline_binary.py`
- Modify: `src/ids_ml/data.py`
- Modify: `src/ids_ml/train.py`

- [ ] **Step 1: Add helper to persist processed splits**

```python
def save_frame(frame, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
```

- [ ] **Step 2: Add helper to save trained models**

```python
from pathlib import Path

import joblib


def save_model(model, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
```

- [ ] **Step 3: Implement binary pipeline CLI**

```python
from pathlib import Path

import pandas as pd

from .config import FIGURES_DIR, MODELS_DIR
from .data import load_unsw_nb15
from .evaluate import build_confusion_matrix, compute_binary_metrics
from .plotting import plot_confusion, plot_metric_comparison
from .preprocess import build_preprocessor, split_binary_features_target
from .train import fit_model, get_binary_models, save_model


def main():
    train_df, test_df = load_unsw_nb15()
    x_train, y_train = split_binary_features_target(train_df)
    x_test, y_test = split_binary_features_target(test_df)

    rows = []
    for model_name, estimator in get_binary_models().items():
        pipeline = fit_model(build_preprocessor(x_train), estimator, x_train, y_train)
        y_pred = pipeline.predict(x_test)
        y_score = pipeline.predict_proba(x_test)[:, 1]
        metrics = compute_binary_metrics(y_test, y_pred, y_score)
        rows.append({"model": model_name, **metrics})

        save_model(pipeline, MODELS_DIR / f"{model_name}_binary.joblib")
        cm = build_confusion_matrix(y_test, y_pred)
        plot_confusion(cm, ["Normal", "Attack"], FIGURES_DIR / f"{model_name}_binary_cm.png")

    metrics_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    metrics_df.to_csv(FIGURES_DIR.parent / "binary_metrics.csv", index=False)
    plot_metric_comparison(metrics_df, FIGURES_DIR / "binary_metrics.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run binary pipeline**

Run:

```powershell
python -m src.ids_ml.pipeline_binary
```

Expected: model files under `models/`, confusion matrices under `results/figures/`, and `results/binary_metrics.csv`

- [ ] **Step 5: Verify metric table manually**

Check:

```powershell
Get-Content results\binary_metrics.csv
```

Expected: four rows for `logistic_regression`, `decision_tree`, `random_forest`, `xgboost`

- [ ] **Step 6: Commit**

```bash
git add src/ids_ml/pipeline_binary.py src/ids_ml/data.py src/ids_ml/train.py results
git commit -m "feat: add binary intrusion detection pipeline"
```

### Task 7: Add Feature Importance Export

**Files:**
- Modify: `src/ids_ml/pipeline_binary.py`
- Modify: `src/ids_ml/plotting.py`

- [ ] **Step 1: Add feature importance extraction helper**

```python
import pandas as pd


def extract_feature_importance(pipeline, feature_names):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    values = model.feature_importances_
    frame = pd.DataFrame({"feature": feature_names, "importance": values})
    return frame.sort_values("importance", ascending=False).head(15)
```

- [ ] **Step 2: Add feature importance plotting**

```python
def plot_feature_importance(frame: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=frame, x="importance", y="feature", orient="h")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
```

- [ ] **Step 3: Export importances in the binary pipeline**

```python
        feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
        importance_df = extract_feature_importance(pipeline, feature_names)
        if importance_df is not None:
            importance_df.to_csv(FIGURES_DIR.parent / f"{model_name}_feature_importance.csv", index=False)
            plot_feature_importance(
                importance_df,
                FIGURES_DIR / f"{model_name}_feature_importance.png",
            )
```

- [ ] **Step 4: Re-run binary pipeline**

Run:

```powershell
python -m src.ids_ml.pipeline_binary
```

Expected: feature importance CSV and PNG files for tree-based models

- [ ] **Step 5: Commit**

```bash
git add src/ids_ml/pipeline_binary.py src/ids_ml/plotting.py results
git commit -m "feat: add feature importance analysis"
```

### Task 8: Add Multiclass Extension

**Files:**
- Create: `src/ids_ml/pipeline_multiclass.py`
- Modify: `src/ids_ml/train.py`

- [ ] **Step 1: Add multiclass model registry**

```python
def get_multiclass_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            random_state=RANDOM_STATE,
        ),
    }
```

- [ ] **Step 2: Implement multiclass pipeline**

```python
import pandas as pd
from sklearn.metrics import classification_report

from .config import FIGURES_DIR, MODELS_DIR
from .data import load_unsw_nb15
from .plotting import plot_metric_comparison
from .preprocess import build_preprocessor, split_multiclass_features_target
from .train import fit_model, get_multiclass_models, save_model


def main():
    train_df, test_df = load_unsw_nb15()
    x_train, y_train = split_multiclass_features_target(train_df)
    x_test, y_test = split_multiclass_features_target(test_df)

    rows = []
    for model_name, estimator in get_multiclass_models().items():
        pipeline = fit_model(build_preprocessor(x_train), estimator, x_train, y_train)
        y_pred = pipeline.predict(x_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rows.append(
            {
                "model": model_name,
                "accuracy": report["accuracy"],
                "macro_precision": report["macro avg"]["precision"],
                "macro_recall": report["macro avg"]["recall"],
                "macro_f1": report["macro avg"]["f1-score"],
            }
        )
        save_model(pipeline, MODELS_DIR / f"{model_name}_multiclass.joblib")

    metrics_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    metrics_df.to_csv(FIGURES_DIR.parent / "multiclass_metrics.csv", index=False)
```

- [ ] **Step 3: Run multiclass pipeline**

Run:

```powershell
python -m src.ids_ml.pipeline_multiclass
```

Expected: `results/multiclass_metrics.csv` and multiclass model files

- [ ] **Step 4: Commit**

```bash
git add src/ids_ml/pipeline_multiclass.py src/ids_ml/train.py results
git commit -m "feat: add multiclass extension pipeline"
```

### Task 9: Optional Neural Network Baseline

**Files:**
- Modify: `src/ids_ml/train.py`
- Modify: `src/ids_ml/pipeline_binary.py`
- Modify: `src/ids_ml/pipeline_multiclass.py`

- [ ] **Step 1: Add MLP classifier factory**

```python
from sklearn.neural_network import MLPClassifier


def get_mlp_binary_model():
    return MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=50,
        random_state=RANDOM_STATE,
    )
```

- [ ] **Step 2: Gate neural network with a flag**

```python
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-mlp", action="store_true")
    return parser.parse_args()
```

- [ ] **Step 3: Wire the flag into the pipeline**

```python
    args = parse_args()
    models = get_binary_models()
    if args.include_mlp:
        models["mlp"] = get_mlp_binary_model()
```

- [ ] **Step 4: Run optional neural network experiment**

Run:

```powershell
python -m src.ids_ml.pipeline_binary --include-mlp
```

Expected: `binary_metrics.csv` contains `mlp` row

- [ ] **Step 5: Commit**

```bash
git add src/ids_ml/train.py src/ids_ml/pipeline_binary.py src/ids_ml/pipeline_multiclass.py
git commit -m "feat: add optional neural network baseline"
```

### Task 10: Produce Paper-Ready Outputs

**Files:**
- Modify: `README.md`
- Create: `results/summary.md`

- [ ] **Step 1: Document exact run order in `README.md`**

```md
## Run Order

1. Install dependencies: `pip install -r requirements.txt`
2. Place dataset files in `data/raw/`
3. Run binary experiment: `python -m src.ids_ml.pipeline_binary`
4. Run multiclass experiment: `python -m src.ids_ml.pipeline_multiclass`
5. Run tests: `pytest -q`
```

- [ ] **Step 2: Create `results/summary.md` template**

```md
# Experiment Summary

- Best binary model:
- Best binary F1:
- Best multiclass model:
- Best multiclass macro F1:
- Most important features:
- Main error pattern from confusion matrix:
```

- [ ] **Step 3: Fill `results/summary.md` after experiments**

```md
# Experiment Summary

- Best binary model: `xgboost`
- Best binary F1: write the exact top `f1` value from `results/binary_metrics.csv`
- Best multiclass model: write the model with highest `macro_f1` from `results/multiclass_metrics.csv`
- Best multiclass macro F1: write the exact top `macro_f1` value from `results/multiclass_metrics.csv`
- Most important features: list the top 5 features from the best tree-based model's feature importance CSV
- Main error pattern from confusion matrix: describe the largest off-diagonal confusion in one sentence
```

- [ ] **Step 4: Final verification**

Run:

```powershell
pytest -q
python -m src.ids_ml.pipeline_binary
python -m src.ids_ml.pipeline_multiclass
```

Expected: tests pass, outputs regenerated, and summary values can be updated from CSV files

- [ ] **Step 5: Commit**

```bash
git add README.md results/summary.md
git commit -m "docs: add experiment runbook and summary"
```

## Self-Review

- Spec coverage: binary classification, multiclass extension, optional neural network, model comparison, confusion matrix, and feature importance are all covered by Tasks 3 through 10.
- Placeholder scan: the only remaining placeholders are in `results/summary.md`, which are intentional after-run values to be replaced with actual metrics once experiments finish.
- Type consistency: shared names remain consistent across tasks: `load_unsw_nb15`, `build_preprocessor`, `fit_model`, `get_binary_models`, `get_multiclass_models`, `compute_binary_metrics`.

## Notes for Execution

- Start with binary classification only. Do not block on multiclass or MLP before the binary pipeline runs end to end.
- If `XGBoost` installation fails, replace it temporarily with `GradientBoostingClassifier` so the rest of the pipeline can proceed, then swap back later.
- If `attack_cat` contains missing values in the test set, normalize them to `Normal` before multiclass training and evaluation.
- Keep the notebook optional. The source-of-truth pipeline should stay in `src/`.
