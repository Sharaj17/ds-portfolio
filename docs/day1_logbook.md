### Day 1 Logbook — Baseline ML Experiment + MLflow (Beginner-Friendly, Detailed)

Objective:
Set up a reproducible Python environment, train a baseline RandomForest classifier on the Iris dataset, and track the run with MLflow (parameters, metrics, artifacts, and the model). Commit everything to GitHub as a portfolio artifact.

## Repository structure (today)
ds-portfolio/
├─ docs/
│  └─ day1_logbook.md        ← this file
├─ notebooks/
│  └─ day1_baseline.ipynb    ← today’s notebook
├─ requirements.txt          ← exact packages used today
└─ venv/                     ← local virtual environment (not committed)


docs/ holds written documentation (logbooks, notes).

notebooks/ contains Jupyter notebooks (experiments, demos).

requirements.txt pins the packages so anyone can reproduce the environment.

venv/ is your isolated Python environment (do not commit).

## Environment setup (Windows + VS Code)

Run these in VS Code Terminal opened at your project root (D:\Projects\ds-portfolio).

1.1 Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1


python -m venv venv — creates a virtual environment named venv so packages installed for this project won’t affect your system Python or other projects.

.\venv\Scripts\Activate.ps1 — activates the venv so the python and pip commands now use the venv’s interpreter.

1.2 (Fix applied) Clean re-install of all packages

We hit a “Module not found / binary mismatch” style issue, so we did a full reset of packages in the venv. This is a safe and effective fix.

# Uninstall everything currently installed in the venv
pip freeze | % { pip uninstall -y $_ }

# Clear pip’s download/build cache to avoid reusing broken wheels
pip cache purge

# Upgrade build tools (ensures correct wheel resolution)
pip install --upgrade pip setuptools wheel

# Install a known-good, Python 3.12 compatible stack
pip install "numpy>=2.0,<2.4" "scikit-learn>=1.5,<1.7" mlflow jupyter fastapi uvicorn ipykernel


Explanations (line by line):

pip freeze | % { pip uninstall -y $_ } — lists everything installed (pip freeze) and pipes each line into pip uninstall; % { ... } is PowerShell’s ForEach-Object.

pip cache purge — removes cached wheels so the next install fetches fresh, correct binaries.

pip install --upgrade pip setuptools wheel — keeps packaging tools modern (prevents accidental source builds).

The final pip install line pins versions that work with Python 3.12 on Windows:

NumPy 2.x supports Python 3.12 (we cap <2.4 to avoid brand-new releases).

scikit-learn 1.5+ ships wheels for Python 3.12 (we cap <1.7 for stability).

mlflow / jupyter for experiment tracking and notebooks.

fastapi / uvicorn we’ll use tomorrow for a minimal inference API.

ipykernel registers this environment as a Jupyter kernel for VS Code.

1.3 Register this venv as a Jupyter kernel
python -m ipykernel install --user --name=ds-portfolio --display-name "Python (ds-portfolio)"


This makes the kernel called “Python (ds-portfolio)” appear in the VS Code notebook kernel picker, ensuring your notebook uses this exact venv.

1.4 Verify imports (quick sanity checks)
python -c "import numpy, sklearn, mlflow; print('OK:', numpy.__version__)"


If this prints without errors, your environment is consistent and ready.

## Start the MLflow UI (dashboard)

MLflow logs runs to disk; the UI is a small web server that lets you browse experiments.

Open a second terminal in VS Code (still in ds-portfolio):

.\venv\Scripts\Activate.ps1
mlflow ui


Keep this terminal running; it serves the UI at http://127.0.0.1:5000
.

The UI does not create experiments; it only reads what your code logs.

## The notebook: notebooks/day1_baseline.ipynb

Open in VS Code → set Kernel to Python (ds-portfolio).
Run each cell top-to-bottom. Every block is explained.

## Reproducibility + imports
# Day 1 — Baseline ML Experiment (Iris + RandomForest + MLflow)

import os
import random
import numpy as np

# Reproducibility: control random number generators
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import pandas as pd


What each part does:

Setting SEED fixes pseudo-random behavior (shuffles, bootstraps) → stable comparisons.

sklearn imports cover dataset loading, splitting, the model, and evaluation metrics.

mlflow and mlflow.sklearn provide experiment tracking and model logging.

matplotlib + pandas help with plots and data inspection.

3.2 Load the dataset as DataFrames
X, y = load_iris(return_X_y=True, as_frame=True)
display(X.head())
display(y.head())
print("Class counts:\n", y.value_counts())


return_X_y=True returns (features, target) directly.

as_frame=True gives pandas structures (easier to inspect).

Iris has 3 balanced classes → perfect for a clean baseline.

## Proper split: train / validation / test ≈ 60 / 20 / 20
# Hold out 20% for final test (unseen data)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# From the remaining 80%, carve out validation = 25% of that (0.25 * 0.80 = 0.20 of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval
)

X_train.shape, X_val.shape, X_test.shape


Key parameters:

test_size=0.20 → 20% of the whole data is test.

random_state=SEED → reproducible splits.

stratify=... → preserves class ratios across splits (important for classification).

## Create (or reuse) an MLflow experiment
mlflow.set_experiment("day1_baseline")


If day1_baseline doesn’t exist, MLflow creates it. If it exists, MLflow reuses it.

Multiple runs will be logged under the same experiment and shown together in the UI.

## Define the RandomForest model (with clear hyperparameters)
params = {
    "n_estimators": 200,    # number of trees; more → lower variance, more compute
    "max_depth": None,      # None = grow fully (may overfit); try small values to regularize
    "min_samples_split": 2, # min samples to split a node (higher → simpler trees)
    "min_samples_leaf": 1,  # min samples at a leaf (higher → simpler trees)
    "max_features": "sqrt", # features considered at each split (good default for classification)
    "random_state": SEED,   # reproducible tree bootstraps/splits
    "n_jobs": -1            # use all CPU cores
}
rf = RandomForestClassifier(**params)
rf


Concepts:

Increasing n_estimators reduces variance but costs time.

Limiting max_depth and increasing min_* increases bias (simpler trees), lowering variance → useful when overfitting.

## Train, evaluate, and log to MLflow (params, metrics, artifacts, model)
with mlflow.start_run(run_name="rf_iris_baseline"):
    # --- Parameters (context for the run) ---
    mlflow.log_params(params)
    mlflow.log_param("split_seed", SEED)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("val_size", len(X_val))
    mlflow.log_param("test_size", len(X_test))

    # --- Train ---
    rf.fit(X_train, y_train)

    # --- Validation (tuning view) ---
    val_preds = rf.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1  = f1_score(y_val, val_preds, average="macro")
    mlflow.log_metric("val_accuracy", val_acc)
    mlflow.log_metric("val_f1_macro", val_f1)

    # --- Final test (unseen data) ---
    test_preds = rf.predict(X_test)
    test_acc = accuracy_score(y_test, test_preds)
    test_f1  = f1_score(y_test, test_preds, average="macro")
    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_f1_macro", test_f1)

    # --- Artifact: Confusion matrix image ---
    cm = confusion_matrix(y_test, test_preds)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, interpolation='nearest')
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha='center', va='center')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close(fig)
    mlflow.log_artifact("confusion_matrix.png")

    # --- Artifact: Detailed classification report (text file) ---
    report = classification_report(y_test, test_preds)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # --- Model artifact (serialized model + env metadata) ---
    mlflow.sklearn.log_model(
        sk_model=rf,
        artifact_path="model",
        registered_model_name=None  # we’ll use a registry later in the program
    )

    print({
        "val_accuracy": round(val_acc, 4),
        "val_f1_macro": round(val_f1, 4),
        "test_accuracy": round(test_acc, 4),
        "test_f1_macro": round(test_f1, 4)
    })


Why this matters:

Params → you can compare “what you tried”.

Metrics → you can sort runs by performance.

Artifacts → visual + textual evidence (plots, reports).

Model → you can reload and serve the exact trained artifact later.

## Browse your results

Keep the terminal running mlflow ui, then open: http://127.0.0.1:5000

Select the day1_baseline experiment.

Click into your run:

Check Parameters (hyperparameters + split sizes).

Compare Metrics (val_* vs test_*).

Open Artifacts (confusion_matrix.png, classification_report.txt).

See the Model under model/.

## (Optional) Micro-experiments

Re-run the training cell a couple of times changing one hyperparameter each time; each execution becomes a new MLflow run.

Variance test: n_estimators = 50 (fewer trees)

Bias test: max_depth = 4 (shallower trees)

In MLflow UI, sort by test_f1_macro and compare runs. Add a 2–3 line note in your README explaining what changed and why.

## Save your work to GitHub

Run in VS Code terminal at repo root:

git add notebooks/day1_baseline.ipynb classification_report.txt confusion_matrix.png
git commit -m "Day 1: Baseline RandomForest + MLflow logging (params, metrics, artifacts)"
git push


Add progress to README.md:

### Day 1 — Baseline Experiment
- RandomForest on Iris with stratified 60/20/20 split.
- Logged hyperparameters, validation/test metrics, confusion matrix image, and the model artifact in MLflow.
- Did small hyperparameter variations to see bias/variance trade-offs.


Then:

git add README.md
git commit -m "Update progress log for Day 1"
git push

## Today’s only hiccup (and fix)

Symptom: Import errors / “module not found” / build mismatch in packages.
Root cause: Mismatched or corrupted wheels in the venv (common after version changes).
Fix we used (clean and robust):

.\venv\Scripts\Activate.ps1
pip freeze | % { pip uninstall -y $_ }
pip cache purge
pip install --upgrade pip setuptools wheel
pip install "numpy>=2.0,<2.4" "scikit-learn>=1.5,<1.7" mlflow jupyter fastapi uvicorn ipykernel
python -m ipykernel install --user --name=ds-portfolio --display-name "Python (ds-portfolio)"


Why it works:

Removes all packages → no leftovers.

Purges cache → no stale wheels.

Installs binary wheels compatible with Python 3.12 on Windows.

Registers the kernel so the notebook uses the same interpreter.

## Key takeaways

Reproducibility: venv + pinned requirements.txt + seeds.

Good practice: always keep a val set separate from test.

Experiment mindset: track params/metrics/artifacts with MLflow.

Portfolio storytelling: screenshots + artifacts + clear README notes.

Troubleshooting skill: if imports fail, a clean uninstall/reinstall (with cache purge) quickly resolves most mismatches.

### End of Day-1

You now have:

A clean environment,

A logged baseline experiment,

Clear documentation,

Commits in GitHub that show progress.

Tomorrow we’ll wrap this model in a tiny FastAPI service (/predict) and document how to run and call it locally.