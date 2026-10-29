"""Train the lab sentiment model and enforce the 60% business quality gate."""

import argparse
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

THRESHOLD = 0.60
SEED = 42
TEST_SIZE = 0.25


def passes_gate(accuracy):
    """An invalid score must never pass; the minimum is inclusive."""
    return math.isfinite(accuracy) and THRESHOLD <= accuracy <= 1.0


def load_data(path):
    if not path.is_file():
        raise ValueError(f"Missing dataset: {path}")
    frame = pd.read_csv(path)
    if not {"text", "sentiment"} <= set(frame.columns):
        raise ValueError("Dataset must have text and sentiment columns.")
    if frame["text"].isna().any() or not frame["text"].map(
        lambda value: isinstance(value, str) and bool(value.strip())
    ).all():
        raise ValueError("Every review must contain nonempty text.")
    if frame["sentiment"].isna().any() or not frame["sentiment"].isin([0, 1]).all():
        raise ValueError("Sentiment must contain only binary labels 0 and 1.")
    counts = frame["sentiment"].value_counts()
    if len(counts) != 2 or counts.min() < 2 or len(frame) < 8:
        raise ValueError("Stratified evaluation needs at least 8 rows and 2 per class.")
    return frame


def train(args):
    data_path = args.data.resolve()
    tracking_dir = args.tracking_dir.resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("sentiment-ci")
    with mlflow.start_run(run_name=data_path.stem) as run:
        mlflow.log_params({
            "C": args.C, "accuracy_threshold": THRESHOLD, "random_state": SEED,
            "test_size": TEST_SIZE, "ngram_range": "(1, 2)", "min_df": 1,
            "dataset": data_path.name,
            "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        })
        mlflow.set_tags({
            "git_commit": os.getenv("GITHUB_SHA", "local-unrecorded"),
            "github_run_id": os.getenv("GITHUB_RUN_ID", "local"),
        })
        mlflow.log_dict({
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("mlflow", "scikit-learn", "pandas", "numpy")},
        }, "environment.json")
        frame = load_data(data_path)
        train_idx, test_idx = train_test_split(
            frame.index, test_size=TEST_SIZE, random_state=SEED,
            stratify=frame["sentiment"],
        )
        X_train = frame.loc[train_idx, "text"]
        X_test = frame.loc[test_idx, "text"]
        y_train = frame.loc[train_idx, "sentiment"]
        y_test = frame.loc[test_idx, "sentiment"]
        print(f"Dataset size: {len(frame)} (pos={int(frame.sentiment.sum())}, "
              f"neg={int((frame.sentiment == 0).sum())})")
        print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}")
        mlflow.log_params({"train_rows": len(train_idx), "test_rows": len(test_idx)})
        mlflow.log_dict({"train_indices": train_idx.tolist(),
                         "test_indices": test_idx.tolist()}, "split.json")
        # Fit TF-IDF only on training rows; the test vocabulary is never fitted.
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), lowercase=True, min_df=1),
            LinearSVC(C=args.C, random_state=SEED),
        )
        pipe.fit(X_train, y_train)
        predictions = pipe.predict(X_test)
        accuracy = float(accuracy_score(y_test, predictions))
        passed = passes_gate(accuracy)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_dict({
            "accuracy": accuracy, "threshold": THRESHOLD,
            "passed": passed, "run_id": run.info.run_id,
        }, "validation.json")
        mlflow.log_dict(classification_report(
            y_test, predictions, labels=[0, 1], output_dict=True, zero_division=0
        ), "classification_report.json")
        mlflow.log_dict({"labels": [0, 1], "matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist()}, "confusion_matrix.json")
        print(f"MLflow run: {run.info.run_id}")
        print(f"Model Accuracy: {accuracy:.3f}")
        if not passed:
            print(f"Validation Failed: Accuracy is below the {THRESHOLD:.2f} threshold.")
            mlflow.set_tag("validation", "failed")
            mlflow.end_run(status="FAILED")
            return 1
        # Only an accepted model is logged; diagnostics remain for rejected runs.
        mlflow.sklearn.log_model(
            pipe, artifact_path="sentiment-model",
            signature=infer_signature(X_train, pipe.predict(X_train)),
            input_example=X_train.iloc[:3],
            pip_requirements=[f"{name}=={importlib.metadata.version(name)}" for name in
                              ("scikit-learn", "pandas", "numpy", "scipy", "cloudpickle")],
        )
        mlflow.set_tag("validation", "passed")
        print("Validation Passed: Accuracy is sufficient.")
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("C", nargs="?", type=float, default=1.0)
    parser.add_argument("--data", type=Path, default=Path("data/reviews.csv"))
    parser.add_argument("--tracking-dir", type=Path, default=Path("mlruns"))
    args = parser.parse_args()
    if not math.isfinite(args.C) or args.C <= 0:
        parser.error("C must be finite and greater than zero.")
    try:
        return train(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Training error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
