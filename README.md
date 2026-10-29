# Sentiment Model CI Pipeline

A Coursera lab project with TF-IDF, LinearSVC, a 60% accuracy gate, and MLflow tracking.

## Status
Prepared source files. Training, tests, and GitHub Actions have not been executed or verified as part of this deliverable.

## Files
- train.py: training, evaluation, input validation, and quality gate.
- data/reviews.csv: supplied standard dataset.
- data/reviews_fail.csv: supplied failure demonstration dataset.
- requirements.txt: fixed direct dependency versions.
- .github/workflows/main.yml: push, pull request, and manual CI triggers.
- tests/test_quality_gate.py: boundary and malformed-input checks.

## Local usage (optional)
Use Python 3.10 or 3.12 in a virtual environment, then run:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python train.py

For a failure demonstration:

    python train.py --data data/reviews_fail.csv

Actual results must be inspected; this README makes no claim that either dataset has been run.

## GitHub setup
1. Create a public repository named cicd-ml-pipeline.
2. Extract the supplied ZIP on your computer.
3. Upload the extracted contents, including .github and .gitignore, to the repository root.
4. Commit to main. This triggers training automatically on GitHub.
5. Open Actions and inspect Model Training CI Pipeline.
6. On the run summary, download mlflow-run-results from Artifacts.
7. Include your actual repository URL in the submission report.

The upload step retains available tracking records on a failed run. A failed training or quality-gate step remains a failed job. Local mlruns files are ignored by Git.

To prevent merging a failed model update, configure train-and-validate as a required pull request check in repository rules.
