# Car Price Tier Classification App (Assignment 3)

Web application that classifies used cars into **4 price tiers** based on **manufacturing year** and **maximum power (bhp)**, using custom Logistic Regression with Ridge Regularization trained on the `Cars.csv` dataset, tracked with **MLflow**, and deployed via automated **CI/CD** pipelines.

---

## Features

- **Classification into 4 Price Tiers**:
  - **Tier 0**: Low (`29,999 - 260,000`)
  - **Tier 1**: Medium (`260,000 - 450,000`)
  - **Tier 2**: High (`450,000 - 680,000`)
  - **Tier 3**: Very High (`680,000 - 10,000,000`)
- **Interactive Web Interface**: Enter year and max power (missing inputs default to training dataset medians).
- **MLflow Registry & pyfunc Wrapper**: Model registered with embedded scaler and class name lookup.
- **Automated Testing & CI/CD**: Pytest unit tests for input/output verification executed automatically via GitHub Actions upon every push.

---

## Tech Stack

- **Backend**: FastAPI + Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript (served via FastAPI static mount)
- **Model**: Custom Multiclass Logistic Regression (Softmax + Ridge Penalty)
- **Experiment Tracking & Registry**: MLflow (`sqlite:///mlflow.db`)
- **Testing**: pytest
- **CI/CD**: GitHub Actions (`.github/workflows/ci-cd.yml`)
- **Package Manager**: uv
- **Containerization**: Docker + Docker Compose

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions CI/CD (pytest + Docker deploy)
├── app/
│   ├── main.py                # FastAPI backend & /predict endpoint
│   ├── model_classes.py       # Custom Logistic Regression & Ridge penalty
│   └── static/
│       └── index.html         # Frontend interface
├── model/
│   └── a3_model.pkl           # Saved model bundle (model, scaler, metadata)
├── tests/
│   ├── __init__.py
│   └── test_model.py          # Unit tests (input validation & output shape)
├── data/
│   └── Cars.csv               # Dataset
├── experiments.ipynb          # EDA, training, custom classes, MLflow tracking
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── uv.lock
```

---

## Running Locally (without Docker)

```bash
# 1. Install dependencies
uv sync

# 2. Run the application
uv run uvicorn app.main:app --reload --port 8000
```
Then open http://localhost:8000 in your browser.

---

## Running Unit Tests

```bash
uv run pytest tests/ -v
```

---

## Running with Docker

```bash
docker compose up --build
```
Then open http://localhost:8000

---

## CI / CD Pipeline

- **Continuous Integration (CI)**: Runs `pytest tests/ -v` on every push and pull request.
- **Continuous Deployment (CD)**: Automatically builds and pushes the Docker container to Docker Hub, then triggers server redeployment when changes are merged into the main branch.
