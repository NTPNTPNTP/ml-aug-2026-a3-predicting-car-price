"""
A3 FastAPI backend — Car Price Tier Classifier.

Pulls the best model from the local MLflow registry (registered as a pyfunc
model with built-in scaler and class-name lookup), then serves it through a
single POST /predict endpoint.

Endpoints
---------
GET  /          → index.html
POST /predict   → { "tier_index": int, "price_range": str }
"""

import sys
import types
import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# ---------------------------------------------------------------------------
# Re-register the custom classes under __main__ so mlflow's internal joblib
# unpickling (skops) can resolve them when loading from the registry.
# ---------------------------------------------------------------------------
from . import model_classes

_main = sys.modules.setdefault("__main__", types.ModuleType("__main__"))
_main.LogisticRegression = model_classes.LogisticRegression
_main.RidgePenalty = model_classes.RidgePenalty

import pathlib
import joblib
import pandas as pd

# ---------------------------------------------------------------------------
# Load model: Try local MLflow registry first, fallback to saved model bundle
# ---------------------------------------------------------------------------
MODEL_NAME = "st126894-a3-model"
MODEL_URI  = f"models:/{MODEL_NAME}/1"

BASE_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
MLFLOW_DB = ROOT_DIR / "mlflow.db"
MODEL_PKL = ROOT_DIR / "model" / "a3_model.pkl"

loaded_model = None

if MLFLOW_DB.exists():
    try:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
        loaded_model = mlflow.pyfunc.load_model(MODEL_URI)
        print(f"Successfully loaded '{MODEL_URI}' from MLflow registry.")
    except Exception as exc:
        print(f"MLflow registry load failed ({exc}), attempting local bundle fallback...")

if loaded_model is None:
    if MODEL_PKL.exists():
        class LocalLogRegWrapper:
            def __init__(self, model, scaler, feature_names, class_names):
                self.model = model
                self.scaler = scaler
                self.feature_names = feature_names
                self.class_names = class_names

            def predict(self, model_input):
                X = pd.DataFrame(model_input)[self.feature_names].astype(float)
                X_scaled = self.scaler.transform(X)
                pred = self.model.predict(X_scaled)
                return pd.DataFrame({
                    "price_tier": pred,
                    "price_range": [self.class_names[p] for p in pred],
                })

        bundle = joblib.load(MODEL_PKL)
        loaded_model = LocalLogRegWrapper(
            bundle["model"],
            bundle["scaler"],
            bundle["feature_names"],
            bundle["class_names"]
        )
        print(f"Successfully loaded model from bundle: {MODEL_PKL}")
    else:
        raise RuntimeError(
            f"Could not load model from MLflow registry ('{MODEL_URI}') or local bundle ('{MODEL_PKL}')."
        )

# ---------------------------------------------------------------------------
# Fallback medians for imputation
# ---------------------------------------------------------------------------
MAX_POWER_MEDIAN = 82.85
YEAR_MEDIAN      = 2015

# ---------------------------------------------------------------------------
STATIC_DIR = BASE_DIR / "static"
app = FastAPI(title="A3 — Car Price Tier Classifier")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class CarInput(BaseModel):
    max_power: Optional[float] = None
    year:      Optional[int]   = None


@app.get("/")
def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/predict")
def predict(data: CarInput):
    """
    Predict car price tier.

    The pyfunc wrapper (LogRegWrapper) internally applies StandardScaler,
    so we pass raw (unscaled) feature values, exactly like the notebook demo.
    """
    max_power = data.max_power if data.max_power is not None else MAX_POWER_MEDIAN
    year      = data.year      if data.year      is not None else YEAR_MEDIAN

    import pandas as pd
    input_df = pd.DataFrame([{"max_power": max_power, "year": year}])

    try:
        result_df = loaded_model.predict(input_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    tier_index  = int(result_df["price_tier"].iloc[0])
    price_range = str(result_df["price_range"].iloc[0])

    print(f"Input: max_power={max_power}, year={year}")
    print(f"Prediction: tier={tier_index}, range={price_range}")

    return {"tier_index": tier_index, "price_range": price_range}
