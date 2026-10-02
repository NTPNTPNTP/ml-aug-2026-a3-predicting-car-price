"""
Unit tests for the A3 Logistic Regression model.

Run with:
    pytest tests/test_model.py -v

Two tests are required by the assignment:
  1. test_model_accepts_expected_input  — model takes the expected input (shape & dtype)
  2. test_model_output_shape            — output of the model has the expected shape
"""

import sys
import types
import numpy as np
import joblib
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_PATH = "model/a3_model.pkl"

# Number of features the model was trained on: ["max_power", "year"]
N_FEATURES = 2
# Number of output classes (4 price tiers)
N_CLASSES = 4


def _load_bundle():
    """
    Load the saved joblib bundle.

    The model was pickled while the custom classes lived in __main__
    (inside the notebook).  We re-register them under __main__ before
    calling joblib.load so the unpickler can resolve them.
    """
    from app.model_classes import LogisticRegression, RidgePenalty

    main_mod = sys.modules.setdefault("__main__", types.ModuleType("__main__"))
    main_mod.LogisticRegression = LogisticRegression
    main_mod.RidgePenalty = RidgePenalty

    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def bundle():
    """Shared model bundle loaded once for the whole test module."""
    return _load_bundle()


@pytest.fixture(scope="module")
def model(bundle):
    return bundle["model"]


@pytest.fixture(scope="module")
def scaler(bundle):
    return bundle["scaler"]


# ---------------------------------------------------------------------------
# Test 1 — The model accepts the expected input
# ---------------------------------------------------------------------------

def test_model_accepts_expected_input(model, scaler):
    """
    The model must accept a 2-D numpy array of shape (m, N_FEATURES) after
    standard-scaling, and must not raise any exception.

    This covers:
      • Correct number of features  (n = 2: max_power, year)
      • Correct dtype               (float64 / compatible numeric)
      • Correct dimensionality      (2-D array)
    """
    # Typical raw values for (max_power, year)
    raw_input = np.array([
        [82.0, 2015],   # sample 1
        [120.5, 2018],  # sample 2
        [55.0, 2010],   # sample 3
    ], dtype=float)

    assert raw_input.shape[1] == N_FEATURES, (
        f"Input must have {N_FEATURES} features, got {raw_input.shape[1]}"
    )

    # Scale exactly as the pipeline does at inference time
    scaled = scaler.transform(raw_input)

    # Must not raise
    predictions = model.predict(scaled)

    assert predictions is not None, "model.predict() returned None"


# ---------------------------------------------------------------------------
# Test 2 — The output of the model has the expected shape
# ---------------------------------------------------------------------------

def test_model_output_shape(model, scaler):
    """
    For an input of m samples the model must return a 1-D array of shape (m,)
    whose values are integer class indices in the range [0, N_CLASSES - 1].
    """
    m = 5  # batch size used for this test

    raw_input = np.array([
        [82.0,  2015],
        [120.5, 2018],
        [55.0,  2010],
        [200.0, 2020],
        [40.0,  2008],
    ], dtype=float)

    scaled = scaler.transform(raw_input)
    predictions = model.predict(scaled)

    # Shape: must be a 1-D array with one prediction per sample
    assert predictions.ndim == 1, (
        f"Expected 1-D output, got ndim={predictions.ndim}"
    )
    assert predictions.shape == (m,), (
        f"Expected output shape ({m},), got {predictions.shape}"
    )

    # Values: each predicted class index must be a valid class
    assert np.all(predictions >= 0) and np.all(predictions < N_CLASSES), (
        f"Predicted class indices must be in [0, {N_CLASSES - 1}], "
        f"got values: {predictions}"
    )
