"""
Custom model classes for A3 — Logistic Regression for Car Price Tier Classification.

Originally defined in experiments.ipynb (pickled under __main__).  Re-defined here
so the saved model can be loaded by the API without running the notebook.

Only inference-time code is kept; training helpers (mlflow calls, fit loop) are
preserved intact so the class is a faithful copy of the notebook version.
"""

import time
import numpy as np
import pandas as pd


class RidgePenalty:
    """L2 regularisation penalty used during training."""

    def __init__(self, l: float):
        self.l = l

    def __call__(self, theta: np.ndarray) -> float:
        return self.l * np.sum(np.square(theta))

    def derivation(self, theta: np.ndarray) -> np.ndarray:
        return self.l * 2 * theta


class LogisticRegression:
    """
    Custom multiclass logistic regression with softmax output.

    Parameters
    ----------
    k : int
        Number of output classes.
    n : int
        Number of input features.
    method : str
        Gradient-descent variant: ``"batch"``, ``"minibatch"``, or ``"sto"``.
    alpha : float
        Learning rate (default 0.001).
    max_iter : int
        Number of training iterations (default 5000).
    regularization : callable or None
        A regularisation object (e.g. :class:`RidgePenalty`) or ``None``.
    """

    def __init__(self, k: int, n: int, method: str,
                 alpha: float = 0.001, max_iter: int = 5000,
                 regularization=None):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.regularization = regularization

    # ------------------------------------------------------------------
    # Core maths
    # ------------------------------------------------------------------

    def softmax(self, theta_t_x: np.ndarray) -> np.ndarray:
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X: np.ndarray, error: np.ndarray) -> np.ndarray:
        return X.T @ error

    def h_theta(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Parameters
        ----------
        X : (m, n)
        W : (n, k)
        b : (k,)

        Returns
        -------
        yhat : (m, k)
        """
        return self.softmax(X @ W + b)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return predicted class index for each sample. Output shape: (m,)."""
        return np.argmax(self.h_theta(X_test, self.W, self.b), axis=1)

    def _pred_onehot(self, X: np.ndarray) -> np.ndarray:
        h = self.h_theta(X, self.W, self.b)
        return np.eye(h.shape[1])[np.argmax(h, axis=1)]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.W = np.random.rand(self.n, self.k)
        self.b = np.zeros(self.k)
        self.losses = []

        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad_W, grad_b = self.gradient(X, Y)
                self._update(loss, grad_W, grad_b, i)
            print(f"time taken: {time.time() - start_time}")

        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0])
                batch_X = X[ix:ix + batch_size]
                batch_Y = Y[ix:ix + batch_size]
                loss, grad_W, grad_b = self.gradient(batch_X, batch_Y)
                self._update(loss, grad_W, grad_b, i)
            print(f"time taken: {time.time() - start_time}")

        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix: set = set()
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while idx in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad_W, grad_b = self.gradient(X_train, Y_train)
                self._update(loss, grad_W, grad_b, i)
                list_of_used_ix.add(idx)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = set()
            print(f"time taken: {time.time() - start_time}")

        else:
            raise ValueError('Method must be "batch", "minibatch", or "sto".')

    def gradient(self, X: np.ndarray, Y: np.ndarray):
        m = X.shape[0]
        h = self.h_theta(X, self.W, self.b)
        loss = -np.sum(Y * np.log(h + 1e-12)) / m
        error = h - Y
        grad_W = self.softmax_grad(X, error) / m
        grad_b = error.sum(axis=0) / m

        if self.regularization is not None:
            loss += self.regularization(self.W)
            grad_W += self.regularization.derivation(self.W)

        return loss, grad_W, grad_b

    def _update(self, loss, grad_W, grad_b, i):
        self.losses.append(loss)
        self.W -= self.alpha * grad_W
        self.b -= self.alpha * grad_b
        if i % 500 == 0:
            print(f"Loss at iteration {i}", loss)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_divide(numerator, denominator):
        numerator = np.asarray(numerator, dtype=float)
        denominator = np.asarray(denominator, dtype=float)
        return np.divide(numerator, denominator,
                         out=np.zeros_like(numerator),
                         where=denominator != 0)[()]

    def _confusion_counts(self, X, Y, c):
        h = self._pred_onehot(X)[:, c]
        y = Y[:, c]
        tp = np.sum((h == 1) & (y == 1))
        fp = np.sum((h == 1) & (y == 0))
        fn = np.sum((h == 0) & (y == 1))
        tn = np.sum((h == 0) & (y == 0))
        return tp, fp, fn, tn

    def _all_class_confusion_counts(self, X, Y):
        H = self._pred_onehot(X)
        tp = np.sum((H == 1) & (Y == 1), axis=0)
        fp = np.sum((H == 1) & (Y == 0), axis=0)
        fn = np.sum((H == 0) & (Y == 1), axis=0)
        tn = np.sum((H == 0) & (Y == 0), axis=0)
        return tp, fp, fn, tn

    def _class_weights(self, Y):
        return Y.sum(axis=0) / Y.sum()

    def macro_accuracy_score(self, X, Y):
        tp, fp, fn, tn = self._all_class_confusion_counts(X, Y)
        return np.mean(self._safe_divide(tp + tn, tp + fp + fn + tn))

    def macro_precision_score(self, X, Y):
        tp, fp, _, _ = self._all_class_confusion_counts(X, Y)
        return np.mean(self._safe_divide(tp, tp + fp))

    def macro_recall_score(self, X, Y):
        tp, _, fn, _ = self._all_class_confusion_counts(X, Y)
        return np.mean(self._safe_divide(tp, tp + fn))

    def macro_f1_score(self, X, Y):
        tp, fp, fn, _ = self._all_class_confusion_counts(X, Y)
        return np.mean(self._safe_divide(2 * tp, 2 * tp + fp + fn))

    def weighted_accuracy_score(self, X, Y):
        tp, fp, fn, tn = self._all_class_confusion_counts(X, Y)
        return np.sum(self._class_weights(Y) * self._safe_divide(tp + tn, tp + fp + fn + tn))

    def weighted_precision_score(self, X, Y):
        tp, fp, _, _ = self._all_class_confusion_counts(X, Y)
        return np.sum(self._class_weights(Y) * self._safe_divide(tp, tp + fp))

    def weighted_recall_score(self, X, Y):
        tp, _, fn, _ = self._all_class_confusion_counts(X, Y)
        return np.sum(self._class_weights(Y) * self._safe_divide(tp, tp + fn))

    def weighted_f1_score(self, X, Y):
        tp, fp, fn, _ = self._all_class_confusion_counts(X, Y)
        return np.sum(self._class_weights(Y) * self._safe_divide(2 * tp, 2 * tp + fp + fn))

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, feature_names, per_class=False, class_names=None):
        W = self.W - self.W.mean(axis=1, keepdims=True)
        if per_class:
            cols = class_names if class_names is not None else range(W.shape[1])
            df = pd.DataFrame(W, index=feature_names, columns=cols)
            order = df.abs().max(axis=1).sort_values(ascending=False).index
            return df.loc[order]
        return pd.Series(np.abs(W).mean(axis=1), index=feature_names).sort_values(ascending=False)

    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.losses)), self.losses, label="Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()
