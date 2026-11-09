"""Baseline models for post-pass player trajectory prediction.

Every model follows the same small interface::

    model.fit(train_rows)            # rows from data.expand_to_frames / build_training_frame
    x_pred, y_pred = model.predict(rows)

Predictions are made in the normalized frame (offense moving right) and are
clipped to the field. Physics baselines need no training; the fitted
baselines learn a handful of parameters (or a residual model) from data.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, FIELD_X, FIELD_Y

Pred = Tuple[np.ndarray, np.ndarray]


def _clip(x, y) -> Pred:
    return np.clip(np.asarray(x, dtype=np.float64), 0, FIELD_X), np.clip(np.asarray(y, dtype=np.float64), 0, FIELD_Y)


class BaselineModel:
    name = "base"
    trainable = False

    def fit(self, rows: pd.DataFrame) -> "BaselineModel":
        return self

    def predict(self, rows: pd.DataFrame) -> Pred:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# --------------------------------------------------------------------------- physics baselines
class LastPosition(BaselineModel):
    """B0: the player freezes at the position of the last pre-pass frame."""
    name = "Last position"

    def predict(self, rows):
        return _clip(rows["x"], rows["y"])


class ConstantVelocity(BaselineModel):
    """B1: the player keeps the speed and direction measured at the pass."""
    name = "Constant velocity"

    def predict(self, rows):
        return _clip(rows["x"] + rows["cv_dx"], rows["y"] + rows["cv_dy"])


class ConstantAcceleration(BaselineModel):
    """B2: constant acceleration, using the acceleration vector estimated from
    the change in velocity over the last frames (capped at ``max_speed``)."""
    name = "Constant acceleration"

    def __init__(self, max_speed: float = 11.0):
        self.max_speed = max_speed

    def predict(self, rows):
        t = rows["t"].to_numpy()
        vx0, vy0 = rows["vx"].to_numpy(), rows["vy"].to_numpy()
        ax, ay = rows["ax_est"].to_numpy(), rows["ay_est"].to_numpy()
        dx = vx0 * t + 0.5 * ax * t ** 2
        dy = vy0 * t + 0.5 * ay * t ** 2
        # do not allow the implied average speed to exceed a human limit
        avg = np.hypot(dx, dy) / np.maximum(t, 1e-6)
        scale = np.minimum(1.0, self.max_speed / np.maximum(avg, 1e-6))
        return _clip(rows["x"] + dx * scale, rows["y"] + dy * scale)


class DampedVelocity(BaselineModel):
    """B3: velocity decays exponentially, v(t) = v0 * exp(-k t).

    Displacement = v0 * (1 - exp(-k t)) / k. The single parameter ``k`` is
    fitted on the training rows by grid search.
    """
    name = "Damped velocity"
    trainable = True

    def __init__(self, k: Optional[float] = None, grid=np.linspace(0.0, 1.5, 61)):
        self.k = k
        self.grid = grid

    @staticmethod
    def _factor(t, k):
        return t if k < 1e-9 else (1 - np.exp(-k * t)) / k

    def fit(self, rows):
        t = rows["t"].to_numpy()
        ex = (rows["x_true"] - rows["x"]).to_numpy()
        ey = (rows["y_true"] - rows["y"]).to_numpy()
        vx, vy = rows["vx"].to_numpy(), rows["vy"].to_numpy()
        best = min(self.grid, key=lambda k: np.mean((ex - vx * self._factor(t, k)) ** 2 + (ey - vy * self._factor(t, k)) ** 2))
        self.k = float(best)
        return self

    def predict(self, rows):
        f = self._factor(rows["t"].to_numpy(), self.k)
        return _clip(rows["x"] + rows["vx"] * f, rows["y"] + rows["vy"] * f)

    def __repr__(self):
        return f"DampedVelocity(k={self.k})"


class StraightToBall(BaselineModel):
    """B4: every player runs in a straight line and reaches the ball landing
    spot exactly when the ball arrives."""
    name = "Straight to ball"

    def predict(self, rows):
        return _clip(rows["x"] + rows["ball_dx_lin"], rows["y"] + rows["ball_dy_lin"])


class RolePhysicsBlend(BaselineModel):
    """B5: per-role blend of damped velocity and straight-to-ball motion.

    For each player role r and flight-progress bin b a weight w[r, b] is
    fitted by least squares:  disp = (1 - w) * damped + w * to_ball.
    Captures, e.g., that targeted receivers bend towards the ball while other
    route runners mostly keep running.
    """
    name = "Role-aware physics blend"
    trainable = True

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.damped = DampedVelocity()
        self.w = {}

    def _parts(self, rows):
        f = DampedVelocity._factor(rows["t"].to_numpy(), self.damped.k)
        dvx, dvy = rows["vx"].to_numpy() * f, rows["vy"].to_numpy() * f
        bx, by = rows["ball_dx_lin"].to_numpy(), rows["ball_dy_lin"].to_numpy()
        bins = np.minimum((rows["t_frac"].to_numpy() * self.n_bins).astype(int), self.n_bins - 1)
        return dvx, dvy, bx, by, bins

    def fit(self, rows):
        self.damped.fit(rows)
        dvx, dvy, bx, by, bins = self._parts(rows)
        ex = rows["x_true"].to_numpy() - rows["x"].to_numpy()
        ey = rows["y_true"].to_numpy() - rows["y"].to_numpy()
        role = rows["role_id"].to_numpy()
        for r in np.unique(role):
            for b in range(self.n_bins):
                m = (role == r) & (bins == b)
                if m.sum() < 50:
                    self.w[(r, b)] = 0.0
                    continue
                # minimise |e - d - w (bl - d)|^2  ->  closed form
                ux, uy = bx[m] - dvx[m], by[m] - dvy[m]
                rx, ry = ex[m] - dvx[m], ey[m] - dvy[m]
                w = (ux @ rx + uy @ ry) / max(ux @ ux + uy @ uy, 1e-9)
                self.w[(r, b)] = float(np.clip(w, 0.0, 1.0))
        return self

    def predict(self, rows):
        dvx, dvy, bx, by, bins = self._parts(rows)
        role = rows["role_id"].to_numpy()
        w = np.array([self.w.get((r, b), 0.0) for r, b in zip(role, bins)])
        return _clip(rows["x"] + (1 - w) * dvx + w * bx, rows["y"] + (1 - w) * dvy + w * by)


# --------------------------------------------------------------------------- learned residual baselines
class _ResidualModel(BaselineModel):
    """Learns the correction on top of constant velocity:
    x = x0 + vx t + f_x(features),  y = y0 + vy t + f_y(features)."""
    trainable = True
    features = FEATURES

    def _make(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def _X(self, rows):
        return rows[self.features].astype(np.float32).fillna(0.0).to_numpy()

    def fit(self, rows):
        X = self._X(rows)
        rx = (rows["x_true"] - rows["x"] - rows["cv_dx"]).to_numpy()
        ry = (rows["y_true"] - rows["y"] - rows["cv_dy"]).to_numpy()
        self.mx, self.my = self._make().fit(X, rx), self._make().fit(X, ry)
        return self

    def predict(self, rows):
        X = self._X(rows)
        return _clip(rows["x"] + rows["cv_dx"] + self.mx.predict(X), rows["y"] + rows["cv_dy"] + self.my.predict(X))


class RidgeResidual(_ResidualModel):
    """B6: linear (ridge) regression on the residual. Each feature is also
    multiplied by t so the correction can grow with time."""
    name = "Ridge residual"

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def _X(self, rows):
        X = super()._X(rows)
        t = rows["t"].to_numpy(np.float32)[:, None]
        return np.hstack([X, X * t])

    def _make(self):
        return make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))


class GBDTResidual(_ResidualModel):
    """B7: histogram gradient boosting on the residual (scikit-learn)."""
    name = "GBDT residual"

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.1, max_leaf_nodes: int = 63,
                 random_state: int = 42):
        self.params = dict(max_iter=max_iter, learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
                           min_samples_leaf=40, l2_regularization=1.0, random_state=random_state,
                           early_stopping=False)

    def _make(self):
        return HistGradientBoostingRegressor(**self.params)


ALL_BASELINES = [LastPosition, ConstantVelocity, ConstantAcceleration, DampedVelocity, StraightToBall,
                 RolePhysicsBlend, RidgeResidual, GBDTResidual]
