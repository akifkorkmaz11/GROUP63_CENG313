"""Metric, cross-validation and plotting helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .data import FIELD_X, FIELD_Y, PLAY_KEYS


# --------------------------------------------------------------------------- metric
def rmse(x_true, y_true, x_pred, y_pred) -> float:
    """Competition metric: sqrt( sum[(dx)^2 + (dy)^2] / (2N) )."""
    dx = np.asarray(x_true) - np.asarray(x_pred)
    dy = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(dx ** 2 + dy ** 2) / 2.0))


def rmse_by(rows: pd.DataFrame, x_pred, y_pred, by) -> pd.Series:
    """RMSE broken down by one or more columns of ``rows``."""
    se = ((rows["x_true"].to_numpy() - x_pred) ** 2 + (rows["y_true"].to_numpy() - y_pred) ** 2) / 2.0
    return pd.Series(se, index=rows.index).groupby([rows[b] for b in np.atleast_1d(by)], observed=True).mean() ** 0.5


# --------------------------------------------------------------------------- validation
def group_kfold_oof(make_model: Callable, rows: pd.DataFrame, n_splits: int = 5, group_col: str = "game_id",
                    verbose: bool = True) -> Dict:
    """Out-of-fold predictions with folds split by game (no game in train and valid).

    Returns a dict with ``x``/``y`` OOF predictions, per-fold scores, the overall
    OOF RMSE, the fitted models and the elapsed time.
    """
    xo = np.zeros(len(rows))
    yo = np.zeros(len(rows))
    folds, models = [], []
    start = time.time()
    for k, (tr, va) in enumerate(GroupKFold(n_splits=n_splits).split(rows, groups=rows[group_col])):
        m = make_model().fit(rows.iloc[tr])
        px, py = m.predict(rows.iloc[va])
        xo[va], yo[va] = px, py
        s = rmse(rows["x_true"].iloc[va], rows["y_true"].iloc[va], px, py)
        folds.append(s)
        models.append(m)
        if verbose:
            print(f"  fold {k}: {s:.4f}")
    return {"x": xo, "y": yo, "folds": folds, "models": models,
            "rmse": rmse(rows["x_true"], rows["y_true"], xo, yo), "seconds": time.time() - start}


@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    print(f"[{label}] {time.time() - t0:.1f}s")


# --------------------------------------------------------------------------- plotting
def draw_field(ax, xlim=(0, FIELD_X), ylim=(0, FIELD_Y), turf="#2d6a4f"):
    import matplotlib.pyplot as plt
    ax.add_patch(plt.Rectangle((0, 0), FIELD_X, FIELD_Y, color=turf, zorder=0))
    for x in (0, 110):
        ax.add_patch(plt.Rectangle((x, 0), 10, FIELD_Y, color="#1b4332", zorder=0))
    for x in range(10, 111, 5):
        ax.plot([x, x], [0, FIELD_Y], color="white", lw=1.0 if x % 10 == 0 else 0.4, alpha=0.6, zorder=1)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def plot_play(inp: pd.DataFrame, out: pd.DataFrame, game_id: int, play_id: int, preds: Optional[pd.DataFrame] = None,
              ax=None, zoom: bool = True, title: Optional[str] = None):
    """Plot one play in original coordinates.

    Dotted: pre-pass path (input). Solid: true post-pass path (output).
    Dashed cyan: predicted post-pass path (``preds`` with nfl_id, x, y). Star: ball landing.
    """
    import matplotlib.pyplot as plt
    pi = inp[(inp.game_id == game_id) & (inp.play_id == play_id)]
    po = out[(out.game_id == game_id) & (out.play_id == play_id)]
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5.5))
    draw_field(ax)
    colors = {"Offense": "#ffd166", "Defense": "#a8dadc"}
    for nid, g in pi.groupby("nfl_id"):
        side = str(g.player_side.iloc[0])
        role = str(g.player_role.iloc[0])
        ax.plot(g.x, g.y, ":", color=colors.get(side, "white"), lw=1.5)
        o = po[po.nfl_id == nid]
        if len(o):
            ax.plot(o.x, o.y, "-", color="#e76f51" if role == "Targeted Receiver" else colors.get(side, "white"), lw=2.5)
        if preds is not None:
            p = preds[preds.nfl_id == nid]
            if len(p):
                ax.plot(p.x, p.y, "--", color="#00e5ff", lw=2)
    bx, by = pi.ball_land_x.iloc[0], pi.ball_land_y.iloc[0]
    ax.plot(bx, by, "*", ms=20, color="#e76f51", mec="white", zorder=5)
    if zoom:
        xs = pd.concat([pi.x, po.x, pd.Series([bx])])
        ys = pd.concat([pi.y, po.y, pd.Series([by])])
        ax.set_xlim(max(-1, xs.min() - 5), min(121, xs.max() + 5))
        ax.set_ylim(max(-1, ys.min() - 4), min(54.3, ys.max() + 4))
    ax.set_title(title or f"game {game_id} · play {play_id} · {pi.play_direction.iloc[0]} · "
                          f"{int(pi.num_frames_output.iloc[0])} frames in the air")
    return ax
