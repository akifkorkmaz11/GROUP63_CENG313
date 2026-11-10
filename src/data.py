"""Data loading, field normalization and feature tables for the
NFL Big Data Bowl 2026 (Prediction) competition.

Conventions
-----------
* Field: x in [0, 120] yards (including end zones), y in [0, 53.3] yards.
* Tracking is sampled at 10 Hz, so one frame = 0.1 s.
* NGS angles: 0 deg points to +y and angles grow clockwise, so a player
  moving with speed ``s`` and direction ``dir`` has
  ``vx = s * sin(dir)`` and ``vy = s * cos(dir)``.
* All modelling happens in a *normalized* frame where the offense always
  moves towards increasing x (plays with ``play_direction == "left"`` are
  rotated by 180 degrees). Predictions are rotated back at the end.
"""
from __future__ import annotations

import glob
import os
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

FIELD_X = 120.0
FIELD_Y = 53.3
FPS = 10.0
DT = 1.0 / FPS

PLAY_KEYS = ["game_id", "play_id"]
PLAYER_KEYS = PLAY_KEYS + ["nfl_id"]
FRAME_KEYS = PLAYER_KEYS + ["frame_id"]

COMPETITION = "nfl-big-data-bowl-2026-prediction"

ROLE_MAP = {"Targeted Receiver": 0, "Defensive Coverage": 1, "Other Route Runner": 2, "Passer": 3}
POSITIONS = ["QB", "RB", "FB", "WR", "TE", "CB", "S", "FS", "SS", "ILB", "OLB", "MLB", "LB", "DE", "DT", "NT"]
POS_MAP = {p: i for i, p in enumerate(POSITIONS)}

_INPUT_DTYPES = {
    "game_id": "int64", "play_id": "int32", "nfl_id": "int32", "frame_id": "int16",
    "absolute_yardline_number": "int16", "player_weight": "int16", "num_frames_output": "int16",
    "x": "float32", "y": "float32", "s": "float32", "a": "float32", "dir": "float32", "o": "float32",
    "ball_land_x": "float32", "ball_land_y": "float32",
    "play_direction": "category", "player_position": "category", "player_side": "category",
    "player_role": "category",
}
_OUTPUT_DTYPES = {"game_id": "int64", "play_id": "int32", "nfl_id": "int32", "frame_id": "int16",
                  "x": "float32", "y": "float32"}


# --------------------------------------------------------------------------- loading
def find_data_dir(candidates: Optional[Iterable[str]] = None) -> str:
    """Return the first directory that contains the competition's ``train/`` folder."""
    env = os.environ.get("NFL_DATA_DIR")
    paths = [env] if env else []
    paths += list(candidates or []) + [
        f"/kaggle/input/{COMPETITION}",
        f"/kaggle/input/competitions/{COMPETITION}",
        f"/content/{COMPETITION}",
        f"/content/drive/MyDrive/{COMPETITION}",
        f"./data/{COMPETITION}",
        f"./{COMPETITION}",
        f"../{COMPETITION}",
    ]
    for p in paths:
        if p and os.path.isdir(os.path.join(p, "train")):
            return os.path.abspath(p)
    # Kaggle may mount competition data in a sub-folder (e.g. /kaggle/input/competitions/<slug>)
    for pattern in ["/kaggle/input/*/train/input_2023_w01.csv", "/kaggle/input/*/*/train/input_2023_w01.csv"]:
        hits = sorted(glob.glob(pattern))
        if hits:
            return os.path.dirname(os.path.dirname(hits[0]))
    raise FileNotFoundError(
        "Competition data not found. Set NFL_DATA_DIR or place the data in ./data/" + COMPETITION)


def _week_files(data_dir: str, kind: str, weeks: Optional[Iterable[int]]) -> list:
    files = sorted(glob.glob(os.path.join(data_dir, "train", f"{kind}_2023_w*.csv")))
    if weeks is not None:
        wanted = {f"w{int(w):02d}" for w in weeks}
        files = [f for f in files if os.path.basename(f).split("_")[-1][:3] in wanted]
    if not files:
        raise FileNotFoundError(f"No {kind} files in {data_dir}/train")
    return files


def load_train(data_dir: str, weeks: Optional[Iterable[int]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the weekly input (pre-pass) and output (post-pass) files.

    A ``week`` column is added to both frames so results can be sliced by week.
    """
    ins, outs = [], []
    for f in _week_files(data_dir, "input", weeks):
        df = pd.read_csv(f, dtype=_INPUT_DTYPES)
        df["week"] = np.int8(os.path.basename(f)[-6:-4])
        ins.append(df)
    for f in _week_files(data_dir, "output", weeks):
        df = pd.read_csv(f, dtype=_OUTPUT_DTYPES)
        df["week"] = np.int8(os.path.basename(f)[-6:-4])
        outs.append(df)
    inp = pd.concat(ins, ignore_index=True)
    out = pd.concat(outs, ignore_index=True)
    for c in ["play_direction", "player_position", "player_side", "player_role"]:
        inp[c] = inp[c].astype("category")
    return inp, out


def load_test(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the public ``test.csv`` (rows to predict) and ``test_input.csv``."""
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    test_input = pd.read_csv(os.path.join(data_dir, "test_input.csv"), dtype=_INPUT_DTYPES)
    return test, test_input


# --------------------------------------------------------------------------- normalization
def normalize_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Rotate left-moving plays by 180 degrees so the offense always moves right.

    Adds ``is_left`` (1 for rotated plays) and ``los_x`` (line of scrimmage in the
    normalized frame). ``absolute_yardline_number`` is given in the original x
    coordinate, so it is rotated together with x.
    """
    df = df.copy()
    left = (df["play_direction"].astype(str) == "left").to_numpy()
    df["is_left"] = left.astype(np.int8)
    for c, L in [("x", FIELD_X), ("y", FIELD_Y), ("ball_land_x", FIELD_X), ("ball_land_y", FIELD_Y)]:
        if c in df:
            df[c] = np.where(left, L - df[c], df[c]).astype(np.float32)
    for c in ["dir", "o"]:
        if c in df:
            df[c] = np.where(left, (df[c] + 180.0) % 360.0, df[c]).astype(np.float32)
    if "absolute_yardline_number" in df:
        df["los_x"] = np.where(left, FIELD_X - df["absolute_yardline_number"],
                               df["absolute_yardline_number"]).astype(np.float32)
    return df


def flip_xy(x: np.ndarray, y: np.ndarray, is_left: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Apply (or undo - the transform is its own inverse) the 180 degree rotation."""
    left = np.asarray(is_left).astype(bool)
    return np.where(left, FIELD_X - x, x), np.where(left, FIELD_Y - y, y)


def height_to_inches(h) -> float:
    try:
        feet, inch = str(h).split("-")
        return int(feet) * 12 + int(inch)
    except (ValueError, AttributeError):
        return np.nan


def angle_diff(a, b):
    """Absolute smallest difference between two angles in degrees, in [0, 180]."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


# --------------------------------------------------------------------------- feature tables
def add_kinematics(df: pd.DataFrame) -> pd.DataFrame:
    """Velocity / acceleration components from s, a, dir and from position differences."""
    df = df.sort_values(FRAME_KEYS).copy()
    rad = np.deg2rad(df["dir"].to_numpy())
    df["vx"] = (df["s"].to_numpy() * np.sin(rad)).astype(np.float32)
    df["vy"] = (df["s"].to_numpy() * np.cos(rad)).astype(np.float32)
    g = df.groupby(PLAYER_KEYS, sort=False, observed=True)
    df["vx_pos"] = ((df["x"] - g["x"].shift(1)) / DT).astype(np.float32)
    df["vy_pos"] = ((df["y"] - g["y"].shift(1)) / DT).astype(np.float32)
    # acceleration vector from the change in velocity over the last 2 frames
    df["ax_est"] = ((df["vx"] - g["vx"].shift(2)) / (2 * DT)).astype(np.float32)
    df["ay_est"] = ((df["vy"] - g["vy"].shift(2)) / (2 * DT)).astype(np.float32)
    df["s_lag5"] = g["s"].shift(5)
    df["dir_lag5"] = g["dir"].shift(5)
    return df


def build_player_table(inp: pd.DataFrame, only_to_predict: bool = True) -> pd.DataFrame:
    """One row per player at the last pre-pass frame, with basic features.

    The features are intentionally *simple* (kinematics, ball landing spot,
    player attributes). Interaction features between players are left for the
    proposed solution in Assignment 2.
    """
    df = add_kinematics(normalize_direction(inp))
    last = df.groupby(PLAYER_KEYS, sort=False, observed=True).tail(1).copy()
    last = last.rename(columns={"frame_id": "last_input_frame"})

    last["s_change5"] = last["s"] - last["s_lag5"]
    last["dir_change5"] = (last["dir"] - last["dir_lag5"] + 180.0) % 360.0 - 180.0
    for c in ["vx_pos", "vy_pos", "ax_est", "ay_est", "s_change5", "dir_change5"]:
        last[c] = last[c].fillna(0.0)

    T = last["num_frames_output"].to_numpy() * DT
    last["flight_time"] = T
    last["dx_ball"] = last["ball_land_x"] - last["x"]
    last["dy_ball"] = last["ball_land_y"] - last["y"]
    last["dist_ball"] = np.hypot(last["dx_ball"], last["dy_ball"])
    last["angle_to_ball"] = np.rad2deg(np.arctan2(last["dx_ball"], last["dy_ball"])) % 360.0
    last["dir_vs_ball"] = angle_diff(last["dir"], last["angle_to_ball"])
    last["o_vs_ball"] = angle_diff(last["o"], last["angle_to_ball"])
    last["req_speed"] = last["dist_ball"] / np.maximum(T, DT)
    last["closing_speed"] = (last["vx"] * last["dx_ball"] + last["vy"] * last["dy_ball"]) / (last["dist_ball"] + 1e-6)

    last["height_in"] = last["player_height"].map(height_to_inches)
    birth = pd.to_datetime(last["player_birth_date"], errors="coerce")
    last["age"] = (pd.Timestamp("2023-09-07") - birth).dt.days / 365.25
    last["role_id"] = last["player_role"].astype(str).map(ROLE_MAP).fillna(-1).astype(np.int8)
    last["pos_id"] = last["player_position"].astype(str).map(POS_MAP).fillna(-1).astype(np.int8)
    last["is_offense"] = (last["player_side"].astype(str) == "Offense").astype(np.int8)
    last["x_from_los"] = last["x"] - last["los_x"]
    last["dist_sideline"] = np.minimum(last["y"], FIELD_Y - last["y"])

    if only_to_predict:
        last = last[last["player_to_predict"].astype(bool)]
    return last.reset_index(drop=True)


PLAYER_FEATURES = [
    "x", "y", "s", "a", "dir", "o", "vx", "vy", "vx_pos", "vy_pos", "ax_est", "ay_est",
    "s_change5", "dir_change5", "flight_time", "num_frames_output", "ball_land_x", "ball_land_y",
    "dx_ball", "dy_ball", "dist_ball", "angle_to_ball", "dir_vs_ball", "o_vs_ball", "req_speed",
    "closing_speed", "height_in", "player_weight", "age", "role_id", "pos_id", "is_offense",
    "x_from_los", "dist_sideline",
]
FRAME_FEATURES = ["frame_id", "t", "t_frac", "frames_left", "cv_dx", "cv_dy", "ball_dx_lin", "ball_dy_lin"]
FEATURES = PLAYER_FEATURES + FRAME_FEATURES


def expand_to_frames(players: pd.DataFrame, frames: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Create one row per (player, output frame).

    If ``frames`` (with ``FRAME_KEYS``) is given, exactly those rows are produced
    (in that order); otherwise frames ``1..num_frames_output`` are generated.
    """
    if frames is None:
        reps = players["num_frames_output"].astype(int).to_numpy()
        rows = players.loc[players.index.repeat(reps)].reset_index(drop=True)
        rows["frame_id"] = np.concatenate([np.arange(1, r + 1) for r in reps]).astype(np.int16)
    else:
        rows = frames[FRAME_KEYS].merge(players, on=PLAYER_KEYS, how="left")
    t = rows["frame_id"].to_numpy() * DT
    rows["t"] = t
    rows["t_frac"] = rows["frame_id"] / rows["num_frames_output"]
    rows["frames_left"] = rows["num_frames_output"] - rows["frame_id"]
    rows["cv_dx"] = rows["vx"] * t
    rows["cv_dy"] = rows["vy"] * t
    rows["ball_dx_lin"] = rows["dx_ball"] * np.minimum(rows["t_frac"], 1.0)
    rows["ball_dy_lin"] = rows["dy_ball"] * np.minimum(rows["t_frac"], 1.0)
    return rows


def build_training_frame(inp: pd.DataFrame, out: pd.DataFrame) -> pd.DataFrame:
    """Join the player table with the post-pass targets (in the normalized frame)."""
    players = build_player_table(inp)
    rows = expand_to_frames(players, frames=out)
    tx, ty = flip_xy(out["x"].to_numpy(), out["y"].to_numpy(), rows["is_left"].to_numpy())
    rows["x_true"] = tx.astype(np.float32)
    rows["y_true"] = ty.astype(np.float32)
    return rows
