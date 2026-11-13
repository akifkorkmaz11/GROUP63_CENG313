# GROUP63_CENG313 · NFL Big Data Bowl 2026 – Prediction

> CENG313 term project · **Group 63** (single-member group) · repository in the course organization
> [`ceng313-2026`](https://github.com/ceng313-2026/GROUP63_CENG313).

**Current Kaggle score:** private RMSE **0.68028** (late submission – the competition is closed, so the score is
not ranked on the leaderboard). Best cross-validated baseline: **0.732** yd (GBDT residual).

## Team
| Name | Student ID | GitHub |
|---|---|---|
| Akif Korkmaz | 24118080032 | [@akifkorkmaz11](https://github.com/akifkorkmaz11) |

## Competition
**[NFL Big Data Bowl 2026 – Prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction)** (Kaggle, code competition).
The competition is **closed** (final scoring on 2025 games ended in January 2026); we evaluate with
**late submissions**, which receive public/private scores but no leaderboard position.

When the quarterback releases a pass, predict the `(x, y)` position of selected players (the targeted receiver
and defenders in coverage) for every frame (10 Hz) until the ball arrives. Inputs are each player's
tracking data before the throw, the identity of the targeted receiver and the ball landing location.

**Metric:** RMSE over both coordinates, `sqrt( Σ[(x−x̂)² + (y−ŷ)²] / 2N )`, in yards.

## Dataset
| File | Content |
|---|---|
| `train/input_2023_w01–w18.csv` | Pre-pass tracking for all 22 players (x, y, speed, acceleration, direction, orientation, role, position, height, weight, …), the ball landing spot and `num_frames_output` |
| `train/output_2023_w01–w18.csv` | Post-pass `(x, y)` of the players to predict – the targets |
| `test.csv`, `test_input.csv` | Public example of the test format (2024 games); the hidden test set is served by the evaluation API |

2023 regular season, 18 weeks · 272 games · 14,108 plays · 562,936 frames to predict · ~830 MB.

## Repository layout
```
GROUP63_CENG313/
├── README.md                        # group, members, competition, how to run, scores
├── requirements.txt                 # Python packages used
├── .gitignore                       # excludes data/
├── assignment1_eda_baseline.ipynb   # Assignment 1: EDA & baselines (executed in Colab, outputs saved)
├── assignment2_solution.ipynb       # Assignment 2: improved solution (due 30 Dec)
├── src/
│   ├── __init__.py                  # empty, makes src importable
│   ├── data.py                      # loading, direction normalization, preprocessing / feature tables
│   ├── models.py                    # baseline models with a common fit/predict interface
│   └── utils.py                     # metric, game-grouped cross-validation, plotting
├── data/                            # NOT committed – competition data is downloaded here (see below)
├── submissions/                     # Kaggle submission score screenshots
├── kaggle/submission_baseline.ipynb # offline Kaggle notebook used for the late submission (code competition)
├── tools/make_kaggle_notebook.py    # builds the Kaggle notebook from src/
├── results/                         # cross-validation result tables
└── reports/figures/                 # figures exported by the notebook
```

## Getting the data
The data is **not** in this repository (Kaggle rules and file size). Accept the competition rules on Kaggle, create an
API key (Kaggle → Settings → API → *Create Legacy API Key*, which downloads `kaggle.json`), then:
```bash
pip install -r requirements.txt
kaggle competitions download -c nfl-big-data-bowl-2026-prediction -p data
unzip data/nfl-big-data-bowl-2026-prediction.zip -d data/nfl-big-data-bowl-2026-prediction
```
The notebook reads the data from the relative path `data/nfl-big-data-bowl-2026-prediction/` (in Colab it runs
these steps automatically).

## How to run
**Google Colab (recommended)**
Direct link: [open `assignment1_eda_baseline.ipynb` in Colab](https://colab.research.google.com/github/ceng313-2026/GROUP63_CENG313/blob/main/assignment1_eda_baseline.ipynb)

1. Open `assignment1_eda_baseline.ipynb` in Colab with the *Open in Colab* badge at the top of the notebook
   (or *File → Open notebook → GitHub* → `ceng313-2026/GROUP63_CENG313`).
2. The first cell clones this repository (`REPO_URL`); keep the repository public or sign in to GitHub in Colab.
3. *Runtime → Run all* and upload `kaggle.json` when asked; the data is downloaded with the Kaggle API.
   The full run takes ~10–15 minutes on the free CPU runtime.

**Locally**
```bash
pip install -r requirements.txt   # after downloading the data as described above
jupyter notebook assignment1_eda_baseline.ipynb
```

## Approach (Assignment 1)
1. **EDA** – data overview (dtypes, missing values, duplicates, consistency), time in the air, pass depth, roles,
   kinematics at the throw, how players move while the ball is in the air, outliers and the target distribution.
2. **Preprocessing** – no rows dropped (no missing values/duplicates, outliers are real plays); direction normalization: rotate plays so the offense always moves towards +x; convert NGS angles to velocity
   components (`vx = s·sin(dir)`, `vy = s·cos(dir)`).
3. **Feature table** – one row per (player, output frame) with the last pre-pass state, ball-landing
   geometry, player attributes and time features.
4. **Baselines** – evaluated with 5-fold `GroupKFold` by game (the test set consists of unseen games).

## Results – Assignment 1 baselines
Out-of-fold RMSE (yards) over all 562,936 training frames, 5-fold GroupKFold by `game_id` (Google Colab run).

| # | Baseline | Idea | OOF RMSE | Fold std | Improvement vs constant velocity |
|---|---|---|---|---|---|
| B0 | Last position | player stops | 4.244 | 0.080 | -158.7% |
| B1 | Constant velocity | keep speed and direction | 1.640 | 0.069 | +0.0% |
| B2 | Constant acceleration | extrapolate the recent acceleration | 1.817 | 0.194 | -10.8% |
| B3 | Damped velocity | velocity decays as e^(−kt), k fitted | 1.660 | 0.058 | -1.2% |
| B4 | Straight to ball | run straight to the landing spot | 3.922 | 0.028 | -139.1% |
| B5 | Role-aware physics blend | per-role/phase blend of the two above (fitted weights) | 1.255 | 0.058 | +23.5% |
| B6 | Ridge residual | linear model of the constant-velocity residual | 0.907 | 0.115 | +44.7% |
| B7 | GBDT residual | gradient-boosted trees on the constant-velocity residual | **0.732** | 0.086 | +55.4% |

![Baselines](reports/figures/07_baseline_rmse.png)

Main EDA findings
- Ball in the air 1.0 s on median (10 frames), long tail up to 9.4 s; air time ≈ linear in pass length (r = 0.95).
- Per play we predict the targeted receiver plus ~1/3 of the coverage defenders (≈3 players; 69% of rows are defenders).
- The targeted receiver closes from 5.3 yd to 1.2 yd (median) from the ball landing spot during the flight.
- Constant velocity is accurate for ~5 frames; afterwards defenders deviate more than receivers (they react to the throw).
- Passes longer than 2 s are ~20% of the rows but ~70% of the squared error of the best baseline.

## Progress log
| Date | Work |
|---|---|
| 2026-11-09 | Repository structure, reusable `src/` modules, EDA and 8 baselines (Assignment 1) |
| 2026-11-10 | Group details; notebook executed in Google Colab (results updated: GBDT 0.732, Ridge 0.907) |
| 2026-11-10 | Kaggle notebook fixed for the new data mount path; first late submission (private RMSE 0.680) |
| 2026-11-12 | Moved to the `ceng313-2026` organization; notebook reorganized into the 7 required sections (duplicates, outliers, target distribution, preprocessing rationale, Kaggle submission); score screenshot in `submissions/` |

## Kaggle submission history
| Date | Notebook / model | Public score | Private score | Note |
|---|---|---|---|---|
| 2026-11-10 | `kaggle/submission_baseline.ipynb` v3 – GBDT residual baseline (B7) | – | 0.68028 | Late submission (not ranked); CV RMSE 0.732; [screenshot](submissions/kaggle_late_submission_v3.png) |

Kaggle displays only a private score for late submissions to this (closed) competition, so the public score is left empty.

## Academic integrity
All code in this repository was written for this project; no Kaggle notebooks or notebook cells were copied.

## Acknowledgements
Data: NFL Next Gen Stats via Kaggle – Lopez, Bliss, Blake, Yan, Plomecka, Howard (2025),
*NFL Big Data Bowl 2026 – Prediction*.
