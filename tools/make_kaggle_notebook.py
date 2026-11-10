"""Build kaggle/submission_baseline.ipynb: a self-contained Kaggle notebook (internet off)
that embeds src/ and submits the best Assignment 1 baseline through the evaluation API."""
import json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lines(s):
    l = s.split("\n")
    return [x + "\n" for x in l[:-1]] + [l[-1]]


cells = [
    ("markdown", "# Kaggle submission – Assignment 1 baseline (GBDT residual)\n"
                 "Self-contained version of the repository's `src/` package for the Kaggle code-competition "
                 "environment (internet off). Add the competition data, then *Save Version → Save & Run All* and submit."),
]
src_cell = "import os\nos.makedirs('src', exist_ok=True)\nFILES = {}\n"
for name in ["__init__.py", "data.py", "models.py", "utils.py"]:
    code = open(os.path.join(ROOT, "src", name), encoding="utf-8").read()
    src_cell += f"FILES[{name!r}] = {code!r}\n"
src_cell += "for n, c in FILES.items():\n    open(os.path.join('src', n), 'w').write(c)\nprint('src/ written')"
cells.append(("code", src_cell))
cells.append(("code", """import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from src import data as D, models as M

DATA_DIR = D.find_data_dir()          # finds the mounted competition data
print('data:', DATA_DIR)
sys.path.insert(0, DATA_DIR)           # kaggle_evaluation package lives next to the data
inp, out = D.load_train(DATA_DIR)
rows = D.build_training_frame(inp, out)
model = M.GBDTResidual().fit(rows)
del inp, out
print('trained on', len(rows), 'rows')"""))
cells.append(("code", """import polars as pl

def predict(test, test_input):
    t = test.to_pandas() if hasattr(test, 'to_pandas') else test
    ti = test_input.to_pandas() if hasattr(test_input, 'to_pandas') else test_input
    for c in ['play_direction', 'player_position', 'player_side', 'player_role']:
        ti[c] = ti[c].astype('category')
    rows_t = D.expand_to_frames(D.build_player_table(ti), frames=t)
    x, y = model.predict(rows_t)
    x, y = D.flip_xy(x, y, rows_t['is_left'].to_numpy())
    assert len(x) == len(t)
    return pl.DataFrame({'x': x, 'y': y})

import kaggle_evaluation.nfl_inference_server
server = kaggle_evaluation.nfl_inference_server.NFLInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    server.serve()
else:
    server.run_local_gateway((DATA_DIR,))"""))

nb = {"cells": [{"cell_type": t, "metadata": {}, "source": lines(s), **({"outputs": [], "execution_count": None} if t == "code" else {})}
                for t, s in cells],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.join(ROOT, "kaggle"), exist_ok=True)
json.dump(nb, open(os.path.join(ROOT, "kaggle", "submission_baseline.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote kaggle/submission_baseline.ipynb")
