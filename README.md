# Fit4Med × Leticia: from Xsens IMU recordings to the modified Barthel Index

Example pipeline for the Fit4Med study (post-stroke patients). It reads Xsens `.mvnx` recordings
and predicts the modified Barthel Index (mBI, 0-100) with classical, deep and foundation models.
Until the real data arrive it runs on a **FAKE** cohort: its numbers have no scientific value.

## Setup (once)
A virtual environment is a private copy of Python with its own libraries, so this project cannot break others.
On the lab workstation it already exists (Python 3.11), outside this folder:

    source /home/phd2/Scrivania/CorsoVenvs/GeneralVenv/bin/activate
    pip install -r requirements.txt

On another computer: create a Python 3.11 environment (`python3.11 -m venv myvenv`) and install the same file.
Neural networks and foundation models need the heavy libraries too (several GB, see the file for computers without a GPU):

    pip install -r requirements.txt -r requirements-deep.txt

## Run
Numbered scripts, run from this folder, one at a time; each says at the top what it reads and writes.

| Step | Command | What it does |
|---|---|---|
| 0 | `python 00_make_fake_cohort.py` | demo only, instead of step 1: 20 FAKE patients made from the example recording |
| 1 | `python 01_extract.py` | every `.mvnx` in `data/raw/` → one table per recording in `data/extracted/` |
| 2 | `python 02_inventory.py` | what we have, what is missing or duplicated → `results/inventory.csv` |
| 3 | `python 03_make_windows.py` | cut the still parts, 4-second windows + hand-made features → `data/windows/` |
| 4 | `python 04_embed_foundation.py [model ...]` | frozen foundation models describe each window → `data/embeddings/` |
| 5 | `python 05_evaluate.py [model ...]` | leave-one-patient-out for each model → `results/predictions/` |
| 6 | `python 06_report.py` | one table and one figure → `results/summary_table.csv`, `results/comparison.png` |

Demo: 0, 2, 3, 4, 5, 6. Real data: 1 to 6. Steps 4-5 for the neural networks and foundation models
need `requirements-deep.txt`. Leave-one-patient-out: each patient in turn is predicted by a model
trained on all the other patients, so no patient ever helps predict itself.

### Reading the results
MAE is the mean error in mBI points (lower is better); the line around each dot is its 95% interval
over patients. The mean-predictor baseline is the bar to beat. On the FAKE data the mBI was made as
100 × walking speed + noise, so the models reach an MAE of about 5-7 (the noise alone costs about 4)
against about 18 for the baseline: this only shows that the pipeline works.

### From the FAKE demo to the real data
1. Delete the folder `data/extracted/` and the file `data/labels.csv` (they are the FAKE demo).
2. Copy the `.mvnx` files into `data/raw/` and the real `labels.csv` (columns `subject, timepoint, mbi`,
   optionally `age, bmi, hemisphere`) into `data/`.
3. Run the steps again from 1.

## Folders
| Folder / file | What it holds |
|---|---|
| `config.py` | all parameters and paths |
| `pipeline/` | building blocks used by the numbered scripts |
| `models/` | one file per model |
| `tests/` | a few automatic checks: `python -m pytest tests` |
| `docs/` | decisions taken, decisions proposed, open questions |
| `CLAUDE.md`, `.claude/` | rules for Claude Code (a hook blocks `git push`) |
| `data/`, `results/`, `examples/` | data and outputs, never in git |

## If I want to change X, I open Y
| To change | Open |
|---|---|
| paths, timepoints, target, channels, window length, "standing still" threshold, max/normal speed (`SPEED_BY_TAKE`), models that run | `config.py` |
| one model (its network or its settings) · add a model | `models/<name>.py` · a new file in `models/` + its name in `MODELS_ENABLED` |
| reading the .mvnx · cleaning and windows · features · validation · scores | `pipeline/` (`mvnx`, `preprocessing`, `features`, `evaluation`, `metrics`) |
| the FAKE cohort · the final table and figure | `00_make_fake_cohort.py` · `06_report.py` |
