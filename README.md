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

## Run
Numbered scripts, run from this folder, one at a time; each says at the top what it reads and writes.

| Step | Command | What it does |
|---|---|---|
| 0 | `python 00_make_fake_cohort.py` | demo only, instead of step 1: 20 FAKE patients made from the example recording |
| 1 | `python 01_extract.py` | every `.mvnx` in `data/raw/` → one table per recording in `data/extracted/` |
| 2 | `python 02_inventory.py` | what we have, what is missing or duplicated → `results/inventory.csv` |
| 3 | `python 03_make_windows.py` | cut the still parts, 4-second windows + hand-made features → `data/windows/` |
| 5 | `python 05_evaluate.py [model ...]` | leave-one-patient-out for each model → `results/predictions/` |

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
