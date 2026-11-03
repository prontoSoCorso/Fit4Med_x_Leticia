# Decisions taken
Short dated entries (1-3 lines), newest last.

- 2026-11-03 Target: modified Barthel Index (mBI, 0-100) only.
- 2026-11-03 Scope now: IMU at t0 → mBI at t0, regression. Later, not now: mBI at t1 from IMU t0 (with or without mBI t0), other targets, fine-tuning of foundation models, external validation.
- 2026-11-03 Language: chat in Italian; everything in the repo in English.
- 2026-11-03 Code: numbered scripts plus small modules; no notebooks.
- 2026-11-03 Demo with a FAKE cohort built from one real .mvnx; its numbers have no scientific value.
- 2026-11-03 Validation: leave-one-patient-out outer loop; tuning on training patients only.
- 2026-11-03 Git: Claude commits (one per milestone, on branch `example-pipeline`) but never pushes; a PreToolUse hook blocks `git push`.
- 2026-11-03 Anonymization is not handled in this phase.
- 2026-11-03 Fake cohort written directly as extracted tables (no fake .mvnx files): the simplest way.
- 2026-11-03 Extraction: subject/test/timepoint/take come from the header's `originalFilename` (disk name as fallback); `-004` is the Xsens take number. Columns are `<group>_<name>_<axis>`; centerOfMass = position, velocity, acceleration (checked on the example).
- 2026-11-03 Features: 145 per window in 4 families (time, frequency, movement quality, bilateral + foot contacts). Flexion/extension = z axis of Xsens joint angles (largest range when walking, checked on the example).
- 2026-11-03 Evaluation: every window gets its patient's label; patient prediction = mean of its windows; tuning by GroupKFold(5) on training patients, scored by MAE; metrics with 2000 bootstrap resamplings of patients. Under LOPO the mean baseline has Spearman = -1 by construction (leaving out a high patient lowers the training mean).
- 2026-11-03 MiniRocket is fitted once per outer fold on the training windows, then Ridge's alpha is tuned on its features. The inner validation patients take part in that label-free fit: no effect on the test patient, much faster.
- 2026-11-03 MOIRAI: first deferred (its library cannot install on Python 3.13); the venv then moved to Python 3.11, so MOIRAI is back. Its library (uni2ts 2.0) pins numpy 1.26, scipy 1.11 and torch 2.4, so the whole venv uses that pip-resolved set.
- 2026-11-03 Because of torch 2.4: transformers pinned to 4.57.6 (version 5 breaks with torch 2.4) and pyarrow to 21.0.0 (newer needs numpy 2 and breaks scikit-learn). Found by running, not by pip check.
- 2026-11-03 PatchTST (transformers) and UniTS are trained from scratch, no pretraining. UniTS = original model file (commit 0e02814, MIT) used through its 'classification' path with one output and squared error; no fallback was needed.
- 2026-11-03 Foundation models stay frozen: embeddings computed once for all windows (step 4), Ridge trained per fold. Chronos-2 and MOMENT use their official embed(); MOIRAI has none, so its encoder output is recorded with a hook (unofficial, documented in models/moirai.py).
- 2026-11-03 Ridge alphas 1 ... 10,000 (smaller ones gave ill-conditioned-matrix warnings with 10,000+ columns); MiniRocket re-run with the same grid.
- 2026-11-03 Report: one table sorted by MAE and one dot-and-interval figure; FAKE data are flagged in the figure title and in the table's `data` column. The switch to real data was tried with the example .mvnx (steps 1-3; step 3 stops, as it should, without labels).
- 2026-11-03 Neural networks: fixed settings (config.DEEP), early stopping on ~20% of the training patients, channels and label standardised with training numbers only; LSTM reads windows at 30 Hz.
