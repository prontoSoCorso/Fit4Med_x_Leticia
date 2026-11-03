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
- 2026-11-03 MOIRAI: first deferred (its library cannot install on Python 3.13); the venv then moved to Python 3.11, so MOIRAI is back. Its library (uni2ts 2.0) pins numpy 1.26, scipy 1.11 and torch 2.4, so the whole venv uses that pip-resolved set.
