---
name: keep-it-simple
description: Checklist before closing any change or milestone in this repo. Use after editing code or docs and before every commit.
---
1. Re-read the whole diff (`git diff`); delete what is not needed now (dead code, unused options, "just in case").
2. Files < 150 lines, small flat functions, no inheritance/decorators/registries, no library without a one-line reason.
3. New parameters live in config.py with a plain-words comment; script headers still say what they read and write.
4. Leakage: everything fitted (scaler, imputer, MiniRocket, tuning, early stopping) sees training patients only.
5. A biologist can follow it, `python -m pytest tests` passes, docs/ updated, open doubts in decisions_proposed.md.
