# Working rules for this repo
Audience: Leticia, PhD student with a biology background. Explain as to a biologist: plain words,
each ML concept in one sentence the first time it appears. Chat in Italian; repo in English.

## Keep it simple
- Small flat functions; files < 150 lines (models/third_party/ is copied code: never edit it).
- No inheritance, decorators or "magic" registries; a class only when a function is not enough.
- Common libraries only, each extra justified in one line; heavy ones in requirements-deep.txt.
- One config file (config.py) for parameters and paths; no config or CLI frameworks.
- Descriptive names; comments say WHY in simple words; every script opens with 3-5 lines on what it reads and writes.
- Numbered steps, each runnable alone; errors say what to do next; at most 5 tests.
- Ask before adding complexity (library, folder, abstraction, option).

## Never leak between patients
- One patient's windows, repetitions and speeds stay in the same fold; scaling, imputation,
  feature selection, tuning and early stopping see training patients only.

## Workflow
- In doubt, do not decide: write it in docs/decisions_proposed.md and ask.
- End of every milestone: /keep-it-simple, re-read the diff, remove the superfluous, update docs/
  (dated entries, 1-3 lines), one commit with a short message. Never push (a hook blocks it).
