# Open questions
Still open, also future or organizational. Short dated entries.

- 2026-11-03 How many patients and recordings? With 19 FAKE patients the 95% intervals are about ±1.5-2 mBI points around an MAE of 6, so smaller differences between models cannot be told apart. Which ID links IMU and labels (the .mvnx header ID differs from the file name)? One ID per patient (two IDs for one patient = leakage)?
- 2026-11-03 Speed protocol: which take is max and which normal, always in the same order?
- 2026-11-03 mBI version and assessment date vs IMU; which metadata exist and are known at prediction time (slide 11: proxies of the target)?
- 2026-11-03 Leticia's laptop: OS, RAM, GPU? It needs Python 3.11 (pinned set); the hook calls `python3`. Data access and anonymization for a visiting student.
- 2026-11-03 Later: mBI at t1 (with or without mBI at t0), other targets, fine-tuning, external validation on Leticia's datasets.
