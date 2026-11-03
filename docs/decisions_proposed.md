# Decisions proposed (to take in this round)
Options and a recommendation; the code uses the recommendation until someone decides otherwise.

- 2026-11-03 Repetitions/speeds: (a) pool all t0 recordings of a patient, score = mean of its windows; (b) normal speed only; (c) one model per speed. Rec.: (a); switch via `SPEEDS_USED` after the inventory.
- 2026-11-03 Max/normal label (absent from the file): (a) take → speed map in `SPEED_BY_TAKE`; (b) lab sheet; (c) faster take = max. Rec.: (a) once the protocol is confirmed; "unknown" until then.
- 2026-11-03 Default channels (slide 4 risks): L/R hip, knee, ankle joint angles × 3 axes + pelvis vertical acceleration (19). Out: position/velocity (drift), orientation (yaw), angular acceleration (noise), raw sensors (gravity, magnetometer). Rec.: as stated.
- 2026-11-03 Cleaning/windows: walking = pelvis horizontal speed > 0.15 m/s (0.5 s mean); 4 s windows, 50% overlap (~9 per trial). Rec.: as stated; revisit after the inventory.
- 2026-11-03 Bilateral features: side-agnostic |L−R|/(L+R) now; affected vs unaffected side once hemisphere is known for all. Rec.: side-agnostic.
- 2026-11-03 Foundation models: chronos-2 (120M parameters), MOMENT-1-large (346M, installed from GitHub), moirai-2.0-R-small (11M; CC-BY-NC license, research only; no official embedding function). Embedding = mean over time, concatenated over channels. Laptop: chronos-2-small, MOMENT-1-small. Rec.: as stated.
- 2026-11-03 Late fusion: (a) concatenate IMU features + metadata → Ridge (slide 13); (b) stacking of predictions. Always report metadata-only, IMU-only (same Ridge), fused. Rec.: (a).
- 2026-11-03 Storage: one CSV per recording (~12 MB, opens in Excel) vs Parquet (~10× smaller, needs pyarrow). Rec.: CSV.
