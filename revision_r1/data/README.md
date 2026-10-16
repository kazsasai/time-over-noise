# Core data (N = 20, k = 3)

`run_rev1_core.sh` writes the two core datasets into this directory with the seeded solver (`../../gcg_solver.py`)
and the runners in `../../experiments/`. The same files are deposited on Zenodo (gzip-compressed).

| File | Runner | Design |
|---|---|---|
| `result_for_criticality_20_3.pkl` | `run_criticality_20_3.py` | Barabási–Albert graphs (m = 2), N = 20, k = 3; SNM at ε and ARM at p = 0.05, 0.15, …, 0.95; 1,000 trials per condition (master seed 2024); 50,000 rounds; per-round histories |
| `result_for_phase_space_20_3.pkl` | `run_phase_space_20_3.py` | as above, ε and p = 0.00, 0.01, …, 0.99; 1,000 trials per condition (master seed 2024); 1,000 rounds; per-round histories |

Each trial uses its own graph and initial coloring, both drawn from the trial seed; the same seed gives the same
graph in both datasets and in every analysis that re-simulates trials (`../analyses/gcg_kick.py`).

**Reproducibility.** Every trial is determined by its seed. Regenerating both datasets with `run_rev1_core.sh`
on the platform and with the versions in `../../requirements-lock.txt` reproduced every trial bit for bit: all 18
columns of all 20,000 and 200,000 trials agree with the deposited files (checked on 2026-10-15). The order of the
rows differs from run to run, because the batched solver collects the trials as they finish, so the regenerated
files are not byte-identical to the deposited ones. The analyses refer to trials by their row within each condition
(for example when trials are resampled for bootstrap intervals). To reproduce the deposited intervals exactly, run
the analyses on the deposited files. Generation took about 30 minutes (long records, peak memory 12.3 GB) and
6 minutes (parameter scans, 5.1 GB) on an Apple M4 Pro with 14 worker processes.

**Columns.** One row per trial: the parameters of the task (`seed`, `node_num`, `color_num`, `label`,
`is_async_model`, `p_noise`, `p_async`, …), the outcome (`solved`, `solve_time`, 0-based round of the first proper
coloring; `step_max` if none) and the per-round histories `history_changed` (agents that change color),
`history_conflict_ratio` (fraction of conflicting nodes) and `history_gridlock` (gridlocked agents). The long records
come from the batched path of the solver (int32 / float32 / int32), the parameter scans from the single-run path
(int32 / float64 / int64).
