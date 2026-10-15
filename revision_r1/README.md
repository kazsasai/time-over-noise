# Analyses, figures and tables of the revised paper

This directory contains the code that turns the simulation data into every figure, supplementary table and
quoted number of the revised paper. The models themselves are in `../gcg_solver.py`; the core data are generated
by the runners in `../experiments/`.

```
revision_r1/
  run_all.sh            every step below, in dependency order
  data/                 core data (N = 20, k = 3): run_rev1_core.sh writes the .pkl files here
  analyses/             analyses; results go to analyses/runs/
  figures_final/        main-text Figs 1-6
  figures_final_supp/   Supplementary Figs S1-S12
  tables/               Supplementary Tables S1-S11 (LaTeX), generated from analyses/runs/
```

## Inputs

| File | Content | How to obtain |
|---|---|---|
| `data/result_for_criticality_20_3.pkl` | long records: N = 20, k = 3, ε, p = 0.05-0.95 (step 0.1), 1,000 trials of 50,000 rounds, with histories | `data/run_rev1_core.sh` or Zenodo |
| `data/result_for_phase_space_20_3.pkl` | parameter scans: N = 20, k = 3, ε, p = 0.00-0.99 (step 0.01), 1,000 trials of 1,000 rounds, with histories | `data/run_rev1_core.sh` or Zenodo |
| `../result_for_plateau_k34.pkl` | Fig. 6a-d: N = 20-100, k = 3, 4, 200 trials of 10,000 rounds | `../experiments/run_plateau_k34.py` or Zenodo |
| `../result_for_scalability.pkl` | Fig. 6e, f: N = 20-500, k = 2-50, 200 trials of 5,000 rounds | `../experiments/run_scalability.py` or Zenodo |

The core data are bit-reproducible: `data/run_rev1_core.sh` regenerates them from the seeds in the runners
(about 36 minutes and up to 12 GB of memory on an Apple M4 Pro with 14 workers). See `data/README.md`.

The analysis results (`analyses/runs/`, about 260 MB) are also deposited on Zenodo. With them in place the
figures and tables can be rebuilt without running the analyses.

## Running

```bash
cd revision_r1
bash run_all.sh analyses   # analyses/runs/ from the data (the longest stage)
bash run_all.sh figures    # figures_final*/ from analyses/runs/
bash run_all.sh tables     # tables/ from analyses/runs/
```

Each analysis script can also be run on its own from `revision_r1/analyses`; its docstring states its inputs and
outputs. `run_all.sh` lists the exact arguments used for the paper. Bootstrap replicates draw their random numbers
from fixed seeds per block of replicates, so the results do not depend on the number of worker processes.

The item-by-item map from each figure and table to its script, the analysis outputs it reads and the analysis that
produces them is in [`../docs/reproduction.md`](../docs/reproduction.md).

## Environment

Python 3.13 with NumPy 2.3, SciPy 1.17, pandas 2.3, NetworkX 3.6, Numba 0.63, Matplotlib 3.10 and seaborn 0.13
(`../requirements.txt` gives the minimum versions). Fig. 1 uses Arial when it is installed.
