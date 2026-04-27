# Time over noise — Asynchrony unlocks robust coordination beyond behavioural fluctuations

Reference implementation and reproduction scripts for the paper

> **Sasai, K., Soga, Y. & Gunji, Y.-P.** *Time over noise: Asynchrony unlocks robust coordination beyond behavioural fluctuations.* Nature Communications (2026).

The code implements two decentralised update regimes on the distributed
graph-colouring problem (GCG):

- **SNM** — Synchronous Noise Model. All agents update in lock-step; symmetry
  is broken by a tunable behavioural noise level ε applied to "bot" nodes.
- **ARM** — Asynchronous Random Model. Updates are state-coupled and
  asynchronous; agents whose local view is gridlocked re-select a colour
  uniformly at random. No parameter tuning.

The two models differ in a single decision rule and, through that difference,
produce qualitatively different dynamics: ARM exhibits robust self-organised
criticality with universal large-scale exponent τ̄ ≈ 1.38, whereas SNM
achieves criticality only at narrowly tuned ε and loses it under any
topological change.

---

## Quick start

```bash
# Clone
git clone https://github.com/kazsasai/time-over-noise.git
cd time-over-noise

# Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# (optional) GPU acceleration — Apple MPS or NVIDIA CUDA
pip install -r requirements-gpu.txt

# Smoke test: verify the solver imports
python -c "import gcg_solver; print('OK')"
```

Simulation results (`.pkl` files) are not bundled in this repository. They
are deposited on Zenodo:

> Zenodo DOI: [10.5281/zenodo.19812421](https://doi.org/10.5281/zenodo.19812421)

Download the archive into the repository root and unpack, or re-run the
runners in `experiments/` to regenerate them (see below).

---

## Repository layout

```
gcg_solver.py        # core solver (SNM + ARM) — numpy / numba
gcg_solver_gpu.py    # optional PyTorch MPS / CUDA batched implementation
experiments/         # scripts that generate result_*.pkl for each figure
figures/             # scripts that render each figure from result_*.pkl
diagnostics/         # supplementary diagnostic scripts (not required for reproduction)
docs/
  reproduction.md    # full figure → runner → data-file mapping
```

All scripts expect to be invoked from the repository root, e.g.
`python experiments/run_plateau_k34.py`. Generated `.pkl` files are written
to the repository root; figure scripts read them from there.

---

## Reproducing the figures

From the repository root, with the `.pkl` files available (either downloaded
from Zenodo or regenerated locally):

| Figure            | Data file                             | Runner                                    | Figure script                 |
| ----------------- | ------------------------------------- | ----------------------------------------- | ----------------------------- |
| Fig. 1 (concept)  | — (hand-drawn schematic)              | —                                         | —                             |
| Fig. 2 (timeseries) | — (generated in-script)              | —                                         | `figures/fig2_timeseries.py`  |
| Fig. 3 (universality) | `result_for_phase_space_20_3.pkl` | `experiments/run_phase_space_20_3.py` | `figures/fig3_universality.py` |
| Fig. 4 (criticality) | `result_for_criticality_20_3.pkl`  | `experiments/run_criticality_20_3.py` | `figures/fig4_criticality.py` |
| Fig. 5 (scalability) | `result_for_scalability.pkl` + `result_for_plateau_k34.pkl` | `experiments/run_scalability.py` + `experiments/run_plateau_k34.py` | `figures/fig5_scalability.py` |
| Supp. CCDF        | `result_for_criticality_20_3.pkl`     | (same as Fig. 4)                          | `figures/fig_supp_ccdf.py`    |
| Supp. networks    | (generated on the fly)                | —                                         | `figures/fig_supp_networks.py` |
| Supp. hybrid      | `result_static_random_20_3.pkl`       | `experiments/run_static_random_20_3.py`   | `figures/fig_supp_hybrid.py`  |

Example — reproduce Fig. 4 end-to-end:

```bash
# 1. regenerate the data  (~a few hours on a modern CPU)
python experiments/run_criticality_20_3.py

# 2. render the figure
python figures/fig4_criticality.py
```

The full table, with trial counts, step budgets, and expected wall-time, is
in [`docs/reproduction.md`](docs/reproduction.md).

---

## Solver API

```python
import gcg_solver

# Run a single simulation
result = gcg_solver.run_simulation(
    node_num       = 20,
    color_num      = 3,
    step_max       = 50_000,
    p_async        = 0.30,      # ARM: share of nodes held per step
    p_noise        = 0.05,      # SNM: ε
    is_async_model = True,      # True → ARM, False → SNM
    seed           = 0,
)
```

The GPU path has the same call signature:

```python
import gcg_solver_gpu
df = gcg_solver_gpu.run_hybrid_simulation_batched(task_list, gpu_n_max=250)
```

GPU acceleration gives ≈ 20× speed-up for N ≤ 100 on an Apple M4 Pro and is
used primarily for the large-scale scalability sweeps.

---

## Data availability

All simulation outputs (`result_*.pkl`) and raw seed logs are archived at

> Zenodo: [10.5281/zenodo.19812421](https://doi.org/10.5281/zenodo.19812421)

Raw `.pkl` files are **not** tracked in Git because they exceed GitHub's
practical size limits. Regeneration scripts are provided for every figure, so
the code base is self-contained even without the archive.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{sasai2026timeovernoise,
  title   = {Time over noise: Asynchrony unlocks robust coordination beyond behavioural fluctuations},
  author  = {Sasai, Kazuto and Soga, Yuto and Gunji, Yukio-Pegio},
  journal = {Nature Communications},
  year    = {2026},
  doi     = {10.xxxx/xxxxx}
}
```

A machine-readable `CITATION.cff` is provided at the repository root so that
GitHub's "Cite this repository" widget renders the correct entry.

---

## License

Released under the [MIT License](LICENSE).
