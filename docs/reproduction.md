# Reproduction guide

This document maps every figure in the main text and supplementary material
to the runner that generates the underlying dataset and the figure-rendering
script that consumes it.

All paths are relative to the repository root. Data files (`result_*.pkl`)
are not in Git; download them from the Zenodo archive or regenerate by
running the corresponding runner.

All scripts expect to be invoked from the repository root, e.g.
`python experiments/run_plateau_k34.py`.

---

## Main text

| Figure | Runner (experiments/) | Output pkl | Figure script (figures/) |
| --- | --- | --- | --- |
| **Fig. 1** Concept schematic | — | — | — (hand-drawn in Keynote; final asset: `tex/diagram.pdf`) |
| **Fig. 2** Time-series evolution | generated in-script | — | `fig2_timeseries.py` |
| **Fig. 3** Universality | `run_phase_space_20_3.py` | `result_for_phase_space_20_3.pkl` | `fig3_universality.py` |
| **Fig. 4** SOC signatures (FFT, avalanches) | `run_criticality_20_3.py` | `result_for_criticality_20_3.pkl` | `fig4_criticality.py` |
| **Fig. 5** Scalability / topology | `run_scalability.py` + `run_plateau_k34.py` | `result_for_scalability.pkl`, `result_for_plateau_k34.pkl` | `fig5_scalability.py` |

## Supplementary material

| Figure | Runner | Output pkl | Figure script |
| --- | --- | --- | --- |
| Supp. avalanche CCDF | (reuses Fig. 4 data) | `result_for_criticality_20_3.pkl` | `fig_supp_ccdf.py` |
| Supp. network topology (ER, WS, BA) | generated in-script | — | `fig_supp_networks.py` |
| Supp. hybrid human–agent | `run_static_random_20_3.py` | `result_static_random_20_3.pkl` | `fig_supp_hybrid.py` |
| Supp. 2×2 conditions | (various) | multiple | `fig_2x2_conditions.py`, `fig_2x2_bar.py` |

## Auxiliary runners

These runners produce datasets that are reused across multiple figures or
appear in supplementary tables:

| Runner | Output | Notes |
| --- | --- | --- |
| `run_criticality_100_4.py`       | `result_for_criticality_100_4.pkl`     | k=4, N=100 |
| `run_criticality_100_15.py`      | `result_for_criticality_100_15.pkl`    | k=15, N=100 |
| `run_criticality_500_3.py`       | `result_for_criticality_500_3.pkl`     | k=3, N=500 |
| `run_criticality_500_20.py`      | `result_for_criticality_500_20.pkl`    | k=20, N=500 |
| `run_criticality_500_32.py`      | `result_for_criticality_500_32.pkl`    | k=32, N=500 |
| `run_criticality_N40_k3.py`      | `result_criticality_N40_k3.pkl`        | k=3, N=40 |
| `run_criticality_multiN_k3.py`   | `result_criticality_multiN_k3.pkl`     | k=3, N∈{20,50,100} |
| `run_criticality_multiN_k3_long.py` | `result_criticality_multiN_k3_long.pkl` | long-run variant |
| `run_plateau.py`                 | `result_for_plateau.pkl`               | coarse plateau sweep |
| `run_size_color.py`              | `result_for_size_color.pkl`            | coarse N×k grid |
| `run_size_color_1k.py`           | `result_for_size_color_1k.pkl`         | cluster-scale N×k grid |

---

## Wall-time estimates

Approximate wall-time on a 2024-class laptop (Apple M4 Pro, 14 cores, CPU
path only). GPU path typically gives a 5–20× speed-up for N ≤ 250.

| Runner | Parameters | CPU time |
| --- | --- | --- |
| `run_criticality_20_3.py`        | N=20, k=3, 500 trials × 100k steps, 11 p values | ~6 h |
| `run_plateau_k34.py`             | N=20..100, k=3,4, 200 trials × 10k steps        | ~4 h |
| `run_scalability.py`             | large N×k sweep                                  | ~8 h |
| `run_phase_space_20_3.py`        | N=20, k=3, 1k trials × 1k steps, 100 p values   | ~2 h |
| `run_size_color_1k.py`           | N=20..1000, k=2..100                             | cluster-scale |

---

## Seeding

All runners fix a `MASTER_SEED` at the top of the file; per-trial seeds are
derived deterministically from it. Running the same runner twice on the same
machine with the same library versions should produce bit-identical output.

If you see non-trivial numerical differences, check your NumPy and Numba
versions against `requirements.txt` — Numba's `@njit(cache=True)` can mask
behavioural changes across minor versions.
