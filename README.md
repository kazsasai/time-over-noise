# Time over noise: Coordination from the unresolved inconsistencies of individual timing

Code for the paper

> **Sasai, K., Soga, Y. & Gunji, Y.-P.** *Time over noise: Coordination from the unresolved inconsistencies of
> individual timing.* (2026, in revision.)

Individuals in human groups and animal collectives perceive only their surroundings and act on their own time.
The paper asks whether the variability that coordination needs can come from the individuality of time itself,
through the inconsistencies it produces, rather than from behavioural noise tuned from outside. It uses the
distributed graph-colouring problem and two decentralised update models:

- **SNM** (synchronous noise model). All agents act at every round on a common clock. With probability 1 − ε an
  agent picks a conflict-free colour (and keeps its colour when gridlocked); with probability ε it picks any colour
  at random.
- **ARM** (asynchronous random model). Each agent acts in a round with probability 1 − p, on its own time. An
  acting agent picks a conflict-free colour, and redraws its colour at random only when it is gridlocked.

---

## Repository layout

```
gcg_solver.py         solver of both models (NumPy / Numba)
gcg_solver_gpu.py     optional batched implementation (PyTorch, Apple MPS or NVIDIA CUDA)
experiments/          runners that generate the simulation datasets (result_*.pkl)
revision_r1/          analyses, figures and supplementary tables of the paper
  run_all.sh            every step, in dependency order
  data/                 core data (N = 20, k = 3) and the script that generates them
  analyses/             analyses; results go to analyses/runs/
  figures_final/        Figs 1-6
  figures_final_supp/   Supplementary Figs S1-S12
  tables/               Supplementary Tables S1-S11
docs/reproduction.md  figure / table -> script -> analysis outputs -> analysis -> data
figures/, diagnostics/  figure and diagnostic scripts of the first version of the manuscript (release v1.0)
```

## Quick start

```bash
git clone https://github.com/kazsasai/time-over-noise.git
cd time-over-noise
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # or requirements-lock.txt: the exact versions we used

python -c "import gcg_solver; print('OK')"
```

## Reproducing the paper

1. **Data.** Download the datasets from Zenodo (below) or regenerate them:
   - core data: `bash revision_r1/data/run_rev1_core.sh` (about 36 minutes on a 14-core machine);
   - parameter scans of Fig. 6: `python experiments/run_plateau_k34.py` and `python experiments/run_scalability.py`,
     run from the repository root.
2. **Analyses, figures, tables.** `bash revision_r1/run_all.sh` (or the stages `analyses`, `figures`, `tables`).
   The analysis results are also on Zenodo; with them, the figures and tables are rebuilt without the analyses.

[`docs/reproduction.md`](docs/reproduction.md) maps each figure and table to its scripts and inputs, and
[`revision_r1/README.md`](revision_r1/README.md) describes the analysis code.

### Reproducibility

Every trial is determined by its seed. On the platform we used (macOS on Apple silicon) with the versions in
`requirements-lock.txt`, regenerating the core data reproduces every trial of the deposited files bit for bit; only
the order of the rows differs, because the batched solver collects trials as they finish. Run on the deposited
files, the analyses reproduce the deposited results. Bootstrap intervals depend on the row order, so on regenerated
files they can differ slightly (see `revision_r1/data/README.md`).

On other platforms or with other library versions, SNM runs can differ trial by trial. The SNM assigns its noise
draws to the agents in the order given by `np.argsort` of their degrees, and the order of agents with equal degree
can depend on the NumPy build. ARM runs do not depend on this order. The parameter scans of Fig. 6 were generated
earlier. Re-running a sample of their trials on our platform reproduced the trials of Fig. 6a-d and the ARM trials
of Fig. 6e, f bit for bit, and about 60% of the SNM trials of Fig. 6e, f, with the same solution rates.

## Solver

```python
import gcg_solver

# one trial, as in the core data
res = gcg_solver._worker_run_single(dict(
    seed=0, node_num=20, color_num=3, ba_m=2, bot_num=20, bot_type='c',
    step_max=1000, return_history=True,
    label='Async Model', is_async_model=True,   # ARM; is_async_model=False gives the SNM
    p_async=0.45,                               # ARM: probability that an agent does not act in a round
    p_noise=0.0,                                # SNM: epsilon
))
print(res['solved'], res['solve_time'])
```

Many trials are run in parallel with `gcg_solver.run_parallel_simulation_batched(task_list)`; the runners in
`experiments/` show the task lists used for the paper.

## Data availability

The simulation datasets and the analysis results are archived on Zenodo:
[10.5281/zenodo.19812420](https://doi.org/10.5281/zenodo.19812420) (all versions).
`.pkl` files are not tracked in Git.

## Versions

- **v2.1**: code of the paper in its present form (Zenodo [10.5281/zenodo.23392863](https://doi.org/10.5281/zenodo.23392863)).
  Corrections after a final check: the name of a copyright holder in `LICENSE` (Yuto Soga); times to the first
  solution in Supplementary Fig. S11 counted from 1, as in Fig. 3; table notes and figure labels aligned with the text;
  the Mann-Whitney test of Fig. 3 over all trials of the 50,000-round records. Added: the paired comparison of the
  times to the first solution (trials with the same seed share the graph and the initial colouring;
  `run_first_passage.py`, `run_mannwhitney.py`); cascade probabilities of injected inconsistencies with 95% intervals
  from bootstrap over graphs (Supplementary Fig. S7j and Table S11; `run_graph_types.py`); the avalanche classes of
  Fig. 5a, b named by the state at the bin ends.
- **v2.0**: the analyses, figures and tables in `revision_r1/`; core data regenerated with the seeded solver.
- **v1.0**: code of the first version of the manuscript
  ([10.5281/zenodo.19812421](https://doi.org/10.5281/zenodo.19812421)).

## Citation

Please cite the paper and the Zenodo archive. `CITATION.cff` provides the entry used by GitHub's
"Cite this repository".

## License

[MIT License](LICENSE).
