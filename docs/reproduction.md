# Reproduction guide

This guide maps every figure and supplementary table of the paper to the script that draws it, the analysis
outputs it reads and the analysis scripts that produce them. In that table, scripts are given relative to
`revision_r1/` and analysis outputs relative to `revision_r1/analyses/runs/`.
`revision_r1/run_all.sh` runs every step in dependency order with the arguments used for the paper.

## Data

| Dataset | Runner (`experiments/`) | Read by (`revision_r1/analyses/`) |
|---|---|---|
| `revision_r1/data/result_for_criticality_20_3.pkl` (long records) | `run_criticality_20_3.py`, via `revision_r1/data/run_rev1_core.sh` | `avalanche_histograms.py`, `export_supp_data.py`, `pre_solution.py`, `run_branching.py`, `run_episodes.py`, `run_first_passage.py`, `run_graph_types.py`, `run_kick.py`, `run_spectra.py`, `save_trial_types.py`, `spectra_supplement.py` |
| `revision_r1/data/result_for_phase_space_20_3.pkl` (parameter scans) | `run_phase_space_20_3.py`, via `revision_r1/data/run_rev1_core.sh` | `run_first_passage.py`, `run_robustness.py` |
| `result_for_plateau_k34.pkl` (Fig. 6a-d) | `run_plateau_k34.py` | `fig6_numbers.py`, `run_fpt_size.py`, `run_supp_extra.py` |
| `result_for_scalability.pkl` (Fig. 6e, f) | `run_scalability.py` | `fig6_numbers.py` |

Some analyses run new simulations with the graphs and seeds of the core data: the control experiments
(`run_factorial_2x2.py`, `pilot_E1_E4.py`, with `gcg_variants.py`), the injected inconsistencies and colour histories
(`run_kick.py`, `run_kick_growth.py`, `run_graph_types.py`, `run_departure_mechanism.py`, `run_agent_spectra.py`, with
`gcg_kick.py`), the mixed models and network ensembles (`run_supp_extra.py`) and the step-budget extension
(`run_budget_extension.py`). `gcg_variants.py` and `gcg_kick.py` reproduce `gcg_solver.py` bit for bit where their
settings coincide.

## Figures and tables

| Item | Script | Analysis outputs read (`analyses/runs/`) | Produced by (`analyses/`) |
|---|---|---|---|
| Fig. 1 | `figures_final/fig1/plot_fig1.py` | — | — |
| Fig. 2 | `figures_final/fig2/plot_fig2.py` | simulated in the script with `gcg_solver.py` | — |
| Fig. 3 | `figures_final/plot_fig3.py` | `fpt/factorial_2x2.json`<br>`fpt/first_passage.json`<br>`fpt/mannwhitney.json`<br>`fpt/survival_at_optima.json` | `run_factorial_2x2.py`<br>`run_first_passage.py`<br>`run_mannwhitney.py` |
| Fig. 4 | `figures_final/plot_fig4.py` | `fpt/first_passage.json`<br>`robust/robustness.json` | `run_first_passage.py`<br>`run_robustness.py` |
| Fig. 5 | `figures_final/plot_fig5.py` | `aval/hist_ARM_*.npz`<br>`aval/planC_sweep/episodes.json`<br>`aval/planC_sweep/planC.json`<br>`kick/kick_growth.json`<br>`kick/kick_raw_*_*.npz` | `avalanche_histograms.py`<br>`run_episodes.py`<br>`run_kick.py`<br>`run_kick_growth.py`<br>`run_planC.py --tag sweep` |
| Fig. 6 | `figures_final/plot_fig6.py` | `fig6/fig6_grids.json` | `fig6_numbers.py` |
| Supplementary Fig. S1 | `figures_final_supp/plot_SF.py` | `fpt/first_passage.json` | `run_first_passage.py` |
| Supplementary Fig. S2 | `figures_final_supp/plot_SG.py` | `aval/hist_*_*.npz`<br>`fpt/solve_times_long.npz`<br>`pre/pre_solution.json` | `avalanche_histograms.py`<br>`export_supp_data.py`<br>`pre_solution.py` |
| Supplementary Fig. S3 | `figures_final_supp/plot_extra.py (hybrid)` | `supp/hybrid.json` | `run_supp_extra.py` |
| Supplementary Fig. S4 | `figures_final_supp/plot_SA.py` | `aval/hist_*_*.npz`<br>`aval/planC_sweep/planC.json` | `avalanche_histograms.py`<br>`run_planC.py --tag sweep` |
| Supplementary Fig. S5 | `figures_final_supp/plot_SB.py` | `aval/hist_*_*.npz`<br>`aval/hump_lowercut_sensitivity.json`<br>`aval/hump_same_region.json`<br>`aval/mle_sweep/point.json` | `avalanche_histograms.py`<br>`hump_same_region.py`<br>`run_aval_mle.py`<br>`run_hump_lowercut.py` |
| Supplementary Fig. S6 | `figures_final_supp/plot_SC.py` | `aval/hist_*_*.npz`<br>`aval/planC_*/planC.json` | `avalanche_histograms.py`<br>`run_planC.py (all tags)` |
| Supplementary Fig. S7 | `figures_final_supp/plot_SH.py` | `branching/branching.json`<br>`branching/branching_by_class.json`<br>`kick/kick_growth.json`<br>`kick/kick_raw_*_*.npz`<br>`kick/kick_raw_ARM_*.npz`<br>`kick/kick_vs_natural_large.json` | `run_branching.py`<br>`run_branching_by_class.py`<br>`run_kick.py`<br>`run_kick_growth.py`<br>`run_kick_vs_natural.py` |
| Supplementary Fig. S8 | `figures_final_supp/plot_SD.py` | `spec/spectra.json`<br>`spec/spectra_supplement.json`<br>`spec/spectrum_curves.npz` | `export_supp_data.py`<br>`run_spectra.py`<br>`spectra_supplement.py` |
| Supplementary Fig. S9 | `figures_final_supp/plot_SE.py` | `agentspec/agent_spectra.json`<br>`agentspec/agent_spectra_curves.npz` | `export_supp_data.py`<br>`run_agent_spectra.py` |
| Supplementary Fig. S10 | `figures_final_supp/plot_extra.py (plateau)` | `fig6/fig6_grids.json`<br>`supp/plateau.json` | `fig6_numbers.py`<br>`run_supp_extra.py` |
| Supplementary Fig. S11 | `figures_final_supp/plot_extra.py (fpt_size)` | `fpt_size/fpt_size.json` | `run_fpt_size.py` |
| Supplementary Fig. S12 | `figures_final_supp/plot_extra.py (networks)` | `supp/colourability.json`<br>`supp/networks.json` | `run_colourability.py`<br>`run_supp_extra.py` |
| Supplementary Table S1 | `tables/make_tables.py (passage_table)` | `fpt/factorial_2x2.json`<br>`fpt/first_passage.json`<br>`fpt/mannwhitney.json`<br>`robust/robustness.json` | `run_factorial_2x2.py`<br>`run_first_passage.py`<br>`run_mannwhitney.py`<br>`run_robustness.py` |
| Supplementary Table S2 | `tables/make_tables.py (controls_table)` | `fpt/first_passage.json`<br>`pilots/pilot_e1_snm_minconf.pkl`<br>`pilots/pilot_e2_sync_gated_q.pkl`<br>`pilots/pilot_factorial.pkl` | `pilot_E1_E4.py`<br>`run_first_passage.py` |
| Supplementary Table S3 | `tables/make_tables.py (population_table)` | `aval/planC_*/planC.json`<br>`aval/planC_sweep/episodes.json`<br>`aval/planC_sweep/planC.json`<br>`aval/planC_sweep/sstar_rule_sensitivity.json` | `run_episodes.py`<br>`run_planC.py (all tags)`<br>`run_planC.py --tag sweep`<br>`run_sstar_rule_sweep.py` |
| Supplementary Table S4 | `tables/make_tables.py (population_table)` | `aval/planC_*/planC.json`<br>`aval/planC_sweep/episodes.json`<br>`aval/planC_sweep/planC.json`<br>`aval/planC_sweep/sstar_rule_sensitivity.json` | `run_episodes.py`<br>`run_planC.py (all tags)`<br>`run_planC.py --tag sweep`<br>`run_sstar_rule_sweep.py` |
| Supplementary Table S5 | `tables/make_tables.py (mle_tables)` | `aval/hist_ARM_*.npz`<br>`aval/mle_sweep/boot.json`<br>`aval/mle_sweep/point.json`<br>`aval/planC_sweep/planC.json` | `avalanche_histograms.py`<br>`run_aval_mle.py`<br>`run_aval_mle_boot.py`<br>`run_planC.py --tag sweep` |
| Supplementary Table S6 | `tables/make_tables.py (mle_tables)` | `aval/hist_ARM_*.npz`<br>`aval/mle_sweep/boot.json`<br>`aval/mle_sweep/point.json`<br>`aval/planC_sweep/planC.json` | `avalanche_histograms.py`<br>`run_aval_mle.py`<br>`run_aval_mle_boot.py`<br>`run_planC.py --tag sweep` |
| Supplementary Table S7 | `tables/make_tables.py (knee_table)` | `aval/hump_same_region.json` | `hump_same_region.py` |
| Supplementary Table S8 | `tables/make_tables.py (spectra_table)` | `agentspec/agent_spectra.json`<br>`spec/spectra.json`<br>`spec/spectra_supplement.json` | `run_agent_spectra.py`<br>`run_spectra.py`<br>`spectra_supplement.py` |
| Supplementary Table S9 | `tables/make_tables.py (departure_table)` | `departure/departure_summary.json` | `run_departure_mechanism.py` |
| Supplementary Table S10 | `tables/make_tables.py (mle_population_table)` | `aval/hist_*_*.npz`<br>`aval/mle_pop/boot.json`<br>`aval/mle_pop/point.json`<br>`aval/planC_sweep/planC.json` | `avalanche_histograms.py`<br>`run_aval_mle_pop.py`<br>`run_planC.py --tag sweep` |
| Supplementary Table S11 | `tables/make_tables.py (graph_type_table)` | `graph_types/graph_types.json` | `run_graph_types.py` |

Supplementary Table S7 and Supplementary Fig. S5c, f, i-k start at the knee of the CCDF. The knee positions and the
least-squares exponents come from the knee procedure of Methods 7, run on the regenerated core data with the
first-submission script `figures/fig_supp_knee_sensitivity.py` (CCDF restricted to log10 P >= -4); they are entered as
constants in `analyses/hump_same_region.py`.

`aval/hist_*_*.npz` are the per-trial avalanche-size histograms of the long records; `figure_common.pooled()` reads
them for Supplementary Figs S2 and S4-S6. Supplementary Figs S1-S2, S4-S9 are written by `figures_final_supp` as
`fig_SF`, `fig_SG`, `fig_SA`, `fig_SB`, `fig_SC`, `fig_SH`, `fig_SD` and `fig_SE` (in the order S1, S2, S4, S5, S6,
S7, S8, S9), and S3, S10-S12 as `fig_hybrid`, `fig_plateau`, `fig_fpt_size` and `fig_networks`.

## Numbers quoted in the text

Every number quoted in the text was recomputed from these outputs. The numbers of the supplementary text are
collected by `tables/make_tables.py` in `tables/quoted_numbers.json`; those of Fig. 6 by `analyses/fig6_numbers.py`
in `analyses/runs/fig6/fig6_numbers.json`.

## First submission

The scripts in `figures/` and `diagnostics/` draw the figures of the first version of the manuscript from the data
of that version; they are archived with release v1.0 (Zenodo DOI 10.5281/zenodo.19812421). The runners in
`experiments/` are shared by both versions.
