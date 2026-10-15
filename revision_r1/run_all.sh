#!/bin/bash
# Every analysis, figure and table of the revised paper, in dependency order.
#
# Inputs (see README.md in this directory):
#   data/result_for_criticality_20_3.pkl, data/result_for_phase_space_20_3.pkl   core data (data/run_rev1_core.sh)
#   <repo>/result_for_plateau_k34.pkl, <repo>/result_for_scalability.pkl          parameter scans of Fig. 6
# Outputs: analyses/runs/ (analysis results), figures_final*/ (figures), tables/ (supplementary tables).
#
# Usage:  bash run_all.sh [analyses|figures|tables|all]   (default: all)
# Stages can be run separately; the figures and tables need only analyses/runs.
set -euo pipefail
cd "$(dirname "$0")"
HERE=$(pwd)
STAGE=${1:-all}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

analyses() {
  cd "$HERE/analyses"
  mkdir -p runs/aval runs/extension
  # Avalanches (core long records): per-trial histograms, two populations, maximum likelihood
  python3 avalanche_histograms.py
  python3 run_planC.py --key sweep --B 500 --chunk 50 --tag sweep          # sweep clock (primary)
  python3 run_planC.py --key th1_dt1 --B 500 --tag primary                 # step clock
  for k in sweep_th0_k1 sweep_th2_k1 sweep_th3_k1 sweep_th1_k2 sweep_nobound sweep_stat sweep_T10000 sweep_T25000; do
    python3 run_planC.py --key $k --B 100 --chunk 25 --tag A3_$k             # definition sensitivity
  done
  python3 run_episodes.py
  python3 run_sstar_rule_sweep.py
  AVAL_XMIN_EXTEND=1 python3 run_aval_mle.py --key 'sweep|all' --B 0 --tag sweep
  python3 run_aval_mle_boot.py --tag sweep --key 'sweep|all' --B 200 --chunk 20
  python3 run_aval_mle_pop.py
  python3 hump_same_region.py
  python3 run_hump_lowercut.py
  # Reaching and holding
  python3 run_first_passage.py
  python3 run_mannwhitney.py
  python3 run_robustness.py
  python3 run_factorial_2x2.py
  python3 pilot_E1_E4.py
  python3 pre_solution.py
  # Spectra and branching
  python3 run_spectra.py
  python3 spectra_supplement.py
  python3 run_agent_spectra.py
  python3 export_supp_data.py
  python3 run_branching.py
  python3 run_branching_by_class.py
  # Injected inconsistencies, leaving the consistent state, graph types
  python3 run_kick.py
  python3 run_kick_growth.py
  python3 run_kick_vs_natural.py
  python3 run_departure_mechanism.py --ps 0.05,0.45,0.85
  python3 run_graph_types.py
  python3 save_trial_types.py
  # Network size, network ensembles, mixing (parameter scans of Fig. 6 and new simulations)
  python3 fig6_numbers.py
  python3 run_fpt_size.py
  python3 run_supp_extra.py plateau hybrid networks
  python3 run_colourability.py
  (cd runs/extension && PYTHONPATH="$(python3 -c 'import sys; sys.path.insert(0, "../.."); from repo_path import REPO; print(REPO)')" \
     python3 ../../run_budget_extension.py)
}

figures() {
  cd "$HERE/figures_final"
  for n in 3 4 5 6; do python3 plot_fig$n.py; done
  python3 fig1/plot_fig1.py
  python3 fig2/plot_fig2.py
  cd "$HERE/figures_final_supp"
  for s in SA SB SC SD SE SF SG SH; do python3 plot_$s.py; done
  python3 plot_extra.py
}

tables() {
  cd "$HERE/tables"
  python3 make_tables.py
}

case "$STAGE" in
  analyses) analyses ;;
  figures) figures ;;
  tables) tables ;;
  all) analyses; figures; tables ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
