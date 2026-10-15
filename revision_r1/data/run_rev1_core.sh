#!/bin/bash
# Regenerate the core datasets (N = 20, k = 3) with the seeded solver. Writes the .pkl files into this directory.
# The solver repository is two levels up in the released repository (../../github_repo in the authors' working tree).
cd "$(dirname "$0")"
REPO=../..; [ -f $REPO/gcg_solver.py ] || REPO=../../github_repo
export PYTHONPATH=$REPO
for r in run_criticality_20_3 run_phase_space_20_3; do
  echo "[$(date '+%F %T')] START $r"
  python3 $REPO/experiments/$r.py > $r.log
  rc=$?
  echo "[$(date '+%F %T')] END $r exit=$rc"
  [ $rc -ne 0 ] && { echo "FAILED $r"; exit $rc; }
done
echo "ALL_DONE"
