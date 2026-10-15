"""Graph type (breakable / locked; run_graph_types.py) of every trial index of the per-trial avalanche histograms.
Trial index = row order of each (model, p) group of the core data, as in avalanche_histograms.py.
Output: runs/graph_types/trial_types.json.  Run from revision_r1/analyses."""
import json
import pandas as pd
from run_first_passage import groups
from run_graph_types import TYPE

df = pd.read_pickle('../data/result_for_criticality_20_3.pkl')
out = {f'{m}_{p:.2f}': [TYPE[int(s)] for s in g['seed']] for m, p, rate, g in groups(df)}
json.dump(out, open('runs/graph_types/trial_types.json', 'w'))
print({k: sum(t == 'locked' for t in v) for k, v in out.items()})
