"""Location of the solver repository (gcg_solver.py, experiments/ and the parameter-scan data result_*.pkl).

In the released repository the analyses live in <repo>/revision_r1/analyses, so the repository is two levels up.
In the authors' working tree it is ../../github_repo.  The environment variable GCG_REPO overrides both.
"""
import os

_up = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
REPO = os.environ.get('GCG_REPO') or (_up if os.path.exists(os.path.join(_up, 'gcg_solver.py')) else os.path.join(_up, 'github_repo'))
