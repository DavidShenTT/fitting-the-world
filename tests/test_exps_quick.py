"""Code check: every experiment task runs end to end with shrunken settings (tuning seed 7099; not a result)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import config as C  # noqa: E402

C.WARM, C.E1_STEPS, C.E1_EVERY, C.RAISE_STEPS, C.ABANDON = 600, 1200, 600, 600, 200
C.E3_CAP, C.E3_EARLY, C.E3_LL_AT, C.CODE_PAIRS = 1000, 500, 250, 200
C.WIDTH_LADDER, C.T2, C.PRETRAIN_CAP, C.BUDGETS, C.UPD_CAP, C.E4_PAIRS = (8, 32), 2000, 4000, (500, 1000), 4000, 1
C.EPS_R = {(r, c): 0.02 for r in ("wear", "eye1") for c in ("skip", "passive")}
C.E2_RANGE = dict(M=(2,), s_u=(0.1,), delta=(0.001,), M_mid=2, s_u_mid=0.1, delta_mid=0.001)
C.E1_M, C.N_AG_GRID, C.N_AG_PRIMARY, C.N_ENS = (2,), 1, 1, 1
import exps  # noqa: E402


def _check(res):
    assert res and all(len(rows) > 0 for k, rows in res.items() if k not in ("upd", "worlds")), {k: len(v) for k, v in res.items()}


def test_all_tasks_run():
    for exp in sys.argv[1:] or ("e1", "e2", "e3", "e4", "e6", "e8", "e10"):
        T = exps.tasks(exp, [7099])
        # defaults captured at import time: rebind shrunken budgets where functions took them as defaults
        for t in T:
            _check(exps.run_task(exp, t))
        print("ok", exp, len(T))


if __name__ == "__main__":
    test_all_tasks_run()
