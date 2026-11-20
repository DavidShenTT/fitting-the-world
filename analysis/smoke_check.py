"""Smoke-test check (PREREGISTRATION.md section 3): every table and column filled. Lists, per CSV, the row count and any
column with missing or non-finite values, for review against the columns allowed to be missing.
Writes results/smoke/smoke_check.txt."""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lines = []
for f in sorted(glob.glob(os.path.join(ROOT, "results", "smoke", "*", "*.csv"))):
    d = pd.read_csv(f)
    num = d.select_dtypes("number")
    bad = {c: int((~np.isfinite(num[c])).sum()) for c in num.columns if (~np.isfinite(num[c])).any()}
    bad.update({c: int(d[c].isna().sum()) for c in d.columns if c not in num.columns and d[c].isna().any()})
    lines.append(f"{os.path.relpath(f, ROOT)}: {len(d)} rows, {d.shape[1]} cols; missing: {bad if bad else 'none'}")
txt = "\n".join(lines)
print(txt)
open(os.path.join(ROOT, "results", "smoke", "smoke_check.txt"), "w").write(txt + "\n")
