"""Run an experiment's tasks over seeds in parallel, one pickle per task (resumable), then merge into CSVs.

    python runs/run.py EXP SEEDS [--workers N] [--out DIR]
SEEDS: e.g. 0-9 or 7010 or 7000,7001. Results go to results/<out or EXP>/<table>.csv; parts in .../parts/.
"""
import argparse
import os
import pickle
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def task_name(t):
    return "_".join(str(x) for x in t).replace(" ", "").replace("(", "").replace(")", "").replace(",", "-")


def work(args):
    exp, t, path = args
    import exps
    t0 = time.time()
    res = exps.run_task(exp, t)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(path + ".tmp", path)
    return t, time.time() - t0


def merge(outdir):
    import pandas as pd
    tables = {}
    for fn in sorted(os.listdir(os.path.join(outdir, "parts"))):
        if fn.endswith(".pkl"):
            with open(os.path.join(outdir, "parts", fn), "rb") as f:
                for k, rows in pickle.load(f).items():
                    tables.setdefault(k, []).extend(rows)
    for k, rows in tables.items():
        pd.DataFrame(rows).to_csv(os.path.join(outdir, f"{k}.csv"), index=False)
    return {k: len(v) for k, v in tables.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("exp"); ap.add_argument("seeds")
    ap.add_argument("--workers", type=int, default=14); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import exps
    outdir = os.path.join(ROOT, "results", a.out or a.exp)
    os.makedirs(os.path.join(outdir, "parts"), exist_ok=True)
    todo = []
    for t in exps.tasks(a.exp, parse_seeds(a.seeds)):
        path = os.path.join(outdir, "parts", task_name(t) + ".pkl")
        if not os.path.exists(path):
            todo.append((a.exp, t, path))
    print(f"{a.exp}: {len(todo)} tasks to run -> {outdir}", flush=True)
    t0 = time.time()
    # longest-looking tasks first: grid and ensemble tasks before the rest
    todo.sort(key=lambda x: (0 if x[1][0] in ("grid", "ens") else 1))
    with Pool(min(a.workers, max(1, len(todo)))) as pool:
        for i, (t, dt) in enumerate(pool.imap_unordered(work, todo), 1):
            print(f"[{time.time() - t0:7.0f}s] {i}/{len(todo)} {t} ({dt:.0f}s)", flush=True)
    print("merged:", merge(outdir), flush=True)
