#!/bin/bash
# E2 grid extension (preregistered fallback) and the smoke test: seed 7010 through every experiment at full size.
cd "$(dirname "$0")/.." && source env.sh
"$PY" runs/run.py tune_e2_ext 7020-7022 --out tuning/e2 > logs/tune_e2_ext.log 2>&1
for e in e10 e8 e3 e4 e6 e1 e2; do
  "$PY" runs/run.py $e 7010 --out smoke/$e > logs/smoke_$e.log 2>&1
done
echo done > logs/stage2_smoke.done
