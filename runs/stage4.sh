#!/bin/bash
# Stage 4 (E2-E6) on seeds 0-9, plus the E8 fixed-pretraining arm (D3) and the passive matched rerun (D4).
cd "$(dirname "$0")/.." && source env.sh
for e in e10_matched_v2 e8_fixed e4 e3 e6 e2; do
  "$PY" runs/run.py $e 0-9 > logs/$e.log 2>&1
done
echo done > logs/stage4.done
