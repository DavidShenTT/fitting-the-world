#!/bin/bash
# Stage-2 tuning and sanity runs on seeds 7000-7099 (PREREGISTRATION.md section 3). Logs in logs/.
cd "$(dirname "$0")/.." && source env.sh
"$PY" runs/run.py tune_epsr 7000-7004 --out tuning/epsr > logs/tune_epsr.log 2>&1
"$PY" runs/run.py sanity_wear 7030-7032 --out tuning/sanity_wear > logs/sanity_wear.log 2>&1
"$PY" runs/run.py tune_e2 7020-7022 --out tuning/e2 > logs/tune_e2.log 2>&1
echo done > logs/stage2_tuning.done
