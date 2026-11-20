#!/bin/bash
# Stage 3: E10 (with E9 and the update-count control inside) and E8 on seeds 0-9. Logs in logs/.
cd "$(dirname "$0")/.." && source env.sh
"$PY" runs/run.py e10 0-9 > logs/e10.log 2>&1
"$PY" runs/run.py e8 0-9 > logs/e8.log 2>&1
echo done > logs/stage3.done
