# Source before running Python here: keeps every cache inside this folder (CLAUDE.md).
_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PIP_CACHE_DIR="$_root/.cache/pip" NUMBA_CACHE_DIR="$_root/.cache/numba" \
       MPLCONFIGDIR="$_root/.cache/mpl" PYTHONPYCACHEPREFIX="$_root/.cache/pycache"
export PY="$_root/.venv/Scripts/python.exe"
