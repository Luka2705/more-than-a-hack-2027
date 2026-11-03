#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements-solution.txt
fi
.venv/bin/python -m solution.infer --intermediates
.venv/bin/python -m solution.verify
printf '\nFertig: artifacts/Import-Export.csv\n'
