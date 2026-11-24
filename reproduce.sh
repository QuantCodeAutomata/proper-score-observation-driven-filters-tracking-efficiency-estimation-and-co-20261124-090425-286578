#!/usr/bin/env bash
# Runs the test suite and every experiment in its default, time-bounded
# configuration, writing all artifacts to results/.
set -uo pipefail
cd "$(dirname "$0")"

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
python -m exp.exp6
python -m exp.exp7
python -m exp.exp8
python -m exp.exp9
