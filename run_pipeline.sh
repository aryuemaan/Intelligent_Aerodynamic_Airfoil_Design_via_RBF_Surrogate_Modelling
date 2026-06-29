#!/usr/bin/env bash
# Run the full pipeline. From the project root in Git Bash:
#     bash run_pipeline.sh            # uses whatever is in data/airfoil_cfd_9k
#     bash run_pipeline.sh --demo     # generate synthetic data first, then run
set -euo pipefail

export PYTHONPATH="src"
# Activate venv if present
if [ -f ".venv/Scripts/activate" ]; then source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi

RUN="python -m airfoil_rbf"

if [ "${1:-}" = "--demo" ]; then
    echo ">> Generating synthetic dataset (60 shapes)"
    $RUN make-synthetic --n 60
fi

echo ">> [1/5] build features"
$RUN build
echo ">> [2/5] assemble X/Y"
$RUN assemble
echo ">> [3/5] filter"
$RUN filter
echo ">> [4/5] train"
$RUN train
echo ">> [5/5] evaluate"
$RUN evaluate

echo
echo ">> Pipeline complete. See figures/ and models/metrics.json"
echo ">> Try an inverse design, e.g.:"
echo "     $RUN design --aoa 4 --style -2.5 0.3 -0.8 0.3"
