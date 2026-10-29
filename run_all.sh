#!/usr/bin/env bash
# Tests, then every result and the report. The benchmark takes ~10 minutes (two-layer VQE with restarts);
# Delta-learning computes 300 CCSD(T) energies once and caches them in results/water_energies.json.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m unittest discover -s tests -v
python3 benchmark.py
python3 measurement.py
python3 delta_ml.py
python3 make_report.py
