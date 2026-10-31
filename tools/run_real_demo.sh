#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ ! -x .venv/bin/python ]]; then
  uv venv --system-site-packages .venv
  uv pip install --python .venv/bin/python -r requirements-experiment.txt
fi
run_dir="${1:-outputs/real_demo_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
mkdir -- "$run_dir"
.venv/bin/python tools/prepare_assets.py
.venv/bin/python tools/probe_provider.py --output "$run_dir/preflight"
.venv/bin/python -m specnet.experiment calibrate --output "$run_dir/calibration"
.venv/bin/python -m specnet.experiment run --calibration "$run_dir/calibration" --output "$run_dir/comparison" --suite demo --tasks 2 --blocks 1
.venv/bin/python -m specnet.isolation --output "$run_dir/isolation"
