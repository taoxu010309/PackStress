#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MAX_JOBS="${MAX_JOBS:-4}"
python -m pip install --no-build-isolation --no-deps ./third_party/causal-conv1d
python -m pip install --no-build-isolation --no-deps ./third_party/mamba
python -c 'import inspect; from mamba_ssm import Mamba; assert "bimamba_type" in inspect.signature(Mamba).parameters; print("Custom BiMamba import OK")'
