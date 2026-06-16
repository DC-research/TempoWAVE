#!/usr/bin/env bash
set -euo pipefail

config_path="${EVAL_CONFIG:-configs/eval.yaml}"

python -m evaluation.run_eval --config "$config_path" "$@"
