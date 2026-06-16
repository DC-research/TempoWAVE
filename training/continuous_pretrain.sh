#!/usr/bin/env bash
set -euo pipefail

config_path="${CONTINUOUS_PRETRAIN_CONFIG:-configs/continuous_pretrain.yaml}"

python -m training.continuous_pretrain --config "$config_path" "$@"
