#!/usr/bin/env bash
set -euo pipefail

config_path="${PRETRAIN_CONFIG:-configs/pretrain.yaml}"

python -m training.pretrain --config "$config_path" "$@"
