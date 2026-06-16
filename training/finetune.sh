#!/usr/bin/env bash
set -euo pipefail

config_path="${SFT_CONFIG:-configs/sft.yaml}"

python -m training.finetune --config "$config_path" "$@"
