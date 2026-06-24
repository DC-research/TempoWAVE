"""Shared YAML/JSON command-line configuration helpers."""

import json
from pathlib import Path


def load_config(path):
    if path is None:
        return {}

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    if config_path.suffix.lower() == ".json":
        with config_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Install pyyaml to use YAML config files") from exc
        with config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {config_path}")
    return data


def parse_args_with_config(parser):
    parser.add_argument("--config", type=str, default=None, help="Optional YAML/JSON config file")
    config_args, _ = parser.parse_known_args()
    config = load_config(config_args.config)
    known_fields = {action.dest for action in parser._actions}
    unknown_fields = sorted(set(config) - known_fields)
    if unknown_fields:
        raise ValueError(
            f"Unknown config field(s) for this command: {unknown_fields}"
        )
    parser.set_defaults(**config)
    for action in parser._actions:
        if action.dest in config:
            action.required = False
    return parser.parse_args()
