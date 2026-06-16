"""Prepare contextual TempoWAVE supervised fine-tuning data."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = REPO_ROOT / "training" / "templates" / "time_series_qwen.json"
REPRESENTATIONS = ("numeric", "raw")
DEFAULT_CONTEXT_FIELDS = ("situational_context", "news", "catch22")


def load_json_records(path):
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{input_path} must contain a JSON list")
    return records


def load_template(path):
    template_path = Path(path)
    with template_path.open("r", encoding="utf-8") as f:
        template = json.load(f)
    if "prompt_input" not in template:
        raise ValueError(f"Prompt template {template_path} is missing 'prompt_input'")
    return template


def representation_value(value):
    if isinstance(value, dict):
        if "digit_context" in value:
            value = value["digit_context"]
        elif "dispersed_context" in value:
            value = value["dispersed_context"]
        else:
            raise ValueError(
                "Encoded series objects must contain 'digit_context'"
            )
    if not isinstance(value, str):
        raise ValueError(f"Time-series values must be strings, got {type(value).__name__}")
    return value


def _context_lookup(record, represented, field):
    aliases = {
        "situational_context": ("situational_context", "input", "context"),
        "news": ("news", "News"),
        "catch22": ("catch22", "Catch22"),
    }
    candidates = aliases.get(field.lower(), (field,))
    containers = (record.get("context", {}), represented, record)
    for container in containers:
        for candidate in candidates:
            if isinstance(container, dict) and container.get(candidate):
                return container[candidate]
    return None


def _format_catch22(value):
    if isinstance(value, dict):
        formatted = []
        for name, number in value.items():
            if isinstance(number, (int, float)):
                number = f"{number:.4f}"
            formatted.append(f"{name}: {number}")
        return ",\n".join(formatted)
    if isinstance(value, list):
        return ",\n".join(str(item) for item in value)
    return str(value)


def build_sft_record(
    record,
    template,
    representation="numeric",
    context_fields=None,
    pred_len=48,
    horizon_label="forecast horizon",
    include_response=True,
):
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"representation must be one of {REPRESENTATIONS}, got {representation!r}"
        )
    if representation not in record:
        raise ValueError(f"Record is missing representation {representation!r}")
    if "raw" not in record:
        raise ValueError("Record is missing the 'raw' representation")

    represented = record[representation]
    raw = record["raw"]
    try:
        history = representation_value(represented["instruction"])
        response = representation_value(represented["output"])
        raw_history = representation_value(raw["instruction"])
        raw_response = representation_value(raw["output"])
    except KeyError as exc:
        raise ValueError(f"Record is missing required field {exc.args[0]!r}") from exc

    response_values = [part.strip() for part in raw_response.split(",") if part.strip()]
    if len(response_values) != pred_len:
        raise ValueError(
            f"Record output contains {len(response_values)} values, "
            f"expected pred_len={pred_len}"
        )

    instruction_parts = [
        "Please predict the following sequence carefully.",
        (
            f"Based on the provided time series data and contextual information, "
            f"predict the next {pred_len} data points ({horizon_label}). "
            f"Your response should only contain the next {pred_len} values."
        ),
    ]

    for field in context_fields or DEFAULT_CONTEXT_FIELDS:
        value = _context_lookup(record, represented, field)
        if value is None:
            continue
        if field.lower() == "catch22":
            value = (
                "To assess similarity between sequences, consider these Catch22 "
                f"statistical descriptors:\n{_format_catch22(value)}"
            )
        else:
            value = str(value)
        instruction_parts.append(value)

    prompt = template["prompt_input"].format(
        input=f"Historical time-series values: {history}",
        instruction="\n\n".join(instruction_parts),
    )
    if include_response:
        prompt += response + "<|im_end|>\n"

    result = {
        "text": prompt,
        "input": raw_history,
        "output": raw_response,
    }
    for field in ("normalization", "dataset", "series_id"):
        if field in record:
            result[field] = record[field]
    return result


def prepare_dataset(
    input_path,
    output_path,
    template,
    representation="numeric",
    context_fields=None,
    pred_len=48,
    horizon_label="forecast horizon",
    include_response=True,
):
    records = load_json_records(input_path)
    prepared = [
        build_sft_record(
            record,
            template,
            representation=representation,
            context_fields=context_fields,
            pred_len=pred_len,
            horizon_label=horizon_label,
            include_response=include_response,
        )
        for record in tqdm(records, desc=f"Preparing {Path(output_path).name}")
    ]

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        json.dump(prepared, f, ensure_ascii=False, indent=2)
    return prepared


def main():
    parser = argparse.ArgumentParser(
        description="Prepare contextual TempoWAVE supervised fine-tuning data"
    )
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--output_prefix", default="sft")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--representation", choices=REPRESENTATIONS, default="numeric")
    parser.add_argument(
        "--context_fields",
        nargs="*",
        default=list(DEFAULT_CONTEXT_FIELDS),
    )
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--horizon_label", default="forecast horizon")
    args = parser.parse_args()

    template = load_template(args.template)
    dataset_dir = Path(args.output_dir) / args.output_prefix
    common = {
        "template": template,
        "representation": args.representation,
        "context_fields": args.context_fields,
        "pred_len": args.pred_len,
        "horizon_label": args.horizon_label,
    }
    prepare_dataset(
        args.train_file,
        dataset_dir / "train.json",
        include_response=True,
        **common,
    )
    if args.val_file:
        prepare_dataset(
            args.val_file,
            dataset_dir / "val.json",
            include_response=True,
            **common,
        )
        prepare_dataset(
            args.val_file,
            dataset_dir / "inference.json",
            include_response=False,
            **common,
        )


if __name__ == "__main__":
    main()
