"""Prepare the five TempoWAVE evaluation benchmarks from official split records."""

import argparse
import json
from pathlib import Path

import numpy as np

from data.benchmark_pipeline import prepare_benchmark_outputs
from data.benchmark_registry import normalize_benchmark_name
from data.prepare_finetune import load_template
from utils.tools import AffineNormalizer, DigitSerializer


def _numeric_array(value, field):
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain numeric values") from exc
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{field} must be a non-empty finite one-dimensional series")
    return result


def _pick(record, *names, required=False):
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    if required:
        raise ValueError(f"Record is missing one of required fields: {names}")
    return None


def compute_catch22(history):
    try:
        import pycatch22
    except ImportError as exc:
        raise ImportError(
            "Install pycatch22 or provide precomputed Catch22 values"
        ) from exc
    result = pycatch22.catch22_all(np.asarray(history, dtype=float).tolist())
    return dict(zip(result["names"], result["values"]))


def load_records(path):
    source = Path(path)
    with source.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        flattened = []
        for split, records in data.items():
            for record in records:
                flattened.append({**record, "split": split})
        data = flattened
    if not isinstance(data, list) or not data:
        raise ValueError("Contextual input must be a non-empty JSON list or split mapping")
    return data


def build_benchmark_record(
    record,
    compute_missing_catch22=False,
    integer_precision=1,
    fractional_precision=4,
    pred_len=None,
):
    dataset = normalize_benchmark_name(_pick(record, "dataset"))
    split = str(_pick(record, "split", required=True)).lower()
    if split in {"validation", "valid"}:
        split = "val"
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split {split!r}; use train, val, or test")

    history = _numeric_array(
        _pick(record, "history", "Hist", "input_values", required=True),
        "history",
    )
    target = _numeric_array(
        _pick(record, "target", "Pred", "output_values", required=True),
        "target",
    )
    if pred_len is not None and len(target) != pred_len:
        raise ValueError(
            f"target contains {len(target)} values, expected pred_len={pred_len}"
        )
    raw_history_value = _pick(record, "raw_history", "raw_input")
    raw_target_value = _pick(record, "raw_target", "raw_output")
    normalization = record.get("normalization", {"kind": "identity"})
    normalizer = AffineNormalizer.from_metadata(normalization)
    raw_history = (
        _numeric_array(raw_history_value, "raw_history")
        if raw_history_value is not None
        else normalizer.inverse_transform(history)
    )
    raw_target = (
        _numeric_array(raw_target_value, "raw_target")
        if raw_target_value is not None
        else normalizer.inverse_transform(target)
    )

    catch22 = _pick(record, "catch22", "Catch22")
    if catch22 is None and compute_missing_catch22:
        catch22 = compute_catch22(history)

    context = {
        "situational_context": _pick(
            record,
            "situational_context",
            "context",
            "input",
        ),
        "news": _pick(record, "news", "News"),
        "catch22": catch22,
    }
    context = {key: value for key, value in context.items() if value}
    serializer = DigitSerializer(
        integer_precision=integer_precision,
        fractional_precision=fractional_precision,
    )
    prepared = {
        "raw": {
            "instruction": ",".join(str(float(value)) for value in raw_history),
            "output": ",".join(str(float(value)) for value in raw_target),
        },
        "numeric": {
            "instruction": {"digit_context": serializer.serialize(history)},
            "output": {"digit_context": serializer.serialize(target)},
        },
        "context": context,
        "normalization": normalization,
        "dataset": dataset,
    }
    for field in ("series_id",):
        if field in record:
            prepared[field] = record[field]
    return split, prepared


def validate_benchmark_splits(splits):
    datasets = {
        record["dataset"]
        for split_records in splits.values()
        for record in split_records
    }
    if not datasets:
        raise ValueError("Benchmark input contains no records")
    for dataset in sorted(datasets):
        missing = [
            split
            for split, split_records in splits.items()
            if not any(record["dataset"] == dataset for record in split_records)
        ]
        if missing:
            raise ValueError(
                f"{dataset} is missing official split records for: {', '.join(missing)}"
            )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare AUL, BIT, MSPG, PTF, or LEU records while preserving "
            "official splits and dataset normalization"
        )
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--horizon_label", default="forecast horizon")
    parser.add_argument(
        "--template",
        default=str(
            Path(__file__).resolve().parents[1]
            / "training"
            / "templates"
            / "time_series_qwen.json"
        ),
    )
    parser.add_argument("--compute_missing_catch22", action="store_true")
    parser.add_argument("--integer_precision", type=int, default=1)
    parser.add_argument("--fractional_precision", type=int, default=4)
    args = parser.parse_args()

    splits = {"train": [], "val": [], "test": []}
    for source_record in load_records(args.input_file):
        split, prepared = build_benchmark_record(
            source_record,
            compute_missing_catch22=args.compute_missing_catch22,
            integer_precision=args.integer_precision,
            fractional_precision=args.fractional_precision,
            pred_len=args.pred_len,
        )
        splits[split].append(prepared)

    validate_benchmark_splits(splits)
    prepare_benchmark_outputs(
        splits,
        output_dir=args.output_dir,
        template=load_template(args.template),
        pred_len=args.pred_len,
        horizon_label=args.horizon_label,
    )
    counts = ", ".join(f"{split}={len(records)}" for split, records in splits.items())
    print(f"Wrote TempoWAVE benchmark artifacts to {args.output_dir} ({counts})")


if __name__ == "__main__":
    main()
