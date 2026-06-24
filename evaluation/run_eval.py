"""Evaluate TempoWAVE with the paper's MAE/RMSE forecasting protocol."""

import argparse
import json
import time
from collections import defaultdict

import numpy as np

from data.benchmark_registry import normalize_benchmark_name
from evaluation.model import TimeSeriesForecaster
from utils.config import parse_args_with_config


def parse_numeric_series(value, field_name):
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, list):
        parts = value
    else:
        raise ValueError(f"{field_name} must be a comma-separated string or list")
    if not parts:
        raise ValueError(f"{field_name} is empty")
    try:
        result = np.asarray([float(part) for part in parts], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains non-numeric values") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{field_name} contains NaN or infinite values")
    return result


def load_eval_records(input_file_path, hist_len=None, pred_len=None):
    with open(input_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation file must contain a non-empty JSON list")

    records = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} must be a JSON object")
        missing = {"dataset", "text", "input", "output"} - set(item)
        if missing:
            raise ValueError(f"Record {index} is missing required fields: {sorted(missing)}")

        history = parse_numeric_series(item["input"], f"record {index} input")
        target = parse_numeric_series(item["output"], f"record {index} output")
        if hist_len is not None and len(history) != hist_len:
            raise ValueError(
                f"Record {index} input length is {len(history)}, expected hist_len={hist_len}"
            )
        if pred_len is not None and len(target) != pred_len:
            raise ValueError(
                f"Record {index} output length is {len(target)}, expected pred_len={pred_len}"
            )
        records.append(
            {
                "input": history,
                "text": item["text"],
                "output": target,
                "normalization": item.get("normalization", {"kind": "identity"}),
                "dataset": normalize_benchmark_name(item["dataset"]),
            }
        )
    return records


def _valid_pairs(ground_truth, predictions):
    ground_truth = np.asarray(ground_truth, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    valid = np.isfinite(ground_truth) & np.isfinite(predictions)
    return ground_truth[valid], predictions[valid], int((~valid).sum())


def calculate_mae(ground_truth, predictions):
    ground_truth, predictions, _ = _valid_pairs(ground_truth, predictions)
    return np.mean(np.abs(ground_truth - predictions)) if ground_truth.size else np.nan


def calculate_rmse(ground_truth, predictions):
    ground_truth, predictions, _ = _valid_pairs(ground_truth, predictions)
    return (
        np.sqrt(np.mean(np.square(ground_truth - predictions)))
        if ground_truth.size
        else np.nan
    )


def print_metrics(title, metrics):
    print(f"\n{title}")
    for metric in ("mae", "rmse"):
        values = np.asarray(metrics[metric], dtype=float)
        print(
            f"{metric.upper()}: {np.nanmean(values):.4f} "
            f"+/- {np.nanstd(values):.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate TempoWAVE forecasting")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--hist_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=48)
    parser.add_argument("--integer_precision", type=int, default=1)
    parser.add_argument("--fractional_precision", type=int, default=4)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--invalid_policy",
        choices=["error", "exclude"],
        default="error",
        help=(
            "How to handle forecasts that remain malformed after deterministic "
            "fallback parsing. 'error' prevents silently optimistic metrics."
        ),
    )
    args = parse_args_with_config(parser)

    records = load_eval_records(
        args.input_file,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
    )
    model = TimeSeriesForecaster(
        model_path=args.model_path,
        pred_len=args.pred_len,
        integer_precision=args.integer_precision,
        fractional_precision=args.fractional_precision,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    metrics = {"mae": [], "rmse": []}
    dataset_metrics = defaultdict(lambda: {"mae": [], "rmse": []})
    invalid_points = 0
    total_points = len(records) * args.pred_len
    start_time = time.time()
    for index, record in enumerate(records, start=1):
        prediction, details = model.predict(
            record["text"],
            normalization=record["normalization"],
        )
        invalid_points += details["invalid_count"]
        if details["invalid_count"] and args.invalid_policy == "error":
            raise ValueError(
                f"Record {index - 1} contains {details['invalid_count']} invalid "
                "forecast values after fallback parsing"
            )
        for metric, function in (("mae", calculate_mae), ("rmse", calculate_rmse)):
            value = function(record["output"], prediction)
            metrics[metric].append(value)
            dataset_metrics[record["dataset"]][metric].append(value)
        print(
            f"[{index}/{len(records)}] "
            f"{record['dataset']} "
            f"MAE={metrics['mae'][-1]:.4f} RMSE={metrics['rmse'][-1]:.4f} "
            f"invalid={details['invalid_count']}"
        )

    print_metrics("Overall Evaluation Metrics", metrics)
    for dataset in sorted(dataset_metrics):
        print_metrics(f"{dataset} Evaluation Metrics", dataset_metrics[dataset])
    print(
        f"Valid numeric coverage: "
        f"{(total_points - invalid_points) / total_points:.2%} "
        f"({invalid_points} invalid points)"
    )
    print(f"Elapsed: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
