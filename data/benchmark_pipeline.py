"""Write training and evaluation artifacts for TempoWAVE benchmarks."""

import csv
import json
from collections import Counter
from pathlib import Path

from data.benchmark_registry import BENCHMARKS
from data.prepare_finetune import build_sft_record


def write_json(path, records):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def write_pretrain_csv(path, records):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text"])
        writer.writeheader()
        for record in records:
            history = record["numeric"]["instruction"]["digit_context"]
            future = record["numeric"]["output"]["digit_context"]
            writer.writerow({"text": f"{history}, {future}"})


def build_manifest(records):
    datasets = {}
    for split_name, split_records in records.items():
        for dataset, count in Counter(
            record["dataset"] for record in split_records
        ).items():
            entry = datasets.setdefault(
                dataset,
                {
                    **BENCHMARKS[dataset],
                    "splits": {"train": 0, "val": 0, "test": 0},
                },
            )
            entry["splits"][split_name] = count
    return {
        "format": "TempoWAVE benchmark artifacts",
        "datasets": datasets,
    }


def prepare_benchmark_outputs(
    records,
    output_dir,
    template,
    pred_len,
    horizon_label,
):
    output_root = Path(output_dir)
    train_sft = [
        build_sft_record(
            record,
            template,
            pred_len=pred_len,
            horizon_label=horizon_label,
            include_response=True,
        )
        for record in records["train"]
    ]
    val_sft = [
        build_sft_record(
            record,
            template,
            pred_len=pred_len,
            horizon_label=horizon_label,
            include_response=True,
        )
        for record in records["val"]
    ]
    inference_records = [
        build_sft_record(
            record,
            template,
            pred_len=pred_len,
            horizon_label=horizon_label,
            include_response=False,
        )
        for record in records["test"]
    ]

    write_pretrain_csv(output_root / "pretrain" / "train.csv", records["train"])
    write_json(output_root / "sft" / "train.json", train_sft)
    write_json(output_root / "sft" / "val.json", val_sft)
    write_json(output_root / "sft" / "inference.json", inference_records)
    write_json(output_root / "eval" / "eval.json", inference_records)
    write_json(output_root / "benchmark_manifest.json", build_manifest(records))
    for split_name, split_records in records.items():
        write_json(output_root / "source" / f"{split_name}.json", split_records)
