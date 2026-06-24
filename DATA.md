# TempoWAVE Data Contract

## Benchmark Datasets

TempoWAVE supports exactly the five context-enriched test sets used in the
paper:

| ID | Test set | Collection | Primary reference | Source records |
| --- | --- | --- | --- | --- |
| AUL | Australian electricity load | From News to Forecast | [paper](https://arxiv.org/abs/2409.17515) | [repository](https://github.com/ameliawong1996/From_News_to_Forecast) |
| BIT | Bitcoin price | From News to Forecast | [paper](https://arxiv.org/abs/2409.17515) | [repository](https://github.com/ameliawong1996/From_News_to_Forecast) |
| MSPG | Melbourne solar power generation | CGTSF | [ChatTime paper](https://arxiv.org/abs/2412.11376) | [dataset](https://huggingface.co/datasets/ChengsenWang/CGTSF) |
| PTF | Paris traffic flow | CGTSF | [ChatTime paper](https://arxiv.org/abs/2412.11376) | [dataset](https://huggingface.co/datasets/ChengsenWang/CGTSF) |
| LEU | London electricity usage | CGTSF | [ChatTime paper](https://arxiv.org/abs/2412.11376) | [dataset](https://huggingface.co/datasets/ChengsenWang/CGTSF) |

Prepare each collection with the preprocessing, normalization, contextual
fields, and train/validation/test assignments distributed by its source.
`data.prepare_benchmarks` converts these segmented records into the common
TempoWAVE training and evaluation layout.

## Contextual Input Schema

`python -m data.prepare_benchmarks` accepts a JSON list or a mapping with
`train`, `val`, and `test` arrays. Every record requires:

- `dataset`: `AUL`, `BIT`, `MSPG`, `PTF`, or `LEU`;
- `split`: `train`, `val`, or `test`;
- `history`: normalized historical values;
- `target`: normalized forecast values.

Each dataset present in an input file must contain all three official splits.
Unknown names and unlabeled records are rejected so this command cannot
silently turn unrelated raw series into paper evaluation data.

Aliases `Hist`/`Pred` and `input_values`/`output_values` are also accepted.

Paper context fields are optional per record:

- `situational_context`: domain, date, holiday, weather, frequency, and horizon;
- `news`: relevant event or news text;
- `catch22`: a mapping or preformatted string of Catch22 descriptors.

Use `--compute_missing_catch22` to calculate Catch22 from `history`.

Optional metadata:

- `series_id`: source entity, site, household, or detector;
- `raw_history` and `raw_target`: values in original benchmark units;
- `normalization`: inverse-transform metadata.

## Normalization

Identity-normalized records use:

```json
{"kind": "identity"}
```

For an affine transform

```text
normalized = (raw - offset) / scale + normalized_offset
```

store:

```json
{
  "kind": "affine",
  "offset": 7000.0,
  "scale": 4000.0,
  "normalized_offset": -0.5
}
```

Evaluation applies the inverse before MAE and RMSE. When `raw_history` and
`raw_target` are omitted, they are reconstructed from this metadata.

## Digit Encoding

Time-series values use one integer digit and four fractional digits by
default. Every decimal digit becomes one dedicated token:

```text
-0.5000
```

becomes:

```text
-<|digit_0|>.<|digit_5|><|digit_0|><|digit_0|><|digit_0|>
```

Signs, decimal points, commas, and language remain standard tokenizer input.
Situational context, news, dates, weather values, and textualized Catch22
descriptors remain ordinary context tokens. TempoWAVE routing is applied only
to the normalized historical and forecast sequences.

## Generated Layout

```text
data/processed/
├── benchmark_manifest.json
├── pretrain/
│   └── train.csv
├── sft/
│   ├── train.json
│   ├── val.json
│   └── inference.json
├── eval/
│   └── eval.json
└── source/
    ├── train.json
    ├── val.json
    └── test.json
```

- `benchmark_manifest.json` records each included test set, split counts, and
  its primary paper and source repository.
- `pretrain/train.csv` contains digit-formatted numeric sequences.
- `sft/train.json` and `sft/val.json` contain response-bearing Qwen prompts.
- `sft/inference.json` and `eval/eval.json` contain response-free test prompts.
- `source/*.json` preserves the normalized values, raw values, context, and
  inverse-normalization metadata.

The evaluation file retains `dataset` on every record. `evaluation.run_eval`
validates the identifier and reports overall metrics followed by a separate
MAE/RMSE summary for every included test set.
