"""Metadata and validation for the TempoWAVE evaluation benchmarks."""

BENCHMARKS = {
    "AUL": {
        "name": "Australian electricity load",
        "collection": "From News to Forecast",
        "reference": "https://arxiv.org/abs/2409.17515",
        "source": "https://github.com/ameliawong1996/From_News_to_Forecast",
    },
    "BIT": {
        "name": "Bitcoin price",
        "collection": "From News to Forecast",
        "reference": "https://arxiv.org/abs/2409.17515",
        "source": "https://github.com/ameliawong1996/From_News_to_Forecast",
    },
    "MSPG": {
        "name": "Melbourne solar power generation",
        "collection": "CGTSF",
        "reference": "https://arxiv.org/abs/2412.11376",
        "source": "https://huggingface.co/datasets/ChengsenWang/CGTSF",
    },
    "PTF": {
        "name": "Paris traffic flow",
        "collection": "CGTSF",
        "reference": "https://arxiv.org/abs/2412.11376",
        "source": "https://huggingface.co/datasets/ChengsenWang/CGTSF",
    },
    "LEU": {
        "name": "London electricity usage",
        "collection": "CGTSF",
        "reference": "https://arxiv.org/abs/2412.11376",
        "source": "https://huggingface.co/datasets/ChengsenWang/CGTSF",
    },
}

SUPPORTED_BENCHMARKS = frozenset(BENCHMARKS)


def normalize_benchmark_name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Each benchmark record requires a non-empty 'dataset' field")
    name = value.strip().upper()
    if name not in SUPPORTED_BENCHMARKS:
        supported = ", ".join(sorted(SUPPORTED_BENCHMARKS))
        raise ValueError(
            f"Unsupported TempoWAVE benchmark {value!r}; expected one of: {supported}"
        )
    return name
