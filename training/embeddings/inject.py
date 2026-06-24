"""Install the TempoWAVE digit codebook into a causal language model."""

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from training.embeddings.mwne import TempoWaveEmbedding
from utils.tools import DIGIT_TOKENS, validate_digit_tokenizer


def add_digit_tokens(tokenizer):
    tokenizer.add_tokens(list(DIGIT_TOKENS), special_tokens=False)
    return validate_digit_tokenizer(tokenizer)


def untie_output_embeddings(model):
    """Give the LM head independent weights before freezing input embeddings."""

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The causal language model does not expose output embeddings")
    if output_embeddings.weight is not input_embeddings.weight:
        return False

    output_layer = nn.Linear(
        input_embeddings.weight.shape[1],
        input_embeddings.weight.shape[0],
        bias=output_embeddings.bias is not None,
        device=input_embeddings.weight.device,
        dtype=input_embeddings.weight.dtype,
    )
    with torch.no_grad():
        output_layer.weight.copy_(output_embeddings.weight)
        if output_embeddings.bias is not None:
            output_layer.bias.copy_(output_embeddings.bias)
    model.set_output_embeddings(output_layer)
    model.config.tie_word_embeddings = False
    return True


def freeze_tempowave_input_embeddings(model):
    """Freeze the input codebook while requiring an independent output head."""

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise ValueError("The causal language model does not expose output embeddings")
    if output_embeddings.weight is input_embeddings.weight:
        raise ValueError(
            "TempoWAVE requires untied input and output embeddings before training"
        )
    input_embeddings.weight.requires_grad_(False)
    output_embeddings.weight.requires_grad_(True)


def inject_tempowave_embeddings(
    model,
    tokenizer,
    wavelet_types=("haar", "db4", "mexh"),
    scales=(1.0, 2.0, 4.0),
    grid_resolution=1000,
    alignment="project",
    projection_seed=3407,
):
    token_ids = add_digit_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    output_was_untied = untie_output_embeddings(model)

    input_embeddings = model.get_input_embeddings()
    embedding_dim = input_embeddings.weight.shape[1]
    tempowave = TempoWaveEmbedding(
        embedding_dim=embedding_dim,
        wavelet_types=wavelet_types,
        scales=scales,
        grid_resolution=grid_resolution,
        alignment=alignment,
        projection_seed=projection_seed,
        device=input_embeddings.weight.device,
    )
    codebook = tempowave.codebook.to(
        device=input_embeddings.weight.device,
        dtype=input_embeddings.weight.dtype,
    )

    with torch.no_grad():
        input_embeddings.weight[token_ids] = codebook
    freeze_tempowave_input_embeddings(model)

    return {
        "digit_tokens": list(DIGIT_TOKENS),
        "digit_token_ids": token_ids,
        "wavelet_types": list(wavelet_types),
        "scales": [float(scale) for scale in scales],
        "grid_resolution": int(grid_resolution),
        "alignment": alignment,
        "projection_seed": int(projection_seed),
        "minimum_separation": float(tempowave.minimum_separation().item()),
        "output_embeddings_untied": output_was_untied,
    }


def save_tempowave_config(output_dir, config):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "tempowave_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def copy_tempowave_config(model_path, output_dir):
    source = Path(model_path) / "tempowave_config.json"
    if source.exists():
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination / source.name)
