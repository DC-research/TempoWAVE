"""Initialize TempoWAVE digit embeddings and optionally pretrain them."""

import argparse

from transformers import AutoTokenizer
from trl import SFTTrainer
from unsloth import FastLanguageModel

from training.common import (
    MergeAndSaveCallback,
    add_common_training_args,
    add_csv_dataset_args,
    build_training_arguments,
    ensure_output_dirs,
    load_text_dataset,
    print_gpu_start,
    print_training_summary,
)
from training.embeddings.inject import (
    add_digit_tokens,
    inject_tempowave_embeddings,
    save_tempowave_config,
)
from utils.config import parse_args_with_config


def main():
    parser = argparse.ArgumentParser(
        description="Initialize and pretrain the paper-faithful TempoWAVE digit interface"
    )
    add_common_training_args(parser)
    add_csv_dataset_args(parser)
    parser.add_argument(
        "--embedding_model",
        choices=["tempowave", "raw"],
        default="tempowave",
        help="Use TempoWAVE or retain default initialization for the ten digit tokens",
    )
    parser.add_argument(
        "--wavelet_types",
        nargs="+",
        default=["haar", "db4", "mexh"],
    )
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=[1.0, 2.0, 4.0],
    )
    parser.add_argument("--grid_resolution", type=int, default=1000)
    parser.add_argument("--alignment", choices=["pad", "project"], default="project")
    parser.add_argument("--projection_seed", type=int, default=3407)
    parser.add_argument(
        "--train_language_model",
        action="store_true",
        help=(
            "After embedding injection, run optional language-model pretraining. "
            "Omit this flag for the paper's initialization stage."
        ),
    )
    args = parse_args_with_config(parser)
    ensure_output_dirs(args.checkpoint_dir, args.output_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Reserve the ten token rows before model loading so Unsloth allocates the final size.
    add_digit_tokens(tokenizer)
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
        resize_model_vocab=len(tokenizer),
    )

    if args.embedding_model == "tempowave":
        tempowave_config = inject_tempowave_embeddings(
            model,
            tokenizer,
            wavelet_types=args.wavelet_types,
            scales=args.scales,
            grid_resolution=args.grid_resolution,
            alignment=args.alignment,
            projection_seed=args.projection_seed,
        )
    else:
        model.resize_token_embeddings(len(tokenizer))
        tempowave_config = {
            "digit_tokens": tokenizer.convert_ids_to_tokens(
                tokenizer.convert_tokens_to_ids(
                    [f"<|digit_{digit}|>" for digit in range(10)]
                )
            ),
            "embedding_model": "raw",
        }

    if not args.train_language_model:
        model.save_pretrained_merged(args.output_path, tokenizer)
        save_tempowave_config(args.output_path, tempowave_config)
        return

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["lm_head"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_seed,
        max_seq_length=args.max_seq_length,
    )

    def formatting_func(example):
        if isinstance(example["text"], list):
            return [text + tokenizer.eos_token for text in example["text"]]
        return example["text"] + tokenizer.eos_token

    dataset = load_text_dataset(
        "csv",
        args.dataset_path,
        split=args.dataset_split,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        callbacks=[MergeAndSaveCallback(model, tokenizer, args.checkpoint_dir)],
        args=build_training_arguments(args),
    )

    start_gpu_memory, max_memory = print_gpu_start()
    trainer_stats = trainer.train()
    print_training_summary(trainer_stats, start_gpu_memory, max_memory)

    model.save_pretrained_merged(args.output_path, tokenizer)
    save_tempowave_config(args.output_path, tempowave_config)


if __name__ == "__main__":
    main()
