import os

import torch
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig

class MergeAndSaveCallback(TrainerCallback):
    def __init__(self, model, tokenizer, checkpoint_dir):
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir

    def on_save(self, args, state, control, **kwargs):
        merged_checkpoint_dir = os.path.join(
            self.checkpoint_dir,
            f"checkpoint-{state.global_step}-merged",
        )
        os.makedirs(merged_checkpoint_dir, exist_ok=True)
        self.model.save_pretrained_merged(merged_checkpoint_dir, self.tokenizer)
        return control


def add_common_training_args(parser, include_val=False):
    parser.add_argument("--model_path", type=str, required=True, default=None)
    parser.add_argument("--dataset_path", type=str, required=True, default=None)
    if include_val:
        parser.add_argument("--val_dataset_path", type=str, required=True, default=None)
    parser.add_argument("--output_path", type=str, required=True, default=None)
    parser.add_argument("--checkpoint_dir", type=str, required=True, default=None)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--dataset_num_proc", type=int, default=8)

    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.00)
    parser.add_argument("--random_seed", type=int, default=3407)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--save_steps", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)


def add_csv_dataset_args(parser):
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Dataset split to load, e.g. train or train[:1000] for smoke tests",
    )


def ensure_output_dirs(*paths):
    for path in paths:
        if path:
            os.makedirs(path, exist_ok=True)


def load_text_dataset(kind, dataset_path, split="train", cache_dir=None):
    if kind == "csv":
        dataset = load_dataset(
            "csv",
            data_files=dataset_path,
            split=split,
            cache_dir=cache_dir or f"{dataset_path}_cache",
        )
    elif kind == "json":
        dataset = load_dataset("json", data_files=dataset_path, split=split)
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

    if "text" not in dataset.column_names:
        raise ValueError(
            f"Dataset {dataset_path} must contain a 'text' column. "
            f"Found columns: {dataset.column_names}"
        )
    return dataset


def build_training_arguments(args, save_total_limit=None, evaluate=False):
    from unsloth import is_bfloat16_supported

    kwargs = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": 1.0,
        "learning_rate": args.learning_rate,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "max_steps": args.max_steps,
        "logging_first_step": True,
        "optim": "adamw_8bit",
        "lr_scheduler_type": "cosine",
        "seed": args.random_seed,
        "output_dir": args.checkpoint_dir,
        "fp16": not is_bfloat16_supported(),
        "bf16": is_bfloat16_supported(),
        "report_to": "none",
        "dataset_text_field": "text",
        "dataset_num_proc": args.dataset_num_proc,
        "max_seq_length": args.max_seq_length,
        "packing": False,
    }
    if save_total_limit is not None:
        kwargs["save_total_limit"] = save_total_limit
    if evaluate:
        kwargs["eval_strategy"] = "steps"
        kwargs["eval_steps"] = args.logging_steps
    return SFTConfig(**kwargs)


def print_gpu_start():
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"\nGPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.\n")
    return start_gpu_memory, max_memory


def print_training_summary(trainer_stats, start_gpu_memory, max_memory):
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"\n{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.\n")
