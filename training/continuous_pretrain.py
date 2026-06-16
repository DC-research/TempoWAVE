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
    copy_tempowave_config,
    freeze_tempowave_input_embeddings,
)
from utils.config import parse_args_with_config
from utils.tools import validate_digit_tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_training_args(parser)
    add_csv_dataset_args(parser)
    args = parse_args_with_config(parser)
    ensure_output_dirs(args.checkpoint_dir, args.output_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    validate_digit_tokenizer(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    model, _ = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    freeze_tempowave_input_embeddings(model)

    # Add LoRA to model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["lm_head"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_seed,
        max_seq_length=args.max_seq_length,
    )

    # Load dataset
    def formatting_func(example):
        if isinstance(example["text"], list):
            return [text + tokenizer.eos_token for text in example["text"]]
        return example["text"] + tokenizer.eos_token

    print(f"\nLoading dataset in {args.dataset_path}")
    dataset = load_text_dataset("csv", args.dataset_path, split=args.dataset_split)

    # Train model
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        callbacks=[MergeAndSaveCallback(model, tokenizer, args.checkpoint_dir)],
        args=build_training_arguments(args, save_total_limit=5),
    )

    start_gpu_memory, max_memory = print_gpu_start()
    trainer_stats = trainer.train()
    print_training_summary(trainer_stats, start_gpu_memory, max_memory)

    # Save final model and tokenizer
    model.save_pretrained_merged(args.output_path, tokenizer)
    copy_tempowave_config(args.model_path, args.output_path)
