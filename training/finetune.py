import argparse

import torch
from trl import SFTTrainer
from unsloth import FastLanguageModel

from training.common import (
    MergeAndSaveCallback,
    add_common_training_args,
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
    add_common_training_args(parser, include_val=True)
    parser.add_argument(
        "--preview_dataset_path",
        type=str,
        default=None,
        help="Optional response-free JSON dataset for post-training generation previews",
    )
    parser.add_argument("--preview_examples", type=int, default=3, help="Number of validation examples to generate after training. Use 0 to skip.")
    args = parse_args_with_config(parser)
    ensure_output_dirs(args.checkpoint_dir, args.output_path)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    validate_digit_tokenizer(tokenizer)
    freeze_tempowave_input_embeddings(model)
    print(f"\nVocabulary number: {len(tokenizer.get_vocab())}\n")


    # add lora to llama model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", ],
        modules_to_save=["lm_head"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_seed,
        max_seq_length=args.max_seq_length,
    )


    print(f"\nLoading dataset in {args.dataset_path} and {args.val_dataset_path}")
    tokenized_datasets = load_text_dataset("json", args.dataset_path)
    tokenized_val_datasets = load_text_dataset("json", args.val_dataset_path)
    preview_dataset = (
        load_text_dataset("json", args.preview_dataset_path)
        if args.preview_dataset_path
        else None
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_val_datasets,
        callbacks=[MergeAndSaveCallback(model, tokenizer, args.checkpoint_dir)],
        args=build_training_arguments(args, evaluate=True),
    )

    start_gpu_memory, max_memory = print_gpu_start()

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        tokenizer=tokenizer,
        instruction_part = "<|im_start|>user\n",
        response_part = "<|im_start|>assistant\n",
    )
    trainer_stats = trainer.train()
    print_training_summary(trainer_stats, start_gpu_memory, max_memory)

    # Test the model with validation dataset examples
    if args.preview_examples and preview_dataset is not None:
        print("\nGenerating response-free preview examples:")
        print("-" * 50)
        model.eval()

        for i in range(min(args.preview_examples, len(preview_dataset))):
            example = preview_dataset[i]
        
            inputs = tokenizer(example["text"], return_tensors="pt").to(model.device)
        
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )
        
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\nExample {i+1}:")
            print(f"Input: {example['text']}")
            print(f"Model Output: {generated_text}")
            print("-" * 50)

    # save model and tokenizer
    model.save_pretrained_merged(args.output_path, tokenizer)
    copy_tempowave_config(args.model_path, args.output_path)
    
    # Save final checkpoint
    trainer.save_model(args.checkpoint_dir + "/final")
