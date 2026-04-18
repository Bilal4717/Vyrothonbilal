"""
scripts/train.py
================
Fine-tunes Gemma-3-270M-IT with QLoRA on the Pocket-Agent tool-call dataset.

Usage (Colab T4):
    python scripts/train.py \
        --data data/train.jsonl \
        --base_model google/gemma-3-270m-it \
        --adapter_dir artifacts/lora_adapter \
        --epochs 5

Outputs:
    artifacts/lora_adapter/   — LoRA adapter weights + tokenizer
    artifacts/training_loss.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        default="data/train.jsonl")
    parser.add_argument("--base_model",  default="google/gemma-3-270m-it")
    parser.add_argument("--adapter_dir", default="artifacts/lora_adapter")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--max_length",  type=int,   default=512)
    parser.add_argument("--lora_r",      type=int,   default=16)
    parser.add_argument("--lora_alpha",  type=int,   default=32)
    parser.add_argument("--hf_token",    default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    return parser.parse_args()


def load_model_and_tokenizer(base_model: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=hf_token)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        token=hf_token,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def apply_lora(model, r: int, alpha: int):
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def plot_loss(log_history, out_path: str):
    sns.set_theme()
    train = [(l["epoch"], l["loss"])      for l in log_history if "loss"      in l]
    val   = [(l["epoch"], l["eval_loss"]) for l in log_history if "eval_loss" in l]
    fig, ax = plt.subplots(figsize=(8, 4))
    if train: ax.plot(*zip(*train), label="Training Loss")
    if val:   ax.plot(*zip(*val),   label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("QLoRA Fine-Tuning — Pocket-Agent Tool Calls")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Loss plot saved to {out_path}")


def main():
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    os.makedirs(args.adapter_dir,    exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("artifacts",         exist_ok=True)

    print(f"Loading model: {args.base_model}")
    model, tokenizer = load_model_and_tokenizer(args.base_model, hf_token)
    model = apply_lora(model, args.lora_r, args.lora_alpha)

    print(f"Loading dataset: {args.data}")
    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(dataset['train'])} | Val: {len(dataset['test'])}")

    sft_args = SFTConfig(
        output_dir=args.checkpoint_dir,
        max_length=args.max_length,
        packing=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=args.lr,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        push_to_hub=False,
        report_to="tensorboard",
        dataset_kwargs={
            "add_special_tokens":  False,
            "append_concat_token": True,
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to: {args.adapter_dir}")
    trainer.model.save_pretrained(args.adapter_dir)
    tokenizer.save_pretrained(args.adapter_dir)

    plot_loss(trainer.state.log_history, "artifacts/training_loss.png")

    # Print adapter size
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(args.adapter_dir)
        for f in fns
    )
    print(f"Adapter size: {total / 1024**2:.1f} MB")
    print("Training complete.")


if __name__ == "__main__":
    main()
