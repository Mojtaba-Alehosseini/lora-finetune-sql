"""
LoRA fine-tuning for NL-to-SQL using peft + transformers.

Hyperparameters (all at the top for reproducibility):
  model_name    : Qwen/Qwen2.5-0.5B-Instruct  (CPU-feasible at 0.5B params)
  lora_r        : 16
  lora_alpha    : 32
  lora_dropout  : 0.05
  target_modules: q_proj, k_proj, v_proj, o_proj
  lr            : 2e-4
  epochs        : 3
  batch_size    : 4
  max_length    : 256
  seed          : 42

Usage:
    python src/train_lora.py
    python src/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset

# ── hyperparameters ──────────────────────────────────────────────────────── #
DEFAULTS = {
    "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "lr": 2e-4,
    "epochs": 3,
    "batch_size": 4,
    "max_length": 256,
    "seed": 42,
}
# ─────────────────────────────────────────────────────────────────────────── #

ROOT = Path(__file__).parent.parent
TRAIN_PATH = ROOT / "data" / "train.jsonl"
ADAPTER_DIR = ROOT / "adapters"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_prompt(question: str, schema: str) -> str:
    return (
        "You are an expert SQL generator. "
        "Given a database schema and a natural language question, "
        "write a single valid SQL query that answers the question. "
        "Return only the SQL query, nothing else.\n\n"
        f"Schema: {schema}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )


class NLSQLDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.samples = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = make_prompt(item["question"], item["schema"])
        full_text = prompt + " " + item["sql"]

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # labels: -100 for prompt tokens (only compute loss on SQL part)
        prompt_enc = self.tokenizer(
            prompt, max_length=self.max_length, truncation=True, return_tensors="pt"
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train(args):
    set_seed(args.seed)
    print(f"Model: {args.model_name}")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}, epochs={args.epochs}, lr={args.lr}")

    print("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model…")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, dtype=torch.float32
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = NLSQLDataset(TRAIN_PATH, tokenizer, args.max_length)
    print(f"Training on {len(dataset)} examples")

    ADAPTER_DIR.mkdir(exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        fp16=False,
        bf16=False,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="none",
        use_cpu=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    trainer.train()

    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"Adapters saved to {ADAPTER_DIR}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", dest="model_name", default=DEFAULTS["model_name"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--batch-size", dest="batch_size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--max-length", dest="max_length", type=int, default=DEFAULTS["max_length"])
    p.add_argument("--lora-r", dest="lora_r", type=int, default=DEFAULTS["lora_r"])
    p.add_argument("--lora-alpha", dest="lora_alpha", type=int, default=DEFAULTS["lora_alpha"])
    p.add_argument("--lora-dropout", dest="lora_dropout", type=float,
                   default=DEFAULTS["lora_dropout"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = p.parse_args()
    args.target_modules = DEFAULTS["target_modules"]
    return args


if __name__ == "__main__":
    args = parse_args()
    train(args)
    sys.exit(0)
