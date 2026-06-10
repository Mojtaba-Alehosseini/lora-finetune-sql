# Project: LoRA/QLoRA fine-tuning + before/after eval (NL→SQL specialisation)

## What this is
Fine-tune a small instruct model (Qwen2.5-0.5B-Instruct) with LoRA (via peft/transformers)
to improve NL→SQL on a known retail schema, evaluate base vs fine-tuned on a held-out set,
and report execution accuracy. Push adapters + model card to HF Hub.

For a larger model (Qwen2.5-7B-Instruct): see notebooks/train_unsloth_qlora.ipynb (Colab).

## Stack
Python 3.11 · transformers · peft · datasets · torch · SQLite · pytest · ruff.
HF Hub for model hosting.

## Commands
- Build data: `python data/build_dataset.py`
- Fine-tune (small model, CPU): `python src/train_lora.py`
- Evaluate before/after: `python src/eval_compare.py`
- Tests: `pytest -q`
- Lint: `ruff check .`

## Conventions
- ALL gold SQL in the dataset must execute against data/sample.sqlite — validated on build.
- Seed: 42. Log hyperparameters at the top of train_lora.py.
- Report honestly: show where the model improved and where it didn't.
- Token in .env only (not committed).

## Done per step
Verify command passes → commit.
