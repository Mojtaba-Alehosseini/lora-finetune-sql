"""
Compare base model vs LoRA fine-tuned model on execution accuracy (NL-to-SQL).

Execution accuracy: does the generated SQL return the same rows as the gold SQL?
Tolerates row-order differences (sets of frozensets).

Usage:
    # after training:
    python src/eval_compare.py --adapter-dir adapters/

    # to evaluate only the base model (no adapters):
    python src/eval_compare.py --base-only
"""

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "sample.sqlite"
HELDOUT_PATH = ROOT / "data" / "heldout.jsonl"
RESULTS_PATH = ROOT / "eval" / "results.json"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 96  # SQL queries are typically 15-80 tokens
SEED = 42


def make_prompt(question: str, schema: str) -> str:
    # Must match train_lora.py prompt format exactly
    return (
        f"### Schema\n{schema}\n\n"
        f"### Question\n{question}\n\n"
        "### SQL\n"
    )


def extract_sql(text: str) -> str:
    """Strip prompt prefix and clean up generated SQL."""
    # remove prompt markers (compact format and legacy)
    for marker in ("### SQL\n", "### SQL", "SQL:\n", "SQL:"):
        if marker in text:
            text = text.split(marker)[-1]
            break
    # strip markdown fences
    text = re.sub(r"```sql", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    # take first statement only
    stmt = text.strip().split(";")[0].strip()
    return stmt + ";" if stmt and not stmt.endswith(";") else stmt


def exec_sql(conn: sqlite3.Connection, sql: str):
    """Execute SQL; return frozenset of rows or None on error."""
    try:
        rows = conn.execute(sql).fetchall()
        return frozenset(frozenset(r) if isinstance(r, (list, tuple)) else r for r in rows)
    except Exception:
        return None


def rows_match(gold_rows, pred_rows) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return gold_rows == pred_rows


def load_heldout():
    return [json.loads(line) for line in HELDOUT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_model(model, tokenizer, items, conn, label="model"):
    correct = 0
    exec_errors = 0
    details = []
    for i, item in enumerate(items):
        prompt = make_prompt(item["question"], item["schema"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_sql = extract_sql(generated)

        gold_rows = exec_sql(conn, item["sql"])
        pred_rows = exec_sql(conn, pred_sql)
        match = rows_match(gold_rows, pred_rows)

        if match:
            correct += 1
        elif pred_rows is None:
            exec_errors += 1

        details.append({
            "question": item["question"],
            "gold_sql": item["sql"],
            "pred_sql": pred_sql,
            "match": match,
            "exec_error": pred_rows is None,
        })

        if (i + 1) % 5 == 0:
            acc = correct / (i + 1)
            print(f"  [{label}] {i+1}/{len(items)} — acc so far: {acc:.1%}")

    n = len(items)
    accuracy = correct / n
    print(f"[{label}] Execution accuracy: {correct}/{n} = {accuracy:.1%}  "
          f"(exec errors: {exec_errors})")
    return {"accuracy": round(accuracy, 4), "correct": correct, "total": n,
            "exec_errors": exec_errors, "details": details}


def main(args):
    items = load_heldout()
    conn = sqlite3.connect(DB_PATH)

    torch.manual_seed(SEED)

    print(f"Loading tokenizer from {MODEL_NAME}…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model ({MODEL_NAME})…")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, trust_remote_code=True, dtype=torch.float32
    )
    base_model.eval()

    print(f"\nEvaluating base model on {len(items)} heldout examples…")
    base_result = evaluate_model(base_model, tokenizer, items, conn, label="base")

    ft_result = None
    if not args.base_only:
        adapter_dir = Path(args.adapter_dir)
        if not adapter_dir.exists():
            print(f"Adapter dir not found: {adapter_dir}. Run src/train_lora.py first.")
            sys.exit(1)

        from peft import PeftModel
        print(f"\nLoading fine-tuned model (base + adapters from {adapter_dir})…")
        ft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        ft_model.eval()

        print(f"\nEvaluating fine-tuned model on {len(items)} heldout examples…")
        ft_result = evaluate_model(ft_model, tokenizer, items, conn, label="finetuned")

    conn.close()

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    results = {
        "model": MODEL_NAME,
        "heldout_n": len(items),
        "base": {k: v for k, v in base_result.items() if k != "details"},
        "finetuned": {k: v for k, v in ft_result.items() if k != "details"} if ft_result else None,
        "delta_accuracy": (
            round(ft_result["accuracy"] - base_result["accuracy"], 4)
            if ft_result else None
        ),
        "details_base": base_result["details"],
        "details_finetuned": ft_result["details"] if ft_result else None,
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH}")

    print("\n=== SUMMARY ===")
    print(f"Base model execution accuracy:  {base_result['accuracy']:.1%}")
    if ft_result:
        delta = results["delta_accuracy"]
        sign = "+" if delta >= 0 else ""
        print(f"Fine-tuned execution accuracy:  {ft_result['accuracy']:.1%}  "
              f"(delta: {sign}{delta:.1%})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", default="adapters/")
    p.add_argument("--base-only", action="store_true")
    main(p.parse_args())
