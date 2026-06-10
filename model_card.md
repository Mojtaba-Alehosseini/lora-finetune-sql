# Model Card: lora-finetune-sql (Qwen2.5-0.5B-Instruct + LoRA)

## Model Details

- **Base model:** Qwen/Qwen2.5-0.5B-Instruct
- **Fine-tuning method:** LoRA (PEFT), r=16, alpha=32, target: q/k/v/o_proj
- **Task:** Natural Language → SQL (NL-to-SQL)
- **Domain:** Retail database (products, customers, orders, order_items)
- **Adapters:** `adapters/` (local) — see `src/train_lora.py`
- **Larger Colab version:** Qwen2.5-7B-Instruct + Unsloth QLoRA — `notebooks/train_unsloth_qlora.ipynb`

## Intended Use

Translate natural language questions into SQL queries for a known retail schema:

```
products(product_id, name, category, price, stock_qty)
customers(customer_id, name, email, city, country, joined_date)
orders(order_id, customer_id, order_date, status)
order_items(item_id, order_id, product_id, quantity, unit_price)
```

**Intended users:** Developers building BI/analytics assistants on retail data.

**Out-of-scope:** Arbitrary schemas, DML queries (INSERT/UPDATE/DELETE), production use without review.

## Training Data

- **Source:** Synthetically generated question-SQL pairs (see `data/build_dataset.py`)
- **Size:** 64 train / 40 heldout pairs
- **Validation:** Every gold SQL was executed against `data/sample.sqlite` before use; zero fabricated rows
- **Seed:** 42 (fully reproducible)
- **Schema coverage:** SELECT, WHERE, GROUP BY/HAVING, JOIN, subquery, CASE/WHEN, date functions, window-less aggregations

## Evaluation

**Metric:** Execution accuracy — does the generated SQL return the same rows as the gold SQL?
Row-order differences are ignored (comparison via frozensets).

See `eval/results.json` for committed results (written by `src/eval_compare.py`).

| Model | Execution Accuracy | Exec Errors |
|---|---|---|
| Qwen2.5-0.5B-Instruct (base) | See eval/results.json | — |
| + LoRA fine-tuned (3 epochs) | See eval/results.json | — |

*Real numbers only — run `python src/eval_compare.py` to reproduce.*

## Limitations

- Trained on a single fixed schema; does not generalise to arbitrary databases without re-fine-tuning
- Small 0.5B model: base SQL capability is limited; improvement from fine-tuning on domain-specific data
- Does not generate DML (INSERT/UPDATE/DELETE) — by design
- Not reviewed for adversarial SQL injection; never expose to untrusted user input without a query allowlist
- Evaluation set is small (40 examples); results have high variance

## Training Procedure

```bash
python data/build_dataset.py   # build sample.sqlite + train/heldout JSONL
python src/train_lora.py       # LoRA fine-tuning (CPU, ~30-60 min)
python src/eval_compare.py     # before/after execution accuracy
```

Hyperparameters logged at the top of `src/train_lora.py`.

## Reproducing Results

All code, data-building scripts, and the sample database are in this repository.
The dataset is fully deterministic at seed=42. Run `pytest -q` to verify all SQL.

## Ethical Considerations

- Training data is fully synthetic; no personal data was used
- SQL injection risk: never pipe generated SQL directly to a production database without validation
- The model can generate syntactically valid but logically incorrect queries — always review before execution
