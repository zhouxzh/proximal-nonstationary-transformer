# Proximal Nonstationary Transformer

Public code for **window-level proximal statistical anchors** on the Nonstationary Transformer (LastVal / Mean / Median).

This repository can **rebuild the paper result table and figures from saved `results/` files**. It does not retrain the 31,500 configurations. Checkpoints are not included.

Paper: *Enhancing Transformer-Based Nonstationary Gold Price Series Forecasting with Proximal Statistical Anchors*.

## What is in this repository

| Path | Role |
| --- | --- |
| `run.py`, `models/`, `exp/`, `layers/` | Training / evaluation code (TSLib-based) |
| `dataset/GLD.csv` | Daily GLD close used in the paper |
| `results/` | Saved `metrics.npy` / `pred.npy` / `true.npy` (**not in Git**; keep locally) |
| `paper/All_results_metrics_sorted_of_JOF.xlsx` | Paper table (7 models × 4500 rows) |
| `paper/reconstruct_jof_workbook.py` | Rebuild the table from `results/` |
| `paper/NST_Paper_Code.py` | Draw paper figures from the table |

Paper models: Nonstationary Transformer, NST-LastVal, NST-Mean, NST-Median, Transformer, Autoformer, Crossformer.

## Reproduce the paper table

Requires a local `results/` tree with 31,500 folders (already present on the ace2 working copy).

```bash
pip install -r paper/requirements-paper.txt

python paper/reconstruct_jof_workbook.py \
  --results-root results \
  --out paper/All_results_metrics_sorted_of_JOF_rebuilt.xlsx \
  --compare paper/All_results_metrics_sorted_of_JOF.xlsx
```

A matching rebuild prints zero key mismatches and metric differences at floating-point noise (~1e-15).

## Draw the paper figures

```bash
python paper/NST_Paper_Code.py
```

Outputs:

- `paper/output_nst_paper_analysis/tables/`
- `paper/output_nst_paper_analysis/figures_main/`
- `paper/output_nst_paper_analysis/figures_appendix/`

## Optional: train one configuration

```bash
python run.py \
  --model Proximal_Nonstationary_Transformer \
  --wnorm LastVal \
  --data custom \
  --root_path ./dataset/ \
  --data_path GLD.csv \
  --features S \
  --target Close \
  --seq_len 96 \
  --pred_len 24 \
  --enc_in 1 \
  --c_out 1 \
  --itr 1
```

`--wnorm` is `LastVal`, `Mean`, or `Median`. Training is **not** required to rebuild the published table.

## License

See `LICENSE`.
