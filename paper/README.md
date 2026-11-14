# Paper table and figures

This directory reconstructs the paper workbook from saved `results/*/metrics.npy`
files and draws the paper figures. It does **not** retrain models.

## Reconstruct the workbook from `results/`

From the repository root:

```bash
python paper/reconstruct_jof_workbook.py \
  --results-root results \
  --out paper/All_results_metrics_sorted_of_JOF.xlsx \
  --compare paper/All_results_metrics_sorted_of_JOF.xlsx
```

The committed workbook is the paper table (31,500 rows, seven models).
`--compare` checks that a rebuilt table matches it.

## Draw the paper figures

```bash
python paper/NST_Paper_Code.py
```

Outputs go to `paper/output_nst_paper_analysis/tables/` and
`paper/output_nst_paper_analysis/figures_main/`.
