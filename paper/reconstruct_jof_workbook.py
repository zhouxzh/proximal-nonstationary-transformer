#!/usr/bin/env python3
"""Rebuild All_results_metrics_sorted_of_JOF.xlsx from Time-Series-Library results/ folders.

Each folder is expected to contain metrics.npy = [mae, mse, rmse, mape, mspe]
computed on the full pred/true tensors (not last-step only), which is the
TSLib utils.metrics.metric() convention.

This script does not train models. It only aggregates already-saved evaluation
arrays into the paper workbook schema.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_MODELS = [
    "Nonstationary_Transformer",
    "Proximal_Nonstationary_Transformer_LastVal",
    "Proximal_Nonstationary_Transformer_Mean",
    "Proximal_Nonstationary_Transformer_Median",
    "Transformer",
    "Autoformer",
    "Crossformer",
]

FOLDER_RE = re.compile(
    r"""
    ^
    (?P<task_name>long_term_forecast)_
    (?P<model>.+?)_
    (?P<dataset>GLD)_
    (?P<raw_model>.+?)_
    custom_
    ft(?P<ft>[^_]+)_
    sl(?P<sl>\d+)_
    ll(?P<ll>\d+)_
    pl(?P<pl>\d+)_
    dm(?P<dm>\d+)_
    nh(?P<nh>\d+)_
    el(?P<el>\d+)_
    dl(?P<dl>\d+)_
    df(?P<df>\d+)_
    fc(?P<fc>\d+)_
    eb(?P<eb>[^_]+)_
    dt(?P<dt>[^_]+)_
    Exp_
    (?P<exp>\d+)
    $
    """,
    re.VERBOSE,
)

COLUMNS = [
    "result_id",
    "model",
    "raw_model",
    "dataset",
    "task_name",
    "setting_name",
    "sl",
    "ll",
    "pl",
    "dm",
    "nh",
    "el",
    "dl",
    "df",
    "fc",
    "ft",
    "eb",
    "dt",
    "exp",
    "mae",
    "mse",
    "rmse",
    "mape",
    "mspe",
    "pred_shape",
    "true_shape",
]


def parse_folder(name: str) -> dict | None:
    m = FOLDER_RE.match(name)
    if not m:
        return None
    rec = m.groupdict()
    rec["setting_name"] = name
    rec["dt"] = rec["dt"] == "True"
    rec["exp"] = int(rec["exp"])
    for key in ("sl", "ll", "pl", "dm", "nh", "el", "dl", "df", "fc"):
        rec[key] = int(rec[key])
    return rec


def shape_str(arr: np.ndarray) -> str:
    return str(tuple(int(x) for x in arr.shape))


def collect_rows(results_root: Path, load_shapes: bool = True) -> pd.DataFrame:
    rows = []
    missing = []
    unmatched = 0
    for path in sorted(results_root.iterdir()):
        if not path.is_dir():
            continue
        rec = parse_folder(path.name)
        if rec is None:
            unmatched += 1
            continue
        if rec["model"] not in PAPER_MODELS:
            continue
        metrics_path = path / "metrics.npy"
        if not metrics_path.exists():
            missing.append(path.name)
            continue
        metrics = np.load(metrics_path).astype(np.float64).reshape(-1)
        if metrics.size < 5:
            raise ValueError(f"{path.name}: metrics.npy has {metrics.size} values, expected 5")
        rec["mae"], rec["mse"], rec["rmse"], rec["mape"], rec["mspe"] = metrics[:5]
        if load_shapes:
            pred = np.load(path / "pred.npy", mmap_mode="r")
            true = np.load(path / "true.npy", mmap_mode="r")
            rec["pred_shape"] = shape_str(pred)
            rec["true_shape"] = shape_str(true)
        else:
            rec["pred_shape"] = ""
            rec["true_shape"] = ""
        rows.append(rec)
    if missing:
        raise FileNotFoundError(f"missing metrics.npy for {len(missing)} folders, e.g. {missing[:3]}")
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no paper-model folders under {results_root} (unmatched={unmatched})")
    df = df.sort_values(["model", "sl", "pl", "setting_name"], kind="mergesort").reset_index(drop=True)
    df.insert(0, "result_id", np.arange(1, len(df) + 1))
    return df[COLUMNS]


def compare_to_workbook(rebuilt: pd.DataFrame, workbook: Path, atol: float = 1e-7) -> dict:
    src = pd.read_excel(workbook, sheet_name="all_metrics")
    src.columns = [str(c).strip() for c in src.columns]
    key = ["model", "sl", "pl"]
    merged = src.merge(rebuilt, on=key, how="outer", suffixes=("_wb", "_rb"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    report = {
        "workbook_rows": int(len(src)),
        "rebuilt_rows": int(len(rebuilt)),
        "matched_keys": int(len(both)),
        "only_workbook": int((merged["_merge"] == "left_only").sum()),
        "only_rebuilt": int((merged["_merge"] == "right_only").sum()),
        "metric_max_abs_diff": {},
        "setting_name_mismatches": 0,
        "shape_mismatches": 0,
    }
    for metric in ("mae", "mse", "rmse", "mape", "mspe"):
        diff = (both[f"{metric}_wb"] - both[f"{metric}_rb"]).abs()
        report["metric_max_abs_diff"][metric] = float(diff.max()) if len(diff) else None
        report[f"{metric}_gt_atol"] = int((diff > atol).sum()) if len(diff) else 0
    if "setting_name_wb" in both.columns:
        report["setting_name_mismatches"] = int((both["setting_name_wb"] != both["setting_name_rb"]).sum())
    if "pred_shape_wb" in both.columns and both["pred_shape_rb"].ne("").any():
        report["shape_mismatches"] = int(
            ((both["pred_shape_wb"] != both["pred_shape_rb"]) | (both["true_shape_wb"] != both["true_shape_rb"])).sum()
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path, default=None)
    parser.add_argument("--skip-shapes", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-7)
    args = parser.parse_args()

    df = collect_rows(args.results_root, load_shapes=not args.skip_shapes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(args.out, sheet_name="all_metrics", index=False)
    print(f"wrote {args.out} rows={len(df)}")
    print(df.groupby("model").size().to_string())
    if args.compare:
        report = compare_to_workbook(df, args.compare, atol=args.atol)
        print("COMPARE", report)
        bad = (
            report["only_workbook"]
            or report["only_rebuilt"]
            or report["setting_name_mismatches"]
            or any(report[f"{m}_gt_atol"] for m in ("mae", "mse", "rmse", "mape", "mspe"))
        )
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
