"""
Generate paper-ready statistics and figures for the Nonstationary-Transformer SCI paper.

Input:
    All_results_metrics_sorted_of_JOF.xlsx, sheet: all_metrics (existing evaluation results)

Target models:
    Nonstationary_Transformer
    Proximal_Nonstationary_Transformer_LastVal
    Proximal_Nonstationary_Transformer_Mean
    Proximal_Nonstationary_Transformer_Median
    Transformer
    Autoformer
    Crossformer

Main outputs:
    output_nst_paper_analysis/tables/*.csv
    output_nst_paper_analysis/tables/summary_tables.xlsx
    output_nst_paper_analysis/figures_main/*.png / *.pdf
    output_nst_paper_analysis/figures_appendix/*.png / *.pdf

Notes:
    This script summarizes existing evaluation rows; it does not train or reproduce models.
    MAE, MSE and RMSE are primary metrics in standardized space. MAPE/MSPE
    remain in legacy supplementary exports, but standardized targets may cross
    zero: these values are not interpretable as original-scale percentage errors.
    Source fc is checked against the actual workbook value 3.
    Overall values are unweighted arithmetic means over 9 input windows x 500
    horizons (4500 settings per model); horizon-group means use all windows and
    all horizons in each inclusive group. Std is sample standard deviation
    across those settings; paired improvement is the mean of per-setting
    percentage changes, not the percentage change of aggregate means.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# =============================================================================
# 0. Global configuration
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
EXCEL_PATH = SCRIPT_DIR / "All_results_metrics_sorted_of_JOF.xlsx"
SHEET_NAME = "all_metrics"
OUTPUT_DIR = SCRIPT_DIR / "output_nst_paper_analysis"

# Optional: set this to your local results folder that contains pred.npy and true.npy.
# Example on Windows: RESULTS_ROOT = Path(r"D:\your_project\results")
RESULTS_ROOT: Optional[Path] = None

# Paper setting
PRIMARY_METRICS = ["mae", "mse", "rmse"]
SUPPLEMENTARY_METRICS = ["mape", "mspe"]
ALL_METRICS = PRIMARY_METRICS + SUPPLEMENTARY_METRICS
EXPECTED_FC = 3
RMSE_ATOL = 1e-6

MODEL_ORDER = [
    "Nonstationary_Transformer",
    "Proximal_Nonstationary_Transformer_LastVal",
    "Proximal_Nonstationary_Transformer_Mean",
    "Proximal_Nonstationary_Transformer_Median",
    "Transformer",
    "Autoformer",
    "Crossformer",
]

MODEL_LABELS = {
    "Nonstationary_Transformer": "NST",
    "Proximal_Nonstationary_Transformer_LastVal": "NST-LastVal",
    "Proximal_Nonstationary_Transformer_Mean": "NST-Mean",
    "Proximal_Nonstationary_Transformer_Median": "NST-Median",
    "Transformer": "Transformer",
    "Autoformer": "Autoformer",
    "Crossformer": "Crossformer",
}

NST_VARIANT_MODELS = [
    "Nonstationary_Transformer",
    "Proximal_Nonstationary_Transformer_LastVal",
    "Proximal_Nonstationary_Transformer_Mean",
    "Proximal_Nonstationary_Transformer_Median",
]
NST_VARIANT_LABELS = [MODEL_LABELS[m] for m in NST_VARIANT_MODELS]

SEQ_LEN_ORDER = [16, 32, 64, 96, 128, 192, 256, 320, 384]

# Horizon protocol. You may adjust these intervals, but keep them fixed in the manuscript once chosen.
HORIZON_GROUPS = [
    ("Short", 1, 50),
    ("Medium", 51, 200),
    ("Long", 201, 500),
]

HEATMAP_BUCKETS = [
    ("1-50", 1, 50),
    ("51-100", 51, 100),
    ("101-200", 101, 200),
    ("201-300", 201, 300),
    ("301-400", 301, 400),
    ("401-500", 401, 500),
]

# Plot style. Keep it clean and journal-friendly.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",  # Change to "Times New Roman" if your system supports it.
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.45,
    "lines.linewidth": 1.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =============================================================================
# 1. Utility functions
# =============================================================================

def ensure_dirs() -> Dict[str, Path]:
    paths = {
        "root": OUTPUT_DIR,
        "tables": OUTPUT_DIR / "tables",
        "fig_main": OUTPUT_DIR / "figures_main",
        "fig_appendix": OUTPUT_DIR / "figures_appendix",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> None:
    """Save as both PNG and PDF for manuscript use."""
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def horizon_group(pl: int) -> str:
    for name, lo, hi in HORIZON_GROUPS:
        if lo <= int(pl) <= hi:
            return name
    return "Other"


def heatmap_bucket(pl: int) -> str:
    for name, lo, hi in HEATMAP_BUCKETS:
        if lo <= int(pl) <= hi:
            return name
    return "Other"


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def require_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {df.columns.tolist()}")


def ordered_labels(labels: Iterable[str]) -> List[str]:
    order = [MODEL_LABELS[m] for m in MODEL_ORDER]
    return [x for x in order if x in set(labels)]


# =============================================================================
# 2. Data loading and validation
# =============================================================================

def validate_matrix(df: pd.DataFrame) -> None:
    """Reject any incomplete, duplicated or numerically invalid source matrix."""
    keys = ["model", "sl", "pl"]
    expected = len(MODEL_ORDER) * len(SEQ_LEN_ORDER) * 500
    if len(df) != expected or df["model"].isna().any() or not df["model"].isin(MODEL_ORDER).all():
        raise ValueError(f"Expected exactly {expected} rows for seven known models")
    numeric = {}
    for column in ["sl", "ll", "pl", "fc", *ALL_METRICS]:
        numeric[column] = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(numeric[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Missing or non-finite {column}")
    for column, allowed in [("sl", SEQ_LEN_ORDER), ("pl", range(1, 501)), ("fc", [EXPECTED_FC])]:
        if not numeric[column].isin(allowed).all():
            raise ValueError(f"Unexpected {column}; fc must match the audited workbook value {EXPECTED_FC}")
    if not (numeric["ll"] == numeric["sl"] / 2).all():
        raise ValueError("ll must equal sl/2 for every row")
    if (df[keys].duplicated()).any():
        raise ValueError("Duplicate model/sl/pl setting")
    if (numeric["mae"] < 0).any() or (numeric["mse"] < 0).any() or (numeric["rmse"] < 0).any():
        raise ValueError("Negative primary metric")
    if not np.allclose(numeric["rmse"], np.sqrt(numeric["mse"]), rtol=1e-6, atol=RMSE_ATOL):
        raise ValueError("RMSE disagrees with sqrt(MSE)")
    counts = df.groupby(["model", "sl"], observed=True)["pl"].nunique()
    if len(counts) != len(MODEL_ORDER) * len(SEQ_LEN_ORDER) or not counts.eq(500).all():
        raise ValueError("Incomplete model/sl/pl coverage")


def load_and_prepare_data(excel_path: Path = EXCEL_PATH, sheet_name: str = SHEET_NAME) -> pd.DataFrame:
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = [str(c).strip().lower() for c in df.columns]

    require_columns(df, ["model", "sl", "ll", "pl", "fc", *ALL_METRICS])
    validate_matrix(df)
    df = df.copy()
    for c in ["sl", "ll", "pl", "fc"]:
        df[c] = pd.to_numeric(df[c]).astype(int)
    for c in ALL_METRICS:
        df[c] = pd.to_numeric(df[c])
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["horizon_group"] = df["pl"].apply(horizon_group)
    df["horizon_bucket"] = df["pl"].apply(heatmap_bucket)

    # Ordered categorical labels for tables and figures.
    df["model_label"] = pd.Categorical(
        df["model_label"],
        categories=[MODEL_LABELS[m] for m in MODEL_ORDER],
        ordered=True,
    )
    df["horizon_group"] = pd.Categorical(
        df["horizon_group"],
        categories=[x[0] for x in HORIZON_GROUPS],
        ordered=True,
    )
    df["horizon_bucket"] = pd.Categorical(
        df["horizon_bucket"],
        categories=[x[0] for x in HEATMAP_BUCKETS],
        ordered=True,
    )

    return df


def validate_experiment_coverage(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    coverage = (
        df.groupby(["model", "model_label"], observed=True)
        .agg(
            rows=("pl", "size"),
            n_seq_len=("sl", "nunique"),
            n_horizons=("pl", "nunique"),
            min_pl=("pl", "min"),
            max_pl=("pl", "max"),
            min_sl=("sl", "min"),
            max_sl=("sl", "max"),
        )
        .reset_index()
    )
    if (len(coverage) != len(MODEL_ORDER) or not coverage["rows"].eq(4500).all()
            or not coverage["n_seq_len"].eq(9).all() or not coverage["n_horizons"].eq(500).all()):
        raise ValueError("Coverage summary contradicts the complete matrix")
    coverage.to_csv(out_dir / "model_coverage.csv", index=False)
    return coverage


# =============================================================================
# 3. Statistical tables
# =============================================================================

def build_summary_tables(df: pd.DataFrame, table_dir: Path) -> Dict[str, pd.DataFrame]:
    # Overall ranking by primary metrics.
    overall = (
        df.groupby("model_label", observed=True)[ALL_METRICS]
        .mean()
        .reset_index()
        .sort_values("mae", ascending=True)
    )
    overall.insert(0, "rank_by_mae", np.arange(1, len(overall) + 1))

    # Standard deviation across all 4500 settings, useful for robustness description.
    overall_std = (
        df.groupby("model_label", observed=True)[ALL_METRICS]
        .std()
        .reset_index()
        .rename(columns={m: f"{m}_std" for m in ALL_METRICS})
    )
    overall_full = overall.merge(overall_std, on="model_label", how="left")

    # Short / Medium / Long horizon statistics.
    horizon = (
        df.groupby(["model_label", "horizon_group"], observed=True)[ALL_METRICS]
        .mean()
        .reset_index()
        .sort_values(["model_label", "horizon_group"])
    )
    horizon_wide = horizon.pivot(index="model_label", columns="horizon_group", values=["mae", "rmse"])
    horizon_wide.columns = [f"{metric}_{grp}" for metric, grp in horizon_wide.columns]
    horizon_wide = horizon_wide.reset_index()

    # Robustness statistics: median, IQR, 95th percentile and worst 5% average.
    def q25(x): return np.quantile(x, 0.25)
    def q75(x): return np.quantile(x, 0.75)
    def q95(x): return np.quantile(x, 0.95)
    def worst5_mean(x):
        threshold = np.quantile(x, 0.95)
        return np.mean(x[x >= threshold])

    robustness = (
        df.groupby("model_label", observed=True)
        .agg(
            mae_median=("mae", "median"),
            mae_q25=("mae", q25),
            mae_q75=("mae", q75),
            mae_q95=("mae", q95),
            mae_worst5_mean=("mae", worst5_mean),
            rmse_median=("rmse", "median"),
            rmse_q25=("rmse", q25),
            rmse_q75=("rmse", q75),
            rmse_q95=("rmse", q95),
            rmse_worst5_mean=("rmse", worst5_mean),
        )
        .reset_index()
    )
    robustness["mae_iqr"] = robustness["mae_q75"] - robustness["mae_q25"]
    robustness["rmse_iqr"] = robustness["rmse_q75"] - robustness["rmse_q25"]

    # Improvement ratios for anchor variants relative to original NST.
    improvement = compute_improvement_ratios(df)
    improvement_summary = (
        improvement.groupby("variant_label", observed=True)
        .agg(
            mean_mae_improvement_pct=("mae_improvement_pct", "mean"),
            mean_rmse_improvement_pct=("rmse_improvement_pct", "mean"),
            positive_mae_improvement_ratio=("mae_improvement_pct", lambda x: np.mean(x > 0)),
            positive_rmse_improvement_ratio=("rmse_improvement_pct", lambda x: np.mean(x > 0)),
        )
        .reset_index()
    )

    # Best seq_len distribution.
    best_seq = compute_best_seq_len_distribution(df, metric="mae")

    tables = {
        "overall_ranking": overall_full,
        "horizon_group_statistics": horizon,
        "horizon_group_statistics_wide": horizon_wide,
        "robustness_statistics": robustness,
        "improvement_ratio_summary": improvement_summary,
        "best_seq_len_distribution": best_seq,
    }

    for name, table in tables.items():
        table.to_csv(table_dir / f"{name}.csv", index=False)

    with pd.ExcelWriter(table_dir / "summary_tables.xlsx", engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)

    return tables


def compute_improvement_ratios(df: pd.DataFrame) -> pd.DataFrame:
    base_model = "Nonstationary_Transformer"
    variant_models = [m for m in NST_VARIANT_MODELS if m != base_model]

    base = df[df["model"] == base_model][["sl", "ll", "pl", "mae", "rmse"]].rename(
        columns={"mae": "base_mae", "rmse": "base_rmse"}
    )
    records = []
    for m in variant_models:
        tmp = df[df["model"] == m][["model", "model_label", "sl", "ll", "pl", "mae", "rmse"]].merge(
            base, on=["sl", "ll", "pl"], how="outer", validate="one_to_one", indicator=True
        )
        if len(tmp) != len(SEQ_LEN_ORDER) * 500 or not tmp["_merge"].eq("both").all():
            raise ValueError(f"Unpaired model settings for {m}")
        tmp = tmp.drop(columns="_merge")
        if (tmp[["base_mae", "base_rmse"]] <= 0).any().any():
            raise ValueError("Cannot compute percentage improvement over zero baseline error")
        tmp["mae_improvement_pct"] = (tmp["base_mae"] - tmp["mae"]) / tmp["base_mae"] * 100.0
        tmp["rmse_improvement_pct"] = (tmp["base_rmse"] - tmp["rmse"]) / tmp["base_rmse"] * 100.0
        tmp["variant_label"] = tmp["model_label"].astype(str)
        records.append(tmp)
    return pd.concat(records, ignore_index=True)


def compute_best_seq_len_distribution(df: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    # For each model and each horizon, find the seq_len with the lowest metric.
    idx = df.groupby(["model_label", "pl"], observed=True)[metric].idxmin()
    best = df.loc[idx, ["model_label", "pl", "sl", metric]].copy()
    count = (
        best.groupby(["model_label", "sl"], observed=True)
        .size()
        .reset_index(name="count")
    )
    total = best.groupby("model_label", observed=True)["pl"].nunique().reset_index(name="total_horizons")
    count = count.merge(total, on="model_label", how="left")
    count["proportion"] = count["count"] / count["total_horizons"]
    return count.sort_values(["model_label", "sl"])


# =============================================================================
# 4. Main manuscript figures
# =============================================================================

def fig_overall_ranking(df: pd.DataFrame, out_dir: Path) -> None:
    overall = (
        df.groupby("model_label", observed=True)[["mae", "rmse"]]
        .mean()
        .reset_index()
        .sort_values("mae", ascending=True)
    )
    labels = overall["model_label"].astype(str).tolist()
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    for ax, metric, title in zip(axes, ["mae", "rmse"], ["Average MAE", "Average RMSE"]):
        values = overall[metric].to_numpy()
        ax.barh(y, values, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"{metric.upper()} in standardized space")
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        for i, v in enumerate(values):
            ax.text(v + max(values) * 0.015, i, f"{v:.3f}", va="center", fontsize=8)
    fig.suptitle("Overall forecasting performance ranking", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig03_overall_performance_ranking")


def fig_horizon_grouped_performance(df: pd.DataFrame, out_dir: Path) -> None:
    # Grouped bar charts for horizon ranges. Main text can use NST variants + main baselines.
    selected_labels = ["NST", "NST-LastVal", "NST-Mean", "NST-Median", "Transformer", "Autoformer", "Crossformer"]
    g = (
        df[df["model_label"].astype(str).isin(selected_labels)]
        .groupby(["model_label", "horizon_group"], observed=True)[["mae", "rmse"]]
        .mean()
        .reset_index()
    )
    labels = selected_labels
    horizons = [x[0] for x in HORIZON_GROUPS]
    x = np.arange(len(horizons))
    width = 0.105

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.8), sharex=True)
    for ax, metric in zip(axes, ["mae", "rmse"]):
        for i, lab in enumerate(labels):
            vals = []
            for h in horizons:
                s = g[(g["model_label"].astype(str) == lab) & (g["horizon_group"].astype(str) == h)][metric]
                vals.append(float(s.iloc[0]) if len(s) else np.nan)
            ax.bar(x + (i - (len(labels)-1)/2) * width, vals, width=width, label=lab, edgecolor="black", linewidth=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels(["Short\n1-50", "Medium\n51-200", "Long\n201-500"])
        ax.set_ylabel(f"{metric.upper()} in standardized space")
        ax.set_title(f"{metric.upper()} by horizon range")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Short-, medium-, and long-term forecasting performance", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig04_horizon_grouped_performance")


def fig_horizon_wise_nst_variants(df: pd.DataFrame, out_dir: Path, smooth_window: int = 10) -> None:
    sub = df[df["model"].isin(NST_VARIANT_MODELS)].copy()
    curve = (
        sub.groupby(["model_label", "pl"], observed=True)[["mae", "rmse"]]
        .mean()
        .reset_index()
        .sort_values(["model_label", "pl"])
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    for ax, metric in zip(axes, ["mae", "rmse"]):
        for lab in NST_VARIANT_LABELS:
            s = curve[curve["model_label"].astype(str) == lab].sort_values("pl")
            y = s[metric].rolling(smooth_window, center=True, min_periods=1).mean()
            ax.plot(s["pl"], y, label=lab)
        ax.set_xlabel("Forecasting horizon (pred_len)")
        ax.set_ylabel(f"{metric.upper()} in standardized space")
        ax.set_title(f"Horizon-wise {metric.upper()}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.set_major_locator(MaxNLocator(6))
    axes[0].legend(frameon=False, ncol=1)
    fig.suptitle("Horizon-wise error evolution of Nonstationary Transformer variants", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig05_horizon_wise_nst_variant_curves")


def fig_improvement_ratio(df: pd.DataFrame, out_dir: Path, smooth_window: int = 10) -> None:
    imp = compute_improvement_ratios(df)
    curve = (
        imp.groupby(["variant_label", "pl"], observed=True)[["mae_improvement_pct", "rmse_improvement_pct"]]
        .mean()
        .reset_index()
        .sort_values(["variant_label", "pl"])
    )
    variant_labels = ["NST-LastVal", "NST-Mean", "NST-Median"]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True)
    for ax, metric, title in zip(
        axes,
        ["mae_improvement_pct", "rmse_improvement_pct"],
        ["MAE improvement", "RMSE improvement"],
    ):
        for lab in variant_labels:
            s = curve[curve["variant_label"] == lab].sort_values("pl")
            y = s[metric].rolling(smooth_window, center=True, min_periods=1).mean()
            ax.plot(s["pl"], y, label=lab)
        ax.axhline(0, linestyle="--", color="black", linewidth=0.8)
        ax.set_xlabel("Forecasting horizon (pred_len)")
        ax.set_ylabel("Improvement over NST (%)")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.xaxis.set_major_locator(MaxNLocator(6))
    axes[0].legend(frameon=False)
    fig.suptitle("Improvement ratios of proximal anchor variants over original NST", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir, "Fig06_improvement_ratio_over_nst")


def _plot_heatmap(ax, matrix: pd.DataFrame, title: str, vmin: float, vmax: float, cmap: str = "YlGnBu_r"):
    im = ax.imshow(matrix.to_numpy(), aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Forecasting horizon bucket")
    ax.set_ylabel("seq_len")
    # Annotate values.
    values = matrix.to_numpy()
    threshold = (vmin + vmax) / 2
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.8, color=color)
    return im


def fig_heatmap_seq_horizon_nst(df: pd.DataFrame, out_dir: Path, metric: str = "mae") -> None:
    sub = df[df["model"].isin(NST_VARIANT_MODELS)].copy()
    matrices = {}
    for lab in NST_VARIANT_LABELS:
        s = sub[sub["model_label"].astype(str) == lab]
        mat = s.pivot_table(
            index="sl", columns="horizon_bucket", values=metric, aggfunc="mean", observed=True
        )
        mat = mat.reindex(index=SEQ_LEN_ORDER, columns=[x[0] for x in HEATMAP_BUCKETS])
        matrices[lab] = mat

    vmin = min(float(np.nanmin(m.to_numpy())) for m in matrices.values())
    vmax = max(float(np.nanmax(m.to_numpy())) for m in matrices.values())

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2))
    axes = axes.ravel()
    last_im = None
    for ax, lab in zip(axes, NST_VARIANT_LABELS):
        last_im = _plot_heatmap(ax, matrices[lab], lab, vmin, vmax)
    fig.subplots_adjust(right=0.90, wspace=0.34, hspace=0.42, top=0.90)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.018, 0.68])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label(f"Mean {metric.upper()} in standardized space")
    fig.suptitle(f"{metric.upper()} heatmaps under different input windows and forecasting horizons", y=0.98, fontsize=13)
    save_figure(fig, out_dir, f"Fig07_seq_len_horizon_heatmap_{metric}")


def fig_error_distribution_boxplot(df: pd.DataFrame, out_dir: Path, metric: str = "mae") -> None:
    labels = ordered_labels(df["model_label"].astype(str).unique())
    data = [df[df["model_label"].astype(str) == lab][metric].dropna().to_numpy() for lab in labels]

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    boxplot_kwargs = dict(
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 0.8},
        whiskerprops={"linewidth": 0.8},
        capprops={"linewidth": 0.8},
    )
    try:
        bp = ax.boxplot(data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, **boxplot_kwargs)
    for patch in bp["boxes"]:
        patch.set_alpha(0.72)
    ax.set_ylabel(f"{metric.upper()} in standardized space")
    ax.set_title(f"Error distribution and robustness comparison ({metric.upper()})")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save_figure(fig, out_dir, f"Fig08_error_distribution_boxplot_{metric}")


def fig_best_seq_len_distribution(df: pd.DataFrame, out_dir: Path, metric: str = "mae") -> None:
    best = compute_best_seq_len_distribution(df, metric=metric)
    labels = ordered_labels(best["model_label"].astype(str).unique())
    mat = (
        best.pivot(index="model_label", columns="sl", values="proportion")
        .reindex(index=labels, columns=SEQ_LEN_ORDER)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    im = ax.imshow(mat.to_numpy() * 100, aspect="auto", interpolation="nearest", cmap="Blues")
    ax.set_xticks(np.arange(len(SEQ_LEN_ORDER)))
    ax.set_xticklabels(SEQ_LEN_ORDER)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("seq_len")
    ax.set_ylabel("Model")
    ax.set_title(f"Distribution of best input window lengths based on {metric.upper()}")
    values = mat.to_numpy() * 100
    threshold = np.nanmax(values) * 0.52 if np.nanmax(values) > 0 else 50
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7.5, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Proportion of horizons where seq_len is best (%)")
    fig.tight_layout()
    save_figure(fig, out_dir, f"Fig09_best_seq_len_distribution_{metric}")


# =============================================================================
# 5. Appendix figures
# =============================================================================

def fig_all_model_horizon_curves(df: pd.DataFrame, out_dir: Path, metric: str, smooth_window: int = 10) -> None:
    curve = (
        df.groupby(["model_label", "pl"], observed=True)[metric]
        .mean()
        .reset_index()
        .sort_values(["model_label", "pl"])
    )
    labels = ordered_labels(curve["model_label"].astype(str).unique())

    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    for lab in labels:
        s = curve[curve["model_label"].astype(str) == lab].sort_values("pl")
        y = s[metric].rolling(smooth_window, center=True, min_periods=1).mean()
        ax.plot(s["pl"], y, label=lab)
    ax.set_xlabel("Forecasting horizon (pred_len)")
    ax.set_ylabel(f"{metric.upper()} in standardized space")
    ax.set_title(f"Full horizon-wise {metric.upper()} curves of all models")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    save_figure(fig, out_dir, f"Appendix_full_horizon_curves_all_models_{metric}")


def fig_supplementary_mape_mspe(df: pd.DataFrame, out_dir: Path) -> None:
    # Legacy standardized-space values: do not interpret as original-scale percentages.
    overall = (
        df.groupby("model_label", observed=True)[["mape", "mspe"]]
        .mean()
        .reset_index()
        .sort_values("mape", ascending=True)
    )
    labels = overall["model_label"].astype(str).tolist()
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    for ax, metric, title in zip(axes, ["mape", "mspe"], ["Average MAPE", "Average MSPE"]):
        values = overall[metric].to_numpy()
        ax.barh(y, values, edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(metric.upper())
        ax.set_title(title)
        ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.suptitle("Supplementary percentage-based metrics in standardized space", y=1.02, fontsize=13)
    fig.tight_layout()
    save_figure(fig, out_dir, "Appendix_supplementary_mape_mspe")


def fig_improvement_heatmap(df: pd.DataFrame, out_dir: Path, metric: str = "mae") -> None:
    imp = compute_improvement_ratios(df)
    metric_col = f"{metric}_improvement_pct"
    imp["horizon_bucket"] = imp["pl"].apply(heatmap_bucket)
    variant_labels = ["NST-LastVal", "NST-Mean", "NST-Median"]
    matrices = {}
    for lab in variant_labels:
        s = imp[imp["variant_label"] == lab]
        mat = s.pivot_table(
            index="sl", columns="horizon_bucket", values=metric_col, aggfunc="mean", observed=True
        )
        mat = mat.reindex(index=SEQ_LEN_ORDER, columns=[x[0] for x in HEATMAP_BUCKETS])
        matrices[lab] = mat

    vmax = max(abs(float(np.nanmin(m.to_numpy()))) for m in matrices.values())
    vmax = max(vmax, max(abs(float(np.nanmax(m.to_numpy()))) for m in matrices.values()))
    vmin = -vmax

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    last_im = None
    for ax, lab in zip(axes, variant_labels):
        mat = matrices[lab]
        last_im = ax.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest", cmap="RdBu", vmin=vmin, vmax=vmax)
        ax.set_title(lab)
        ax.set_xticks(np.arange(mat.shape[1]))
        ax.set_xticklabels(mat.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(mat.shape[0]))
        ax.set_yticklabels(mat.index)
        ax.set_xlabel("Forecasting horizon bucket")
        ax.set_ylabel("seq_len")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.iloc[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.8)
    fig.subplots_adjust(right=0.90, wspace=0.34, top=0.84)
    cbar_ax = fig.add_axes([0.92, 0.20, 0.018, 0.58])
    cbar = fig.colorbar(last_im, cax=cbar_ax)
    cbar.set_label(f"{metric.upper()} improvement over NST (%)")
    fig.suptitle(f"Improvement heatmaps of proximal variants based on {metric.upper()}", y=0.96, fontsize=13)
    save_figure(fig, out_dir, f"Appendix_improvement_heatmap_{metric}")


# =============================================================================
# 6. Optional representative prediction curves from pred.npy / true.npy
# =============================================================================

def parse_setting_from_folder_name(folder_name: str) -> Dict[str, Optional[int | str]]:
    """Parse model, sl, ll and pl from a result folder name.
    This is a helper for optional prediction-curve plotting. It may need small adjustments
    if your local folder naming differs.
    """
    info: Dict[str, Optional[int | str]] = {"model": None, "sl": None, "ll": None, "pl": None}
    for model in sorted(MODEL_ORDER, key=len, reverse=True):
        if model in folder_name:
            info["model"] = model
            break
    for key in ["sl", "ll", "pl"]:
        m = re.search(rf"_{key}(\d+)", folder_name)
        if m:
            info[key] = int(m.group(1))
    return info


def find_result_folder(results_root: Path, model: str, sl: int, pl: int, ll: Optional[int] = None) -> Optional[Path]:
    candidates = []
    for p in results_root.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if model not in name:
            continue
        info = parse_setting_from_folder_name(name)
        if info["sl"] == sl and info["pl"] == pl and (ll is None or info["ll"] == ll):
            if (p / "pred.npy").exists() and (p / "true.npy").exists():
                candidates.append(p)
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: len(x.name))[0]


def plot_representative_prediction_case(
    results_root: Path,
    out_dir: Path,
    sl: int,
    pl: int,
    ll: Optional[int] = None,
    sample_index: int = 0,
) -> None:
    """Optional figure: true vs predicted curves for NST variants.

    Use this only when you have the local result folders with pred.npy and true.npy.
    """
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    true_plotted = False
    x_axis = np.arange(1, pl + 1)

    for model in NST_VARIANT_MODELS:
        folder = find_result_folder(results_root, model, sl=sl, pl=pl, ll=ll)
        if folder is None:
            print(f"Warning: no pred/true folder found for model={model}, sl={sl}, pl={pl}")
            continue
        pred = np.load(folder / "pred.npy")
        true = np.load(folder / "true.npy")

        # Common shapes are [num_samples, pred_len, channels]. For univariate, channel index is 0.
        pred_series = pred[sample_index, :pl, 0].reshape(-1)
        true_series = true[sample_index, :pl, 0].reshape(-1)
        if not true_plotted:
            ax.plot(x_axis, true_series, label="True", color="black", linewidth=2.2)
            true_plotted = True
        ax.plot(x_axis, pred_series, label=MODEL_LABELS[model], linewidth=1.5)

    ax.set_xlabel("Forecast step")
    ax.set_ylabel("Standardized Close")
    ax.set_title(f"Representative prediction case: sl={sl}, pl={pl}")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_dir, f"Fig10_representative_prediction_sl{sl}_pl{pl}")


# =============================================================================
# 7. Main execution
# =============================================================================

def main() -> None:
    df = load_and_prepare_data(EXCEL_PATH, SHEET_NAME)
    # Validate paired comparisons before producing any files.
    compute_improvement_ratios(df)
    paths = ensure_dirs()

    coverage = validate_experiment_coverage(df, paths["tables"])
    print("Model coverage:")
    print(coverage.to_string(index=False))

    tables = build_summary_tables(df, paths["tables"])
    print("\nOverall ranking by MAE:")
    print(tables["overall_ranking"][["rank_by_mae", "model_label", "mae", "mse", "rmse"]].to_string(index=False))

    # Main figures.
    fig_overall_ranking(df, paths["fig_main"])
    fig_horizon_grouped_performance(df, paths["fig_main"])
    fig_horizon_wise_nst_variants(df, paths["fig_main"])
    fig_improvement_ratio(df, paths["fig_main"])
    fig_heatmap_seq_horizon_nst(df, paths["fig_main"], metric="mae")
    fig_error_distribution_boxplot(df, paths["fig_main"], metric="mae")
    fig_best_seq_len_distribution(df, paths["fig_main"], metric="mae")

    # Appendix figures.
    fig_all_model_horizon_curves(df, paths["fig_appendix"], metric="mae")
    fig_all_model_horizon_curves(df, paths["fig_appendix"], metric="rmse")
    fig_supplementary_mape_mspe(df, paths["fig_appendix"])
    fig_improvement_heatmap(df, paths["fig_appendix"], metric="mae")
    fig_improvement_heatmap(df, paths["fig_appendix"], metric="rmse")

    # Optional prediction curves. Uncomment after setting RESULTS_ROOT to your local results folder.
    # if RESULTS_ROOT is not None and RESULTS_ROOT.exists():
    #     plot_representative_prediction_case(RESULTS_ROOT, paths["fig_main"], sl=64, pl=50, sample_index=0)
    #     plot_representative_prediction_case(RESULTS_ROOT, paths["fig_main"], sl=64, pl=200, sample_index=0)
    #     plot_representative_prediction_case(RESULTS_ROOT, paths["fig_main"], sl=64, pl=500, sample_index=0)

    print(f"\nDone. Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
