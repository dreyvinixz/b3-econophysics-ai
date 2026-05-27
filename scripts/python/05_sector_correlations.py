from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

SECTOR_MAP_PATH = TABLES_DIR / "assets_sector_map.csv"
CORRELATION_PAIRS_PATH = TABLES_DIR / "core_historical_correlation_pairs_1998_2025.csv"

SECTOR_CORRELATIONS_PATH = TABLES_DIR / "core_historical_sector_correlations_1998_2025.csv"
SECTOR_SUMMARY_PATH = TABLES_DIR / "core_historical_sector_correlation_summary_1998_2025.csv"

FIGURE_PDF_PATH = FIGURES_DIR / "vector" / "figure_3_sector_correlation_distribution.pdf"
FIGURE_PNG_PATH = FIGURES_DIR / "preview" / "figure_3_sector_correlation_distribution.png"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.7,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    sector_map = pd.read_csv(SECTOR_MAP_PATH)
    pairs = pd.read_csv(CORRELATION_PAIRS_PATH)

    required_map_columns = {"symbol", "sector", "subsector", "segment"}
    required_pair_columns = {"symbol1", "symbol2", "correlation", "n_obs"}

    missing_map = required_map_columns - set(sector_map.columns)
    missing_pairs = required_pair_columns - set(pairs.columns)

    if missing_map:
        raise ValueError(f"Missing columns in {SECTOR_MAP_PATH}: {sorted(missing_map)}")
    if missing_pairs:
        raise ValueError(f"Missing columns in {CORRELATION_PAIRS_PATH}: {sorted(missing_pairs)}")

    return sector_map, pairs


def attach_sector_classification(sector_map: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    left = sector_map.rename(
        columns={
            "symbol": "symbol1",
            "sector": "sector1",
            "subsector": "subsector1",
            "segment": "segment1",
            "company_name": "company_name1",
        }
    )
    right = sector_map.rename(
        columns={
            "symbol": "symbol2",
            "sector": "sector2",
            "subsector": "subsector2",
            "segment": "segment2",
            "company_name": "company_name2",
        }
    )

    enriched = pairs.merge(left, on="symbol1", how="left").merge(right, on="symbol2", how="left")

    missing = enriched[enriched["sector1"].isna() | enriched["sector2"].isna()]
    if not missing.empty:
        missing_symbols = sorted(
            set(missing.loc[missing["sector1"].isna(), "symbol1"])
            | set(missing.loc[missing["sector2"].isna(), "symbol2"])
        )
        raise ValueError(f"Missing sector classification for symbols: {missing_symbols}")

    enriched["correlation_group"] = np.where(
        enriched["sector1"] == enriched["sector2"],
        "Within-sector",
        "Between-sector",
    )
    enriched["same_subsector"] = enriched["subsector1"] == enriched["subsector2"]
    enriched["classification_version"] = "initial macro-sector classification"

    ordered_columns = [
        "symbol1",
        "symbol2",
        "company_name1",
        "company_name2",
        "sector1",
        "sector2",
        "subsector1",
        "subsector2",
        "segment1",
        "segment2",
        "correlation_group",
        "same_subsector",
        "correlation",
        "n_obs",
        "classification_version",
    ]
    return enriched[ordered_columns].sort_values(["correlation_group", "correlation"], ascending=[False, False])


def summarize_groups(sector_pairs: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    summary = (
        sector_pairs.groupby("correlation_group")["correlation"]
        .agg(
            n_pairs="count",
            mean="mean",
            median="median",
            std="std",
            min="min",
            max="max",
        )
        .reset_index()
    )

    within = sector_pairs.loc[
        sector_pairs["correlation_group"] == "Within-sector", "correlation"
    ].dropna()
    between = sector_pairs.loc[
        sector_pairs["correlation_group"] == "Between-sector", "correlation"
    ].dropna()

    test = mannwhitneyu(within, between, alternative="greater")
    summary["mann_whitney_u_within_gt_between"] = test.statistic
    summary["mann_whitney_p_value_within_gt_between"] = test.pvalue

    return summary, float(test.statistic), float(test.pvalue)


def print_summary(summary: pd.DataFrame, u_statistic: float, p_value: float) -> None:
    print("\nSector correlation summary")
    print("=" * 80)

    for group in ["Within-sector", "Between-sector"]:
        row = summary.loc[summary["correlation_group"] == group].iloc[0]
        print(f"\n{group}:")
        print(f"  n_pairs: {int(row['n_pairs'])}")
        print(f"  mean:    {row['mean']:.4f}")
        print(f"  median:  {row['median']:.4f}")
        print(f"  std:     {row['std']:.4f}")
        print(f"  min:     {row['min']:.4f}")
        print(f"  max:     {row['max']:.4f}")

    print("\nMann-Whitney U test")
    print("  H1: within-sector correlations > between-sector correlations")
    print(f"  U statistic: {u_statistic:.4f}")
    print(f"  p-value:     {p_value:.6g}")


def plot_sector_correlations(sector_pairs: pd.DataFrame) -> None:
    configure_matplotlib()
    FIGURE_PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIGURE_PNG_PATH.parent.mkdir(parents=True, exist_ok=True)

    groups = ["Within-sector", "Between-sector"]
    data = [
        sector_pairs.loc[sector_pairs["correlation_group"] == group, "correlation"].dropna().to_numpy()
        for group in groups
    ]
    overall_mean = sector_pairs["correlation"].mean()

    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    box = ax.boxplot(
        data,
        positions=np.arange(1, len(groups) + 1),
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.1},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 0.8},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )

    colors = ["#4c78a8", "#9a9a9a"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.18)

    for position, values, color in zip([1, 2], data, colors):
        jitter = rng.normal(0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=7,
            alpha=0.28,
            color=color,
            edgecolors="none",
            rasterized=True,
        )

    ax.axhline(
        overall_mean,
        color="0.25",
        linestyle="--",
        linewidth=0.8,
        alpha=0.85,
        label=fr"overall mean = {overall_mean:.2f}",
    )

    ax.set_xticks([1, 2])
    ax.set_xticklabels(groups)
    ax.set_ylabel(r"Pearson correlation $\rho_{ij}$")
    ax.set_xlabel("correlation group")
    ax.legend(loc="upper right")
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(FIGURE_PDF_PATH, bbox_inches="tight")
    fig.savefig(FIGURE_PNG_PATH, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    sector_map, pairs = load_inputs()
    sector_pairs = attach_sector_classification(sector_map, pairs)
    summary, u_statistic, p_value = summarize_groups(sector_pairs)

    sector_pairs.to_csv(SECTOR_CORRELATIONS_PATH, index=False)
    summary.to_csv(SECTOR_SUMMARY_PATH, index=False)
    plot_sector_correlations(sector_pairs)

    print(f"Saved {SECTOR_CORRELATIONS_PATH}")
    print(f"Saved {SECTOR_SUMMARY_PATH}")
    print(f"Saved {FIGURE_PDF_PATH}")
    print(f"Saved {FIGURE_PNG_PATH}")
    print_summary(summary, u_statistic, p_value)


if __name__ == "__main__":
    main()
