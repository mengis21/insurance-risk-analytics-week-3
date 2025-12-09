import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data_utils import load_enriched_dataset

OUTPUT_DIR = os.path.join("reports", "images")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def kpis(df) -> dict:
    total_premium = df["TotalPremium"].sum(min_count=1)
    total_claims = df["TotalClaims"].sum(min_count=1)
    loss_ratio = (total_claims / total_premium) if total_premium and not np.isnan(total_premium) else np.nan

    return {
        "total_premium": total_premium,
        "total_claims": total_claims,
        "loss_ratio": loss_ratio,
    }


def plot_distributions(df: pd.DataFrame) -> None:
    if "TotalPremium" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df["TotalPremium"].dropna(), bins=30, kde=True)
        plt.title("Distribution of TotalPremium")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "dist_totalpremium.png"))
        plt.close()

    if "TotalClaims" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df["TotalClaims"].dropna(), bins=30, kde=True, color="tomato")
        plt.title("Distribution of TotalClaims")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "dist_totalclaims.png"))
        plt.close()


def plot_loss_ratio_by_segments(df) -> None:
    for segment in ["Province", "PostalCode", "Gender"]:
        if segment in df.columns:
            grp = df.groupby(segment).agg({"TotalPremium": "sum", "TotalClaims": "sum"})
            grp["LossRatio"] = grp["TotalClaims"] / grp["TotalPremium"]
            grp = grp.sort_values("LossRatio", ascending=False)
            plt.figure(figsize=(12, 6))
            sns.barplot(x=grp.index, y=grp["LossRatio"], color="steelblue")
            plt.xticks(rotation=45, ha="right")
            plt.title(f"Loss Ratio by {segment}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f"loss_ratio_by_{segment.lower()}.png"))
            plt.close()


def plot_monthly_trends(df) -> None:
    dt_col = None
    if "TransactionMonth" in df.columns:
        dt_col = "TransactionMonth"
    elif "The transaction date" in df.columns:
        dt_col = "The transaction date"
    if not dt_col or "TotalPremium" not in df.columns or "TotalClaims" not in df.columns:
        return

    monthly = df.dropna(subset=[dt_col]).copy()
    monthly["month"] = monthly[dt_col].dt.to_period("M").dt.to_timestamp()
    agg = monthly.groupby("month").agg({"TotalPremium": "sum", "TotalClaims": "sum"}).reset_index()

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=agg, x="month", y="TotalPremium", label="TotalPremium")
    sns.lineplot(data=agg, x="month", y="TotalClaims", label="TotalClaims")
    plt.title("Monthly Premium vs Claims")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "monthly_premium_claims.png"))
    plt.close()


def plot_correlations(df) -> None:
    cols = [c for c in ["TotalPremium", "TotalClaims", "CalculatedPremiumPerTerm", "CustomValueEstimate"] if c in df.columns]
    if not cols:
        return
    corr = df[cols].corr(numeric_only=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="Blues")
    plt.title("Correlation Matrix (Key Financial Variables)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
    plt.close()


def main():
    ensure_output_dir(OUTPUT_DIR)
    df, _ = load_enriched_dataset()
    metrics = kpis(df)

    # Save KPIs to file
    kpi_path = os.path.join("reports", "kpis_interim.json")
    os.makedirs(os.path.dirname(kpi_path), exist_ok=True)
    pd.Series(metrics).to_json(kpi_path)

    # Plots for interim
    plot_distributions(df)
    plot_loss_ratio_by_segments(df)
    plot_monthly_trends(df)
    plot_correlations(df)

    print("Interim EDA complete.")
    print(metrics)


if __name__ == "__main__":
    main()
