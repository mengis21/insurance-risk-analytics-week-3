import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join("reports", "images")
DATA_PATH = os.path.join("SM", "data", "insurance.csv")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Parse dates when present
    for col in ["TransactionMonth", "The transaction date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    # Normalize numeric types
    for col in ["TotalPremium", "TotalClaims", "CalculatedPremiumPerTerm", "CustomValueEstimate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fallback mapping when working with simplified sample datasets
    if "TotalClaims" not in df.columns and "charges" in df.columns:
        df["TotalClaims"] = pd.to_numeric(df["charges"], errors="coerce")
    if "TotalPremium" not in df.columns and "TotalClaims" in df.columns:
        # Assume a 20% loading over claims when premium is not provided
        df["TotalPremium"] = df["TotalClaims"] * 1.2
    if "Gender" not in df.columns and "sex" in df.columns:
        df["Gender"] = df["sex"].str.capitalize()
    if "Province" not in df.columns and "region" in df.columns:
        df["Province"] = df["region"].str.replace("_", " ").str.title()

    return df


def kpis(df: pd.DataFrame) -> dict:
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


def plot_loss_ratio_by_segments(df: pd.DataFrame) -> None:
    for segment in ["Province", "VehicleType", "Gender"]:
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


def plot_monthly_trends(df: pd.DataFrame) -> None:
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


def plot_correlations(df: pd.DataFrame) -> None:
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
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    df = load_data(DATA_PATH)
    df = preprocess(df)
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
