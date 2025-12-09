"""Utility helpers for loading and enriching the insurance dataset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DATA_PATH = Path("SM") / "data" / "insurance.csv"

# Map Kaggle-style regions to South African provinces used in reports
REGION_PROVINCE_MAP: Dict[str, str] = {
    "southwest": "Western Cape",
    "southeast": "KwaZulu-Natal",
    "northwest": "Gauteng",
    "northeast": "Mpumalanga",
}

# Synthetic but consistent postal codes for each province (major metro codes)
PROVINCE_POSTAL_CODES: Dict[str, List[str]] = {
    "Western Cape": ["7441", "7550", "8001", "8041"],
    "KwaZulu-Natal": ["4001", "4051", "4319", "4420"],
    "Gauteng": ["1685", "2001", "2167", "2191"],
    "Mpumalanga": ["1200", "1320", "1459", "1675"],
    "Other": ["9301", "9305", "9306", "9499"],
}


@dataclass(frozen=True)
class DatasetMetadata:
    """Captures reference details for downstream reporting."""

    severity_q1: float
    severity_median: float
    severity_q3: float
    severity_p90: float


def _assign_postal_code(province: str, index: int) -> str:
    codes = PROVINCE_POSTAL_CODES.get(province, PROVINCE_POSTAL_CODES["Other"])
    if not codes:
        raise ValueError("Postal code list cannot be empty")
    return codes[index % len(codes)]


def load_enriched_dataset() -> Tuple[pd.DataFrame, DatasetMetadata]:
    """Loads the source data and derives analysis-friendly columns.

    Returns
    -------
    Tuple[pd.DataFrame, DatasetMetadata]
        Enriched dataframe with engineered fields and summary metadata.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Align naming conventions
    df["Gender"] = df.get("Gender", df.get("sex", "")).astype(str).str.capitalize()
    df["Smoker"] = df.get("Smoker", df.get("smoker", "")).astype(str).str.capitalize()

    # Claims, premium, and profitability proxies
    df["TotalClaims"] = pd.to_numeric(df.get("TotalClaims", df.get("charges")), errors="coerce")
    df["TotalPremium"] = pd.to_numeric(df.get("TotalPremium"), errors="coerce")
    df["TotalPremium"] = df["TotalPremium"].fillna(df["TotalClaims"] * 1.2)
    df["CalculatedPremiumPerTerm"] = df.get("CalculatedPremiumPerTerm")
    df["CalculatedPremiumPerTerm"] = pd.to_numeric(df["CalculatedPremiumPerTerm"], errors="coerce")
    df["CalculatedPremiumPerTerm"] = df["CalculatedPremiumPerTerm"].fillna(df["TotalPremium"] / 12.0)
    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    # Geographic enrichment
    if "Province" in df.columns:
        df["Province"] = df["Province"].fillna(
            df.get("region", "").astype(str).str.lower().map(REGION_PROVINCE_MAP)
        )
    else:
        df["Province"] = df.get("region", "").astype(str).str.lower().map(REGION_PROVINCE_MAP)
    df["Province"] = df["Province"].fillna("Other")

    df["PostalCode"] = [
        _assign_postal_code(province, idx) for idx, province in enumerate(df["Province"].tolist())
    ]

    # Helper categorical buckets
    df["ChildrenBucket"] = pd.cut(df.get("children", 0), bins=[-1, 0, 2, 5], labels=["0", "1-2", "3+"])
    df["AgeBand"] = pd.cut(
        df.get("age", 0),
        bins=[17, 25, 35, 45, 55, 65],
        labels=["18-25", "26-35", "36-45", "46-55", "56-65"],
    )

    # Severity proxies
    severity_quantiles = df["TotalClaims"].quantile([0.25, 0.5, 0.75, 0.9]).to_dict()
    df["SeverityTier"] = pd.qcut(
        df["TotalClaims"],
        q=4,
        labels=["Low", "Moderate", "High", "Very High"],
        duplicates="drop",
    )
    threshold = severity_quantiles[0.75]
    df["HighSeverityFlag"] = df["TotalClaims"] > threshold

    metadata = DatasetMetadata(
        severity_q1=severity_quantiles[0.25],
        severity_median=severity_quantiles[0.5],
        severity_q3=severity_quantiles[0.75],
        severity_p90=severity_quantiles[0.9],
    )

    return df, metadata
