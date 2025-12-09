"""Task 3 statistical hypothesis testing pipeline."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

from data_utils import load_enriched_dataset

REPORTS_DIR = Path("reports")
SUMMARY_JSON = REPORTS_DIR / "task3_hypothesis_tests.json"
SUMMARY_CSV = REPORTS_DIR / "task3_hypothesis_tests.csv"


def _run_anova(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, float]:
    grouped = [segment[value_col].dropna().values for _, segment in df.groupby(group_col)]
    if len(grouped) < 2:
        raise ValueError("ANOVA requires at least two groups")
    f_stat, p_val = stats.f_oneway(*grouped)
    grand_mean = df[value_col].mean()
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in grouped)
    ss_within = sum(((group - group.mean()) ** 2).sum() for group in grouped)
    eta_sq = ss_between / (ss_between + ss_within) if (ss_between + ss_within) else np.nan
    return {
        "statistic": float(f_stat),
        "p_value": float(p_val),
        "effect_size": eta_sq,
        "df_num": float(len(grouped) - 1),
        "df_denom": float(len(df) - len(grouped)),
    }


def _run_chi_square(df: pd.DataFrame, group_col: str, flag_col: str) -> Dict[str, float]:
    table = pd.crosstab(df[group_col], df[flag_col])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return {"statistic": np.nan, "p_value": np.nan, "effect_size": np.nan, "dof": np.nan}
    chi2, p_val, dof, expected = stats.chi2_contingency(table)
    cramers_v = np.sqrt((chi2 / table.sum().sum()) / (min(table.shape) - 1))
    return {
        "statistic": float(chi2),
        "p_value": float(p_val),
        "effect_size": float(cramers_v),
        "dof": float(dof),
    }


def _run_welch_t(df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, float]:
    groups = [vals[value_col].dropna().values for _, vals in df.groupby(group_col)]
    if len(groups) != 2:
        raise ValueError("Welch t-test expects exactly two groups")
    stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
    n1, n2 = len(groups[0]), len(groups[1])
    s1_sq, s2_sq = np.var(groups[0], ddof=1), np.var(groups[1], ddof=1)
    pooled = ((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2)
    d = (np.mean(groups[0]) - np.mean(groups[1])) / np.sqrt(pooled) if pooled > 0 else np.nan
    df_num = (s1_sq / n1 + s2_sq / n2) ** 2
    df_denom = ((s1_sq ** 2) / (n1 ** 2 * (n1 - 1))) + ((s2_sq ** 2) / (n2 ** 2 * (n2 - 1)))
    df_est = df_num / df_denom if df_denom else np.nan
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "effect_size": float(d),
        "df": float(df_est) if not np.isnan(df_est) else np.nan,
    }


def _group_summary(df: pd.DataFrame, group_col: str, value_col: str) -> List[Dict[str, float]]:
    summary = (
        df.groupby(group_col)[value_col]
        .agg(["mean", "median", "count", "sum"])
        .rename(columns={"mean": "avg", "median": "median", "count": "n"})
        .reset_index()
    )
    return summary.to_dict(orient="records")


def _format_decision(p_value: float, alpha: float = 0.05) -> str:
    if np.isnan(p_value):
        return "indeterminate"
    return "reject" if p_value < alpha else "fail_to_reject"


def run_hypothesis_tests() -> List[Dict[str, object]]:
    df, _metadata = load_enriched_dataset()
    results: List[Dict[str, object]] = []

    # 1. Province severity differences
    anova_res = _run_anova(df, "Province", "TotalClaims")
    province_summary = _group_summary(df, "Province", "TotalClaims")
    top_province = max(province_summary, key=lambda item: item["avg"]) if province_summary else None
    if top_province:
        province_note = (
            f"Highest observed average severity in {top_province['Province']} at {top_province['avg']:.0f}."
        )
    else:
        province_note = "Unable to compute provincial averages."
    results.append(
        {
            "hypothesis": "H1: Average claim severity differs across provinces.",
            "segment": "Province",
            "metric": "TotalClaims",
            "test": "One-way ANOVA",
            "decision": _format_decision(anova_res["p_value"]),
            "statistic": anova_res["statistic"],
            "p_value": anova_res["p_value"],
            "effect_size": anova_res["effect_size"],
            "interpretation": province_note,
            "comparison_summary": province_summary,
        }
    )

    # 2. Province high severity incidence
    chi_province = _run_chi_square(df, "Province", "HighSeverityFlag")
    incidence_df = df.groupby("Province")["HighSeverityFlag"].mean().reset_index(name="high_severity_rate")
    incidence = incidence_df.to_dict(orient="records")
    national_rate = float(df["HighSeverityFlag"].mean())
    results.append(
        {
            "hypothesis": "H2: The share of high severity claims varies by province.",
            "segment": "Province",
            "metric": "HighSeverityFlag",
            "test": "Chi-square",
            "decision": _format_decision(chi_province["p_value"]),
            "statistic": chi_province["statistic"],
            "p_value": chi_province["p_value"],
            "effect_size": chi_province["effect_size"],
            "interpretation": f"National high severity rate {national_rate:.2%}; provinces above this level require remediation.",
            "comparison_summary": incidence,
        }
    )

    # 3. Postal code severity spread
    anova_zip = _run_anova(df, "PostalCode", "TotalClaims")
    zip_summary = _group_summary(df, "PostalCode", "TotalClaims")
    results.append(
        {
            "hypothesis": "H3: Claim severity differs across postal codes within provinces.",
            "segment": "PostalCode",
            "metric": "TotalClaims",
            "test": "One-way ANOVA",
            "decision": _format_decision(anova_zip["p_value"]),
            "statistic": anova_zip["statistic"],
            "p_value": anova_zip["p_value"],
            "effect_size": anova_zip["effect_size"],
            "interpretation": "Postal clusters with persistently higher severity warrant tighter underwriting controls.",
            "comparison_summary": zip_summary,
        }
    )

    # 4. Postal code contribution margin spread
    anova_margin = _run_anova(df, "PostalCode", "Margin")
    margin_summary = _group_summary(df, "PostalCode", "Margin")
    if margin_summary:
        worst_margin = min(margin_summary, key=lambda item: item["avg"])
        margin_note = (
            f"Lowest contribution margin in postal code {worst_margin['PostalCode']} at {worst_margin['avg']:.0f}."
        )
    else:
        margin_note = "Margin summary unavailable."
    results.append(
        {
            "hypothesis": "H4: Contribution margin differs materially across postal codes.",
            "segment": "PostalCode",
            "metric": "Margin",
            "test": "One-way ANOVA",
            "decision": _format_decision(anova_margin["p_value"]),
            "statistic": anova_margin["statistic"],
            "p_value": anova_margin["p_value"],
            "effect_size": anova_margin["effect_size"],
            "interpretation": margin_note,
            "comparison_summary": margin_summary,
        }
    )

    # 5. Gender severity differences (Welch t-test)
    welch_gender = _run_welch_t(df, "Gender", "TotalClaims")
    gender_summary = _group_summary(df, "Gender", "TotalClaims")
    if len(gender_summary) == 2:
        diff = abs(gender_summary[0]["avg"] - gender_summary[1]["avg"])
        gender_note = f"Average severity gap between genders is {diff:.0f}."
    else:
        gender_note = "Gender distribution insufficient for comparison."
    results.append(
        {
            "hypothesis": "H5: Male vs female insured show different claim severity.",
            "segment": "Gender",
            "metric": "TotalClaims",
            "test": "Welch t-test",
            "decision": _format_decision(welch_gender["p_value"]),
            "statistic": welch_gender["statistic"],
            "p_value": welch_gender["p_value"],
            "effect_size": welch_gender["effect_size"],
            "interpretation": gender_note,
            "comparison_summary": gender_summary,
        }
    )

    return results


def _clean_for_json(value):
    if isinstance(value, dict):
        return {key: _clean_for_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_for_json(item) for item in value]
    if isinstance(value, (np.floating, float, np.integer, int)):
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def save_results(results: List[Dict[str, object]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cleaned = [_clean_for_json(entry) for entry in results]
    with SUMMARY_JSON.open("w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, indent=2)
    pd.DataFrame(cleaned).to_csv(SUMMARY_CSV, index=False)


def main() -> None:
    results = run_hypothesis_tests()
    save_results(results)
    for entry in results:
        print(f"{entry['test']} | {entry['hypothesis']} -> decision: {entry['decision']} (p={entry['p_value']:.4f})")


if __name__ == "__main__":
    main()
