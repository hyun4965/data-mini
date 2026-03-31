"""Feature loading helpers for early-life battery modeling.

This module builds question-aligned feature blocks so the final model can
explicitly trace which input signals came from Q1~Q5 insights.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .multi_batch_eda import build_batch_analysis, resolve_batch_specs
except ImportError:  # pragma: no cover - fallback for direct script execution
    from multi_batch_eda import build_batch_analysis, resolve_batch_specs


QUESTION_FEATURE_BLOCKS = {
    "Q1": [
        "Qd_10",
        "QC_10",
        "IR_10",
        "Tavg_10",
        "Tmax_10",
        "Tmin_10",
        "temp_range_10",
        "chargetime_10",
        "policy_charge_c_first",
        "policy_charge_c_last",
        "policy_soc_pct",
        "policy_has_varcharge",
        "policy_has_newstructure",
        "policy_has_slowcycle",
    ],
    "Q2": [
        "Qd_100",
        "Qd_delta_100_10",
        "Qd_retention_100_10",
        "Qd_slope_1_100",
        "Qd_drop_per_100",
        "Qd_mean_1_100",
        "Qd_std_1_100",
        "Qd_resid_std_1_100",
    ],
    "Q3": [
        "IR_100_mean",
        "IR_100_std",
        "IR_delta_100_10",
        "IR_slope_1_100",
        "Tavg_100_mean",
        "Tavg_100_std",
        "Tavg_delta_100_10",
        "Tmax_100_mean",
        "Tmin_100_mean",
        "temp_range_100_mean",
    ],
    "Q4": [
        "QC_100",
        "QC_delta_100_10",
        "QC_retention_100_10",
        "QC_slope_1_100",
        "chargetime_100_mean",
        "chargetime_100_std",
        "chargetime_delta_100_10",
        "Qd_QC_ratio_100",
        "coulombic_efficiency_100_mean",
    ],
    "Q5": [
        "Qd_drop_x_IR_mean",
        "Qd_retention_x_chargetime_mean",
        "IR_x_Tavg_mean",
        "Qd_delta_x_policy_charge",
        "Qd_cv_1_100",
        "IR_cv_1_100",
        "chargetime_cv_1_100",
    ],
}

HONEST_LOCAL_REFIT_BASE_FEATURES = [
    "Qd_delta_100_10",
    "Qd_retention_100_10",
    "IR_delta_100_10",
    "QC_retention_100_10",
    "chargetime_100_mean",
    "IR_cv_1_100",
    "Qd_QC_ratio_100",
    "Qd_slope_1_100",
    "Qd_drop_per_100",
]

HONEST_LOCAL_REFIT_EXTRA_POOL = [
    "policy_soc_pct",
    "policy_charge_c_last",
    "Qd_cv_1_100",
    "Qd_resid_std_1_100",
    "IR_100_mean",
    "Qd_mean_1_100",
    "coulombic_efficiency_100_mean",
    "QC_delta_100_10",
    "Qd_100",
]

HONEST_LOCAL_REFIT_SELECTED_FEATURES = [
    "Qd_delta_100_10",
    "Qd_retention_100_10",
    "IR_delta_100_10",
    "QC_retention_100_10",
    "chargetime_100_mean",
    "IR_cv_1_100",
    "Qd_QC_ratio_100",
    "Qd_slope_1_100",
    "Qd_100",
]

FEATURE_SET_MAP = {
    "baseline_q2": [
        "Qd_10",
        "Qd_100",
        "Qd_delta_100_10",
        "Qd_retention_100_10",
        "Qd_slope_1_100",
        "Qd_drop_per_100",
        "IR_100_mean",
        "Tavg_100_mean",
        "chargetime_100_mean",
    ],
    "full_q1_q5": [
        feature
        for features in QUESTION_FEATURE_BLOCKS.values()
        for feature in features
    ],
    "batch1_focused": [
        "Qd_10",
        "Qd_100",
        "Qd_delta_100_10",
        "Qd_retention_100_10",
        "Qd_slope_1_100",
        "Qd_drop_per_100",
        "chargetime_10",
        "chargetime_100_mean",
        "Qd_resid_std_1_100",
    ],
    "cross_batch_stable": [
        "Qd_retention_100_10",
        "QC_retention_100_10",
        "Qd_QC_ratio_100",
        "coulombic_efficiency_100_mean",
        "Qd_cv_1_100",
        "IR_cv_1_100",
        "chargetime_cv_1_100",
    ],
    "q1_q5_balanced": [
        "policy_charge_c_first",
        "policy_charge_c_last",
        "policy_soc_pct",
        "Qd_retention_100_10",
        "Qd_resid_std_1_100",
        "IR_delta_100_10",
        "IR_cv_1_100",
        "Tavg_delta_100_10",
        "QC_retention_100_10",
        "Qd_QC_ratio_100",
        "coulombic_efficiency_100_mean",
        "Qd_cv_1_100",
        "chargetime_cv_1_100",
    ],
    "q1_q5_pruned": [
        "policy_charge_c_first",
        "policy_charge_c_last",
        "policy_soc_pct",
        "Qd_delta_100_10",
        "Qd_retention_100_10",
        "Qd_drop_per_100",
        "Qd_resid_std_1_100",
        "IR_delta_100_10",
        "IR_100_std",
        "Tavg_delta_100_10",
        "QC_delta_100_10",
        "QC_retention_100_10",
        "Qd_QC_ratio_100",
        "coulombic_efficiency_100_mean",
        "Qd_cv_1_100",
        "IR_cv_1_100",
        "chargetime_cv_1_100",
    ],
    "q1_q5_policy_aware": [
        "policy_charge_c_first",
        "policy_charge_c_last",
        "policy_soc_pct",
        "policy_has_varcharge",
        "policy_has_newstructure",
        "policy_has_slowcycle",
        "Qd_retention_100_10",
        "QC_retention_100_10",
        "Qd_QC_ratio_100",
        "coulombic_efficiency_100_mean",
        "IR_delta_100_10",
        "Tavg_delta_100_10",
        "Qd_cv_1_100",
        "IR_cv_1_100",
        "chargetime_cv_1_100",
        "Qd_delta_x_policy_charge",
    ],
    "small_subset_best": [
        "Qd_cv_1_100",
        "IR_cv_1_100",
        "policy_soc_pct",
    ],
    "q1_q5_allblocks_best": [
        "policy_soc_pct",
        "Qd_resid_std_1_100",
        "IR_delta_100_10",
        "coulombic_efficiency_100_mean",
        "chargetime_cv_1_100",
    ],
    # PDF-guided set:
    # - PDF 1 introduces battery index / regression / feature framing.
    # - PDF 2 organizes EDA as Q1~Q5, so we keep one-to-two core signals per block.
    "pdf_guided_q1_q5": [
        "policy_soc_pct",
        "policy_charge_c_last",
        "Qd_retention_100_10",
        "Qd_cv_1_100",
        "Qd_resid_std_1_100",
        "IR_delta_100_10",
        "IR_cv_1_100",
        "QC_retention_100_10",
        "coulombic_efficiency_100_mean",
        "chargetime_cv_1_100",
    ],
    "pdf_guided_minimal": [
        "policy_soc_pct",
        "Qd_retention_100_10",
        "Qd_cv_1_100",
        "IR_cv_1_100",
        "QC_retention_100_10",
    ],
    # Final submission-safe presets:
    # - first 100 cycles only
    # - no knee / threshold / transition features
    # - no label-adjacent late-life diagnostics
    "submission_final_minimal": [
        "policy_soc_pct",
        "Qd_retention_100_10",
        "Qd_cv_1_100",
        "IR_cv_1_100",
        "QC_retention_100_10",
    ],
    "submission_final_balanced": [
        "policy_soc_pct",
        "Qd_delta_100_10",
        "Qd_retention_100_10",
        "IR_delta_100_10",
        "QC_retention_100_10",
        "chargetime_100_mean",
        "Qd_cv_1_100",
    ],
    "q1_q5_transfer_compact": [
        "policy_soc_pct",
        "Qd_retention_100_10",
        "Qd_cv_1_100",
        "IR_cv_1_100",
        "QC_retention_100_10",
        "chargetime_cv_1_100",
    ],
    "q1_q5_transfer_plus": [
        "policy_soc_pct",
        "Qd_retention_100_10",
        "Qd_cv_1_100",
        "IR_delta_100_10",
        "IR_cv_1_100",
        "QC_retention_100_10",
        "coulombic_efficiency_100_mean",
    ],
    "honest_local_refit_base": HONEST_LOCAL_REFIT_BASE_FEATURES,
    "honest_local_refit_best": HONEST_LOCAL_REFIT_SELECTED_FEATURES,
}

DEFAULT_FEATURE_SET = "full_q1_q5"
DEFAULT_NUMERIC_FEATURES = FEATURE_SET_MAP[DEFAULT_FEATURE_SET]

LEAKY_FEATURES = [
    "knee_cycle",
    "knee_ratio",
    "accel_onset_cycle",
    "accel_onset_ratio",
    "cycle_to_95_retention",
    "cycle_to_90_retention",
]


def _analysis_by_batch(project_root: Path) -> dict[str, dict]:
    results_dir = project_root / "results"
    analyses: dict[str, dict] = {}
    for spec in resolve_batch_specs(project_root):
        analyses[spec.key] = build_batch_analysis(spec, results_dir)
    return analyses


def _value_at_cycle(sub: pd.DataFrame, column: str, cycle: int) -> float:
    hit = sub.loc[sub["cycle"] == cycle, column]
    return float(hit.iloc[0]) if len(hit) else np.nan


def _series_mean(sub: pd.DataFrame, column: str) -> float:
    return float(sub[column].mean()) if sub[column].notna().any() else np.nan


def _series_std(sub: pd.DataFrame, column: str) -> float:
    return float(sub[column].std(ddof=0)) if sub[column].notna().any() else np.nan


def _series_cv(sub: pd.DataFrame, column: str) -> float:
    mean = _series_mean(sub, column)
    std = _series_std(sub, column)
    if pd.isna(mean) or mean == 0 or pd.isna(std):
        return np.nan
    return float(std / abs(mean))


def _slope(sub: pd.DataFrame, column: str) -> float:
    clean = sub[["cycle", column]].dropna()
    if len(clean) < 2:
        return np.nan
    return float(np.polyfit(clean["cycle"], clean[column], 1)[0])


def _residual_std(sub: pd.DataFrame, column: str) -> float:
    clean = sub[["cycle", column]].dropna()
    if len(clean) < 3:
        return np.nan
    coef = np.polyfit(clean["cycle"], clean[column], 1)
    pred = np.polyval(coef, clean["cycle"])
    return float(np.std(clean[column] - pred))


def _safe_ratio(num: float, den: float) -> float:
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return float(num / den)


def _parse_policy(policy: str) -> dict[str, float]:
    policy = "" if pd.isna(policy) else str(policy)
    lower = policy.lower()
    c_values = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)c", lower)]
    pct_values = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)%", lower)]
    return {
        "policy_charge_c_first": c_values[0] if c_values else np.nan,
        "policy_charge_c_last": c_values[-1] if c_values else np.nan,
        "policy_soc_pct": pct_values[0] if pct_values else np.nan,
        "policy_has_varcharge": float("varcharge" in lower),
        "policy_has_newstructure": float("newstructure" in lower),
        "policy_has_slowcycle": float("slowcycle" in lower),
    }


def build_question_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build a comprehensive Q1~Q5 feature table from cycle-level summaries."""
    rows: list[dict[str, float | str]] = []
    for cell_id, sub in df.groupby("cell_id"):
        sub = sub.sort_values("cycle").copy()
        first_100 = sub[sub["cycle"] <= 100].copy()
        if len(first_100) < 20:
            continue

        qd_10 = _value_at_cycle(sub, "QD", 10)
        qd_100 = _value_at_cycle(sub, "QD", 100)
        qc_10 = _value_at_cycle(sub, "QC", 10)
        qc_100 = _value_at_cycle(sub, "QC", 100)
        ir_10 = _value_at_cycle(sub, "IR", 10)
        ir_100 = _value_at_cycle(sub, "IR", 100)
        tavg_10 = _value_at_cycle(sub, "Tavg", 10)
        tavg_100 = _value_at_cycle(sub, "Tavg", 100)
        tmax_10 = _value_at_cycle(sub, "Tmax", 10)
        tmin_10 = _value_at_cycle(sub, "Tmin", 10)
        chargetime_10 = _value_at_cycle(sub, "chargetime", 10)
        policy_features = _parse_policy(sub["charging_policy"].iloc[0])

        qd_slope = _slope(first_100, "QD")
        qc_slope = _slope(first_100, "QC")
        ir_slope = _slope(first_100, "IR")

        qd_drop_per_100 = -qd_slope * 100 if pd.notna(qd_slope) else np.nan
        qd_retention = _safe_ratio(qd_100, qd_10)
        qc_retention = _safe_ratio(qc_100, qc_10)
        qd_delta = qd_100 - qd_10 if pd.notna(qd_10) and pd.notna(qd_100) else np.nan
        qc_delta = qc_100 - qc_10 if pd.notna(qc_10) and pd.notna(qc_100) else np.nan
        ir_delta = ir_100 - ir_10 if pd.notna(ir_10) and pd.notna(ir_100) else np.nan
        tavg_delta = tavg_100 - tavg_10 if pd.notna(tavg_10) and pd.notna(tavg_100) else np.nan
        chargetime_100_mean = _series_mean(first_100, "chargetime")
        chargetime_delta = (
            chargetime_100_mean - chargetime_10
            if pd.notna(chargetime_10) and pd.notna(chargetime_100_mean)
            else np.nan
        )

        qd_mean = _series_mean(first_100, "QD")
        ir_mean = _series_mean(first_100, "IR")
        tavg_mean = _series_mean(first_100, "Tavg")
        qc_mean = _series_mean(first_100, "QC")

        row = {
            "cell_id": int(cell_id),
            "cycle_life": float(sub["cycle_life"].iloc[0]),
            "Qd_10": qd_10,
            "QC_10": qc_10,
            "IR_10": ir_10,
            "Tavg_10": tavg_10,
            "Tmax_10": tmax_10,
            "Tmin_10": tmin_10,
            "temp_range_10": (
                tmax_10 - tmin_10 if pd.notna(tmax_10) and pd.notna(tmin_10) else np.nan
            ),
            "chargetime_10": chargetime_10,
            "Qd_100": qd_100,
            "Qd_delta_100_10": qd_delta,
            "Qd_retention_100_10": qd_retention,
            "Qd_slope_1_100": qd_slope,
            "Qd_drop_per_100": qd_drop_per_100,
            "Qd_mean_1_100": qd_mean,
            "Qd_std_1_100": _series_std(first_100, "QD"),
            "Qd_resid_std_1_100": _residual_std(first_100, "QD"),
            "IR_100_mean": ir_mean,
            "IR_100_std": _series_std(first_100, "IR"),
            "IR_delta_100_10": ir_delta,
            "IR_slope_1_100": ir_slope,
            "Tavg_100_mean": tavg_mean,
            "Tavg_100_std": _series_std(first_100, "Tavg"),
            "Tavg_delta_100_10": tavg_delta,
            "Tmax_100_mean": _series_mean(first_100, "Tmax"),
            "Tmin_100_mean": _series_mean(first_100, "Tmin"),
            "temp_range_100_mean": _series_mean(first_100.assign(temp_range=first_100["Tmax"] - first_100["Tmin"]), "temp_range"),
            "QC_100": qc_100,
            "QC_delta_100_10": qc_delta,
            "QC_retention_100_10": qc_retention,
            "QC_slope_1_100": qc_slope,
            "chargetime_100_mean": chargetime_100_mean,
            "chargetime_100_std": _series_std(first_100, "chargetime"),
            "chargetime_delta_100_10": chargetime_delta,
            "Qd_QC_ratio_100": _safe_ratio(qd_100, qc_100),
            "coulombic_efficiency_100_mean": _safe_ratio(qd_mean, qc_mean),
            "Qd_drop_x_IR_mean": (
                qd_drop_per_100 * ir_mean
                if pd.notna(qd_drop_per_100) and pd.notna(ir_mean)
                else np.nan
            ),
            "Qd_retention_x_chargetime_mean": (
                qd_retention * chargetime_100_mean
                if pd.notna(qd_retention) and pd.notna(chargetime_100_mean)
                else np.nan
            ),
            "IR_x_Tavg_mean": (
                ir_mean * tavg_mean if pd.notna(ir_mean) and pd.notna(tavg_mean) else np.nan
            ),
            "Qd_delta_x_policy_charge": (
                qd_delta * policy_features["policy_charge_c_first"]
                if pd.notna(qd_delta) and pd.notna(policy_features["policy_charge_c_first"])
                else np.nan
            ),
            "Qd_cv_1_100": _series_cv(first_100, "QD"),
            "IR_cv_1_100": _series_cv(first_100, "IR"),
            "chargetime_cv_1_100": _series_cv(first_100, "chargetime"),
            **policy_features,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def _feature_frame_from_analysis(analysis: dict, variant: str) -> pd.DataFrame:
    df_key = f"{variant}_df"
    if df_key not in analysis:
        raise KeyError(f"Unknown variant={variant!r}. Expected 'raw' or 'filtered'.")

    feature_df = build_question_feature_table(analysis[df_key])
    meta = (
        analysis["df"]
        .drop_duplicates("cell_id")[
            ["cell_id", "global_cell_id", "batch_key", "batch_label", "charging_policy"]
        ]
        .copy()
    )
    return feature_df.merge(meta, on="cell_id", how="left")


def feature_set_columns(feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    if feature_set not in FEATURE_SET_MAP:
        available = ", ".join(sorted(FEATURE_SET_MAP))
        raise KeyError(f"Unknown feature_set={feature_set!r}. Available: {available}")
    return FEATURE_SET_MAP[feature_set]


def load_batch_feature_frame(
    project_root: str | Path,
    batch_key: str,
    variant: str = "filtered",
) -> pd.DataFrame:
    """Load one batch as a cell-level feature table."""
    project_root = Path(project_root).resolve()
    analyses = _analysis_by_batch(project_root)
    if batch_key not in analyses:
        available = ", ".join(sorted(analyses))
        raise KeyError(f"Unknown batch_key={batch_key!r}. Available: {available}")

    return _feature_frame_from_analysis(analyses[batch_key], variant)


def load_train_test_feature_frames(
    project_root: str | Path,
    train_batch_key: str = "batch1",
    test_batch_key: str = "batch2",
    variant: str = "filtered",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test feature tables for the requested batches."""
    project_root = Path(project_root).resolve()
    analyses = _analysis_by_batch(project_root)
    if train_batch_key not in analyses or test_batch_key not in analyses:
        available = ", ".join(sorted(analyses))
        raise KeyError(
            f"Unknown batch key. Requested train={train_batch_key!r}, "
            f"test={test_batch_key!r}. Available: {available}"
        )

    train_df = _feature_frame_from_analysis(analyses[train_batch_key], variant)
    test_df = _feature_frame_from_analysis(analyses[test_batch_key], variant)
    return train_df, test_df
