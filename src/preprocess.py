"""Preprocessing helpers for early-life battery modeling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build a preprocessing pipeline for mixed feature types."""
    categorical_features = categorical_features or []

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("At least one feature column is required.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_numeric_pipeline(scaler: str = "standard") -> Pipeline:
    """Build a numeric-only preprocessing pipeline used by nonlinear models."""
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaler == "standard":
        steps.append(("scaler", StandardScaler()))
    elif scaler == "robust":
        steps.append(("scaler", RobustScaler()))
    elif scaler != "none":
        raise ValueError(f"Unknown scaler={scaler!r}. Expected standard, robust, or none.")
    return Pipeline(steps=steps)


def transform_target(values: np.ndarray, log_target: bool = True) -> np.ndarray:
    """Transform target values on the fit scale."""
    values = np.asarray(values, dtype=float)
    if log_target:
        return np.log1p(values)
    return values


def inverse_transform_target(values: np.ndarray, log_target: bool = True) -> np.ndarray:
    """Invert target values back to the original cycle-life scale."""
    values = np.asarray(values, dtype=float)
    if log_target:
        return np.expm1(values)
    return values


def prepare_xy_arrays(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    scaler: str = "robust",
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare aligned numeric arrays for train/test evaluation."""
    numeric_pipeline = build_numeric_pipeline(scaler=scaler)
    X_train = numeric_pipeline.fit_transform(train_df[feature_cols])
    X_test = numeric_pipeline.transform(test_df[feature_cols])
    return X_train, X_test


def prepare_train_test_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    target_col: str = "cycle_life",
    log_target: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Extract aligned train/test frames and transformed targets."""
    categorical_features = categorical_features or []
    feature_cols = [*numeric_features, *categorical_features]

    train = train_df.dropna(subset=[target_col]).copy()
    test = test_df.dropna(subset=[target_col]).copy()

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    y_train = train[target_col].to_numpy(dtype=float)
    y_test = test[target_col].to_numpy(dtype=float)

    y_train = transform_target(y_train, log_target=log_target)
    y_test = transform_target(y_test, log_target=log_target)

    return X_train, y_train, X_test, y_test
