"""Training helpers for Batch 1 -> Batch 2 early-life battery experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import NuSVR, SVR

try:
    from .features import (
        DEFAULT_FEATURE_SET,
        FEATURE_SET_MAP,
        HONEST_LOCAL_REFIT_BASE_FEATURES,
        HONEST_LOCAL_REFIT_EXTRA_POOL,
        QUESTION_FEATURE_BLOCKS,
        feature_set_columns,
        load_train_test_feature_frames,
    )
    from .preprocess import (
        build_preprocessor,
        inverse_transform_target,
        prepare_train_test_frames,
        prepare_xy_arrays,
        transform_target,
    )
except ImportError:  # pragma: no cover - fallback for direct script execution
    from features import (
        DEFAULT_FEATURE_SET,
        FEATURE_SET_MAP,
        HONEST_LOCAL_REFIT_BASE_FEATURES,
        HONEST_LOCAL_REFIT_EXTRA_POOL,
        QUESTION_FEATURE_BLOCKS,
        feature_set_columns,
        load_train_test_feature_frames,
    )
    from preprocess import (
        build_preprocessor,
        inverse_transform_target,
        prepare_train_test_frames,
        prepare_xy_arrays,
        transform_target,
    )


def build_elastic_net_pipeline(
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """Build an ElasticNetCV pipeline."""
    categorical_features = categorical_features or []
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        alphas=np.logspace(-4, 2, 120),
        cv=KFold(n_splits=5, shuffle=True, random_state=random_state),
        max_iter=200000,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return standard regression metrics on the original target scale."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape_pct = float(mean_absolute_percentage_error(y_true, y_pred) * 100)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "mape_pct": mape_pct, "r2": r2}


def extract_coefficients(
    pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Return fitted coefficients with feature names."""
    categorical_features = categorical_features or []
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coef_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "coefficient": model.coef_,
        }
    )
    return coef_df.sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)


def train_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    target_col: str = "cycle_life",
    log_target: bool = True,
    random_state: int = 42,
) -> dict[str, object]:
    """Train ElasticNet on train_df and evaluate on test_df."""
    numeric_features = numeric_features or feature_set_columns(DEFAULT_FEATURE_SET)
    categorical_features = categorical_features or []

    X_train, y_train_fit, X_test, y_test_fit = prepare_train_test_frames(
        train_df=train_df,
        test_df=test_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_col=target_col,
        log_target=log_target,
    )

    pipeline = build_elastic_net_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train_fit)

    train_pred_fit = pipeline.predict(X_train)
    test_pred_fit = pipeline.predict(X_test)

    y_train = inverse_transform_target(y_train_fit, log_target)
    y_test = inverse_transform_target(y_test_fit, log_target)
    train_pred = inverse_transform_target(train_pred_fit, log_target)
    test_pred = inverse_transform_target(test_pred_fit, log_target)

    coefficients = extract_coefficients(
        pipeline,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    predictions = test_df.dropna(subset=[target_col]).copy()[
        ["cell_id", "global_cell_id", "batch_key", "batch_label", target_col]
    ]
    predictions = predictions.rename(columns={target_col: "y_true"})
    predictions["y_pred"] = test_pred
    predictions["error"] = predictions["y_pred"] - predictions["y_true"]
    predictions["abs_error"] = predictions["error"].abs()

    model = pipeline.named_steps["model"]
    summary = {
        "target_col": target_col,
        "log_target": log_target,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
        "train_metrics": evaluate_predictions(y_train, train_pred),
        "test_metrics": evaluate_predictions(y_test, test_pred),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

    return {
        "pipeline": pipeline,
        "summary": summary,
        "coefficients": coefficients,
        "predictions": predictions,
    }


def run_batch_experiment(
    project_root: str | Path,
    train_batch_key: str = "batch1",
    test_batch_key: str = "batch2",
    variant: str = "filtered",
    include_policy: bool = False,
    feature_set: str = DEFAULT_FEATURE_SET,
    log_target: bool = True,
) -> dict[str, object]:
    """Run the standard Batch 1 train / Batch 2 test experiment."""
    project_root = Path(project_root).resolve()
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_train_test_feature_frames(
        project_root=project_root,
        train_batch_key=train_batch_key,
        test_batch_key=test_batch_key,
        variant=variant,
    )
    numeric_features = feature_set_columns(feature_set)
    categorical_features = ["charging_policy"] if include_policy else []
    result = train_model(
        train_df=train_df,
        test_df=test_df,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        log_target=log_target,
    )

    target_tag = "logtarget" if log_target else "rawtarget"
    policy_tag = "withpolicy" if include_policy else "nopolicy"
    stem = (
        f"{train_batch_key}_train_{test_batch_key}_test_"
        f"{feature_set}_{variant}_{policy_tag}_{target_tag}_elasticnet"
    )
    predictions_path = results_dir / f"{stem}_predictions.csv"
    coefficients_path = results_dir / f"{stem}_coefficients.csv"
    train_features_path = results_dir / f"{stem}_train_features.csv"
    test_features_path = results_dir / f"{stem}_test_features.csv"
    summary_path = results_dir / f"{stem}_summary.json"

    result["predictions"].to_csv(predictions_path, index=False)
    result["coefficients"].to_csv(coefficients_path, index=False)
    train_df.to_csv(train_features_path, index=False)
    test_df.to_csv(test_features_path, index=False)
    summary = {
        **result["summary"],
        "train_batch_key": train_batch_key,
        "test_batch_key": test_batch_key,
        "variant": variant,
        "include_policy": include_policy,
        "feature_set": feature_set,
        "target_transform": "log" if log_target else "none",
        "question_feature_blocks": QUESTION_FEATURE_BLOCKS,
        "predictions_path": str(predictions_path),
        "coefficients_path": str(coefficients_path),
        "train_features_path": str(train_features_path),
        "test_features_path": str(test_features_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    result["summary"] = summary
    result["summary_path"] = summary_path
    return result


NONLINEAR_MODEL_SPECS = {
    "svr": {
        "scaler": "robust",
        "builder": lambda: SVR(kernel="rbf", C=20.0, epsilon=0.05, gamma="scale"),
    },
    "nusvr_a": {
        "scaler": "robust",
        "builder": lambda: NuSVR(kernel="rbf", C=30.0, nu=0.25, gamma="scale"),
    },
    "nusvr_b": {
        "scaler": "robust",
        "builder": lambda: NuSVR(kernel="rbf", C=50.0, nu=0.20, gamma="scale"),
    },
    "kr": {
        "scaler": "standard",
        "builder": lambda: KernelRidge(alpha=0.1, kernel="rbf", gamma=0.2),
    },
}


def available_model_keys() -> list[str]:
    """Return all supported nonlinear model keys including simple ensembles."""
    return [*NONLINEAR_MODEL_SPECS.keys(), "ens_nu", "ens_svr_kr"]


def fit_predict_nonlinear(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model_key: str,
    target_col: str = "cycle_life",
    log_target: bool = True,
) -> np.ndarray:
    """Fit one nonlinear model on train_df and predict on test_df."""
    if model_key == "ens_nu":
        pred_a = fit_predict_nonlinear(
            train_df, test_df, features, "nusvr_a", target_col=target_col, log_target=log_target
        )
        pred_b = fit_predict_nonlinear(
            train_df, test_df, features, "nusvr_b", target_col=target_col, log_target=log_target
        )
        return 0.75 * pred_a + 0.25 * pred_b
    if model_key == "ens_svr_kr":
        pred_a = fit_predict_nonlinear(
            train_df, test_df, features, "svr", target_col=target_col, log_target=log_target
        )
        pred_b = fit_predict_nonlinear(
            train_df, test_df, features, "kr", target_col=target_col, log_target=log_target
        )
        return 0.5 * pred_a + 0.5 * pred_b
    if model_key not in NONLINEAR_MODEL_SPECS:
        available = ", ".join(sorted(available_model_keys()))
        raise KeyError(f"Unknown model_key={model_key!r}. Available: {available}")

    clean_train = train_df.dropna(subset=[target_col]).copy()
    clean_test = test_df.dropna(subset=[target_col]).copy()
    spec = NONLINEAR_MODEL_SPECS[model_key]
    X_train, X_test = prepare_xy_arrays(
        clean_train,
        clean_test,
        feature_cols=features,
        scaler=spec["scaler"],
    )
    y_train = transform_target(clean_train[target_col].to_numpy(dtype=float), log_target=log_target)
    model = spec["builder"]()
    model.fit(X_train, y_train)
    pred_fit = model.predict(X_test)
    return inverse_transform_target(pred_fit, log_target=log_target)


def select_best_model_by_group_cv(
    train_df: pd.DataFrame,
    features: list[str],
    group_col: str = "charging_policy",
    model_keys: list[str] | None = None,
    target_col: str = "cycle_life",
    log_target: bool = True,
) -> dict[str, float | str]:
    """Select the best nonlinear model using GroupKFold CV."""
    model_keys = model_keys or available_model_keys()
    groups = train_df[group_col].astype(str).to_numpy()
    n_groups = pd.Series(groups).nunique()
    splitter = GroupKFold(n_splits=min(5, max(2, n_groups)))

    best: dict[str, float | str] | None = None
    for model_key in model_keys:
        fold_scores = []
        for tr_idx, va_idx in splitter.split(train_df, groups=groups):
            tr = train_df.iloc[tr_idx].copy()
            va = train_df.iloc[va_idx].copy()
            pred = fit_predict_nonlinear(
                tr,
                va,
                features,
                model_key=model_key,
                target_col=target_col,
                log_target=log_target,
            )
            score = mean_absolute_percentage_error(
                va[target_col].to_numpy(dtype=float),
                np.clip(pred, 1e-6, None),
            ) * 100.0
            fold_scores.append(float(score))

        row: dict[str, float | str] = {
            "cv_model_key": model_key,
            "cv_mean_mape_pct": float(np.mean(fold_scores)),
            "cv_std_mape_pct": float(np.std(fold_scores, ddof=0)),
        }
        if best is None or (row["cv_mean_mape_pct"], row["cv_std_mape_pct"]) < (
            best["cv_mean_mape_pct"],
            best["cv_std_mape_pct"],
        ):
            best = row

    if best is None:
        raise RuntimeError("No model candidates were evaluated.")
    return best


def build_candidate_specs(
    base_features: list[str] | None = None,
    extra_pool: list[str] | None = None,
) -> list[dict[str, object]]:
    """Generate add/remove/swap feature candidates from the notebook workflow."""
    base_features = list(base_features or HONEST_LOCAL_REFIT_BASE_FEATURES)
    extra_pool = list(extra_pool or HONEST_LOCAL_REFIT_EXTRA_POOL)
    candidates: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()

    def add_candidate(name: str, features: list[str]) -> None:
        key = tuple(sorted(features))
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "candidate_name": name,
                "features": list(features),
                "feature_count": len(features),
            }
        )

    add_candidate("base", base_features)
    for feature in base_features:
        add_candidate(f"remove:{feature}", [f for f in base_features if f != feature])
    for feature in extra_pool:
        add_candidate(f"add:{feature}", base_features + [feature])
    for remove_feature in base_features:
        for add_feature in extra_pool:
            proposal = [f for f in base_features if f != remove_feature] + [add_feature]
            add_candidate(f"swap:{remove_feature}->{add_feature}", proposal)

    return candidates


def run_honest_local_refit(
    project_root: str | Path,
    train_batch_key: str = "batch1",
    test_batch_key: str = "batch2",
    variant: str = "filtered",
    target_col: str = "cycle_life",
    group_col: str = "charging_policy",
    log_target: bool = True,
    random_state: int = 42,
) -> dict[str, object]:
    """Run the notebook-style honest local refit workflow on train/test batches."""
    train_df, test_df = load_train_test_feature_frames(
        project_root=project_root,
        train_batch_key=train_batch_key,
        test_batch_key=test_batch_key,
        variant=variant,
    )
    train_df = train_df.dropna(subset=[target_col]).copy().reset_index(drop=True)
    test_df = test_df.dropna(subset=[target_col]).copy().reset_index(drop=True)
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    test_df = test_df.replace([np.inf, -np.inf], np.nan)

    groups = train_df[group_col].astype(str).to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    train_idx, valid_idx = next(gss.split(train_df, groups=groups))
    train_sub = train_df.iloc[train_idx].copy().reset_index(drop=True)
    valid_sub = train_df.iloc[valid_idx].copy().reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for spec in build_candidate_specs():
        features = list(spec["features"])
        cv = select_best_model_by_group_cv(
            train_sub,
            features,
            group_col=group_col,
            target_col=target_col,
            log_target=log_target,
        )
        valid_pred = fit_predict_nonlinear(
            train_sub,
            valid_sub,
            features,
            model_key=str(cv["cv_model_key"]),
            target_col=target_col,
            log_target=log_target,
        )
        valid_mape = mean_absolute_percentage_error(
            valid_sub[target_col].to_numpy(dtype=float),
            np.clip(valid_pred, 1e-6, None),
        ) * 100.0
        rows.append(
            {
                "candidate_name": spec["candidate_name"],
                "features": features,
                "feature_count": len(features),
                **cv,
                "valid_mape_pct": float(valid_mape),
            }
        )

    candidates = pd.DataFrame(rows).sort_values(
        ["valid_mape_pct", "cv_mean_mape_pct", "cv_std_mape_pct", "feature_count"]
    ).reset_index(drop=True)
    best = candidates.iloc[0].to_dict()
    selected_features = list(best["features"])
    selected_model = str(best["cv_model_key"])
    test_pred = fit_predict_nonlinear(
        train_df,
        test_df,
        selected_features,
        model_key=selected_model,
        target_col=target_col,
        log_target=log_target,
    )
    test_metrics = evaluate_predictions(
        test_df[target_col].to_numpy(dtype=float),
        np.clip(test_pred, 1e-6, None),
    )
    predictions = test_df[["cell_id", "global_cell_id", "batch_key", "batch_label", group_col, target_col]].copy()
    predictions = predictions.rename(columns={target_col: "y_true"})
    predictions["y_pred"] = np.clip(test_pred, 1e-6, None)
    predictions["abs_error"] = (predictions["y_pred"] - predictions["y_true"]).abs()
    predictions["ape_pct"] = predictions["abs_error"] / predictions["y_true"] * 100.0

    summary = {
        "train_batch_key": train_batch_key,
        "test_batch_key": test_batch_key,
        "variant": variant,
        "target_transform": "log1p" if log_target else "none",
        "selected_candidate": best["candidate_name"],
        "selected_features": selected_features,
        "selected_model": selected_model,
        "train_cv_mape_pct": float(best["cv_mean_mape_pct"]),
        "train_cv_std_mape_pct": float(best["cv_std_mape_pct"]),
        "valid_mape_pct": float(best["valid_mape_pct"]),
        "test_mape_pct": float(test_metrics["mape_pct"]),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_r2": float(test_metrics["r2"]),
        "gap_train_valid_pp": float(best["valid_mape_pct"] - best["cv_mean_mape_pct"]),
        "gap_valid_test_pp": float(test_metrics["mape_pct"] - best["valid_mape_pct"]),
        "group_col": group_col,
        "candidate_count": int(len(candidates)),
    }
    return {
        "summary": summary,
        "candidates": candidates,
        "predictions": predictions.sort_values("ape_pct", ascending=False).reset_index(drop=True),
        "train_sub": train_sub,
        "valid_sub": valid_sub,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ElasticNet on Batch 1 and test on Batch 2.")
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[1]),
        help="Project root containing data/, src/, and results/.",
    )
    parser.add_argument("--train-batch", default="batch1", help="Training batch key.")
    parser.add_argument("--test-batch", default="batch2", help="Test batch key.")
    parser.add_argument(
        "--variant",
        default="filtered",
        choices=["filtered", "raw"],
        help="Use filtered or raw early-life features.",
    )
    parser.add_argument(
        "--include-policy",
        action="store_true",
        help="Include charging_policy as a categorical feature.",
    )
    parser.add_argument(
        "--feature-set",
        default=DEFAULT_FEATURE_SET,
        choices=sorted(FEATURE_SET_MAP),
        help="Feature preset to use for training.",
    )
    parser.add_argument(
        "--no-log-target",
        action="store_true",
        help="Use raw cycle_life instead of log(cycle_life).",
    )
    args = parser.parse_args()

    result = run_batch_experiment(
        project_root=args.project_root,
        train_batch_key=args.train_batch,
        test_batch_key=args.test_batch,
        variant=args.variant,
        include_policy=args.include_policy,
        feature_set=args.feature_set,
        log_target=not args.no_log_target,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
