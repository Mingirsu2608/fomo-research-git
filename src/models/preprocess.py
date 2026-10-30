"""Shared feature funnel and fitted scaling/PCA for pipelines and notebooks."""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build import CANDIDATE_32
from src.pipelines.common import checked_step


def select_features(frame, settings):
    """Filter by coverage/dispersion, then greedily remove Spearman redundancy.

    The configured lower-priority features are considered last. Selection uses
    pairwise available observations; complete cases are selected afterwards.
    """
    missing = set(CANDIDATE_32) - set(frame.columns)
    if missing:
        raise ValueError(f"Feature file missing model features: {sorted(missing)}")
    data = frame[CANDIDATE_32].replace([np.inf, -np.inf], np.nan)
    report = pd.DataFrame({
        "coverage": data.notna().mean(),
        "std": data.std(),
        "iqr": data.quantile(.75) - data.quantile(.25),
    })
    report["passed_coverage"] = report.coverage.ge(settings["min_coverage"])
    report["passed_dispersion"] = (
        report["std"].gt(settings["min_std"])
        & report.iqr.gt(settings["min_iqr"])
    )
    eligible = report.index[report.passed_coverage & report.passed_dispersion].tolist()
    correlation = data[eligible].corr(method="spearman")
    lower_priority = settings.get("lower_priority_features", [])
    ordered = [f for f in eligible if f not in lower_priority]
    ordered += [f for f in eligible if f in lower_priority]
    kept = []
    report["redundant_with"] = ""
    for feature in ordered:
        redundant = next((other for other in kept
                          if abs(correlation.loc[feature, other]) >= settings["max_spearman"]), None)
        if redundant is None:
            kept.append(feature)
        else:
            report.loc[feature, "redundant_with"] = redundant
    selected = [f for f in CANDIDATE_32 if f in kept]
    report["selected"] = report.index.isin(selected)
    if not selected:
        raise ValueError("Feature funnel removed all features; review preprocessing thresholds")
    report.index.name = "feature"
    return selected, report


def model_cohort(frame, feature_names):
    """Use only rows with finite values for all selected features, as in the baseline."""
    data = frame[feature_names].replace([np.inf, -np.inf], np.nan)
    cohort = frame.loc[data.notna().all(axis=1)].copy().reset_index(drop=True)
    if len(cohort) < 2:
        raise ValueError("Fewer than two complete investor-windows after feature selection")
    return cohort


def fit_preprocessing(frame, config):
    settings = config["preprocessing"]
    def check_selection(result):
        names, selection_report = result
        if not names or selection_report.loc[selection_report.selected].index.tolist() != names:
            raise ValueError("Feature funnel report differs from selected features")
        corr = frame[names].replace([np.inf, -np.inf], np.nan).corr(method="spearman").abs()
        upper = corr.to_numpy()[np.triu_indices(len(names), k=1)]
        if np.any(upper >= settings["max_spearman"]):
            raise ValueError("Feature funnel retained redundant features")

    feature_names, report = checked_step(
        "feature_funnel", lambda: select_features(frame, settings), check_selection,
    )
    cohort = checked_step(
        "complete_case", lambda: model_cohort(frame, feature_names),
        lambda result: None,  # model_cohort validates the minimum cohort size.
    )
    target = settings["pca_variance_target"]
    if not 0 < target < 1:
        raise ValueError("pca_variance_target must be between 0 and 1")
    scaler = StandardScaler()

    def check_scaled(values):
        if values.shape != (len(cohort), len(feature_names)) or not np.isfinite(values).all():
            raise ValueError("Scaling produced invalid shape or non-finite values")
        if not np.allclose(values.mean(axis=0), 0, atol=1e-7):
            raise ValueError("Scaled features are not centered")
        variance = scaler.var_
        varying = variance > 1e-12
        if not np.allclose(values[:, varying].std(axis=0), 1, atol=1e-7):
            raise ValueError("Scaled features do not have unit standard deviation")

    scaled = checked_step("scaling", lambda: scaler.fit_transform(cohort[feature_names]), check_scaled)
    pca = PCA(n_components=target, svd_solver="full", random_state=config["seed"])

    def check_pca(values):
        if values.shape[0] != len(cohort) or not 1 <= values.shape[1] <= len(feature_names):
            raise ValueError("PCA produced invalid shape")
        if not np.isfinite(values).all() or not np.isfinite(pca.explained_variance_ratio_).all():
            raise ValueError("PCA produced non-finite values or variance")
        if pca.explained_variance_ratio_.sum() + 1e-10 < target:
            raise ValueError("PCA retained variance is below configured target")

    X = checked_step("pca", lambda: pca.fit_transform(scaled), check_pca)
    preprocessor = Pipeline([("scaler", scaler), ("pca", pca)])
    return cohort, feature_names, report, preprocessor, X
