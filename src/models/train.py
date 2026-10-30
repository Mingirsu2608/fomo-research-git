from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from src.models.preprocess import fit_preprocessing


NON_MODEL_COLUMNS = {"investor_id", "window_start", "window_end"}


def _score(X, labels, seed):
    rng = np.random.default_rng(seed)
    index = rng.choice(len(X), min(3000, len(X)), replace=False)
    return float(silhouette_score(X[index], np.asarray(labels)[index]))


def train_models(config):
    frame = pd.read_csv(config["features"]["output_path"])
    cohort, feature_names, feature_report, preprocessor, X = fit_preprocessing(frame, config)
    settings = config["model"]
    seed = config["seed"]
    if any(k < 2 or k >= len(cohort) for k in settings["k_candidates"]):
        raise ValueError("Each candidate K must be >= 2 and smaller than the complete-case cohort")
    rows, candidates = [], {}

    for name in settings["algorithms"]:
        for k in settings["k_candidates"]:
            if name == "kmeans":
                model = KMeans(
                    n_clusters=k, n_init=settings["n_init"],
                    max_iter=settings["max_iter"], random_state=seed,
                )
            elif name == "minibatch_kmeans":
                model = MiniBatchKMeans(
                    n_clusters=k, n_init=settings["n_init"], batch_size=1024,
                    max_iter=settings["max_iter"], random_state=seed,
                )
            elif name == "gmm":
                model = GaussianMixture(
                    n_components=k, covariance_type="diag", n_init=2,
                    max_iter=settings["max_iter"], random_state=seed,
                )
            else:
                raise ValueError(f"Unsupported algorithm: {name}")

            labels = model.fit_predict(X)
            score = _score(X, labels, seed)
            rows.append({"model": name, "k": k, "silhouette": score})
            candidates[(name, k)] = model

    tuning = pd.DataFrame(rows)
    output_dir = Path(settings["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_report.to_csv(output_dir / "feature_selection.csv")
    pca = preprocessor.named_steps["pca"]
    pd.DataFrame({
        "component": np.arange(1, pca.n_components_ + 1),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(output_dir / "pca_variance.csv", index=False)
    joblib.dump(preprocessor, output_dir / "preprocessor.joblib")
    (output_dir / "features.txt").write_text("\n".join(feature_names), encoding="utf-8")

    mlflow.set_tracking_uri(config["tracking"]["uri"])
    mlflow.set_experiment(config["tracking"]["experiment"])
    with mlflow.start_run(run_name="cluster"):
        mlflow.log_param("as_of_date", config["as_of_date"])
        mlflow.log_param("feature_set", config["features"]["feature_set"])
        mlflow.log_param("seed", seed)
        mlflow.log_params({f"preprocessing_{key}": value
                           for key, value in config["preprocessing"].items()})
        mlflow.log_params({"selected_features": len(feature_names),
                           "pca_components": pca.n_components_,
                           "cohort_rows": len(cohort),
                           "excluded_rows": len(frame) - len(cohort)})
        mlflow.log_artifact(str(output_dir / "feature_selection.csv"))
        mlflow.log_artifact(str(output_dir / "pca_variance.csv"))
        for name in settings["algorithms"]:
            best = tuning.loc[tuning["model"].eq(name)].sort_values(
                ["silhouette", "k"], ascending=[False, True]
            ).iloc[0]
            model = candidates[(name, int(best["k"]))]
            joblib.dump(model, output_dir / f"{name}.joblib")
            mlflow.log_metric(f"{name}_silhouette", float(best["silhouette"]))
            mlflow.log_param(f"{name}_k", int(best["k"]))

        tuning.to_csv(output_dir / "tuning_results.csv", index=False)
        mlflow.log_artifact(str(output_dir / "tuning_results.csv"))
    return tuning
