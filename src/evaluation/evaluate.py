import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from src.models.preprocess import model_cohort


def evaluate_models(config):
    frame = pd.read_csv(config["features"]["output_path"])
    model_dir = Path(config["model"]["output_dir"])
    feature_names = (model_dir / "features.txt").read_text(encoding="utf-8").splitlines()
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    frame = model_cohort(frame, feature_names)
    X = preprocessor.transform(frame[feature_names])

    metrics, sizes, profiles = {}, [], []
    for name in config["model"]["algorithms"]:
        model = joblib.load(model_dir / f"{name}.joblib")
        labels = model.predict(X)
        metrics[name] = {
            "silhouette": float(silhouette_score(X, labels, sample_size=min(3000, len(X)))),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
        }
        labeled = frame.assign(model=name, cluster=labels)
        size = labeled.groupby(["model", "cluster"]).size().rename("count").reset_index()
        size["share"] = size["count"] / len(labeled)
        sizes.append(size)
        profile = labeled.groupby(["model", "cluster"])[feature_names].mean().reset_index()
        profiles.append(profile)

    output = Path(config["evaluation"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.concat(sizes, ignore_index=True).to_csv(output / "cluster_sizes.csv", index=False)
    pd.concat(profiles, ignore_index=True).to_csv(
        output / "cluster_profiles.csv", index=False
    )
    return metrics

