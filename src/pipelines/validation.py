"""Runtime checks for stage outputs. Raise ValueError to stop invalid runs."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.extract import sha256
from src.features.build import CANDIDATE_32, REQUIRED_MARKET_COLUMNS, REQUIRED_TRADE_COLUMNS
from src.models.preprocess import model_cohort


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_extract(result, config):
    data = config['data']
    manifest = json.loads(Path(data['manifest_path']).read_text())
    require(manifest == result, 'Saved manifest differs from extract result')
    require(manifest['as_of_date'] == config['as_of_date'], 'Manifest as_of_date mismatch')
    for name, required in [('market', REQUIRED_MARKET_COLUMNS), ('trades', REQUIRED_TRADE_COLUMNS)]:
        entry = manifest['files'][name]
        path = Path(data[f'{name}_path'])
        require(str(path) == entry['path'], f'{name}: manifest path mismatch')
        require(sha256(path) == entry['sha256'], f'{name}: SHA256 mismatch')
        frame = pd.read_csv(path)
        require(not frame.empty, f'{name}: raw data is empty')
        require(required <= set(frame.columns), f'{name}: missing columns {sorted(required - set(frame.columns))}')
        dates = pd.to_datetime(frame['timestamp'], errors='raise')
        require(dates.notna().all(), f'{name}: missing timestamps')
        if name == 'market':
            days = dates.dt.normalize().sort_values()
            require(not days.duplicated().any(), 'market: duplicate dates')
            require(len(days) == len(pd.date_range(days.min(), days.max())), 'market: missing calendar dates')
        else:
            require(frame['investor_id'].notna().all(), 'trades: missing investor_id')
            require(frame['side'].str.strip().str.upper().isin(['BUY', 'SELL']).all(), 'trades: invalid side')


def validate_features(result, config):
    frame = pd.read_csv(config['features']['output_path'])
    required = set(CANDIDATE_32) | {'investor_id', 'window_start', 'window_end', 'observed_days', 'n_buys', 'buy_days'}
    require(required <= set(frame.columns), f'features: missing columns {sorted(required - set(frame.columns))}')
    require(not frame.empty and len(frame) == len(result), 'features: empty output or saved row count mismatch')
    require(not frame[['investor_id', 'window_start', 'window_end']].isna().any().any(), 'features: missing window identifiers')
    require(not frame.duplicated(['investor_id', 'window_start', 'window_end']).any(), 'features: duplicate investor-window')
    require(not np.isinf(frame[CANDIDATE_32].to_numpy(dtype=float)).any(), 'features: infinite values')
    settings = config['features']
    days = settings.get('window_days', 180)
    length = (pd.to_datetime(frame.window_end) - pd.to_datetime(frame.window_start)).dt.days + 1
    require(length.eq(days).all() and frame.observed_days.eq(days).all(), 'features: invalid window length')
    require(frame.n_buys.ge(settings.get('min_buys', settings.get('min_trades', 5))).all(), 'features: insufficient buys')
    require(frame.buy_days.ge(settings.get('min_buy_days', 3)).all(), 'features: insufficient buy days')


def validate_cluster(result, config):
    directory = Path(config['model']['output_dir'])
    names = (directory / 'features.txt').read_text().splitlines()
    require(bool(names) and len(names) == len(set(names)) and set(names) <= set(CANDIDATE_32), 'cluster: invalid selected features')
    report = pd.read_csv(directory / 'feature_selection.csv')
    require(report.loc[report.selected, 'feature'].tolist() == names, 'cluster: feature report differs from features.txt')
    frame = model_cohort(pd.read_csv(config['features']['output_path']), names)
    pipeline = joblib.load(directory / 'preprocessor.joblib')
    X = pipeline.transform(frame[names])
    require(np.isfinite(X).all(), 'cluster: non-finite transformed data')
    pca_report = pd.read_csv(directory / 'pca_variance.csv')
    require(len(pca_report) == X.shape[1], 'cluster: PCA dimension mismatch')
    require(pca_report.cumulative_variance.iloc[-1] + 1e-10 >= config['preprocessing']['pca_variance_target'], 'cluster: PCA variance below target')
    tuning = pd.read_csv(directory / 'tuning_results.csv')
    expected = {(name, k) for name in config['model']['algorithms'] for k in config['model']['k_candidates']}
    require(set(zip(tuning.model, tuning.k)) == expected and len(tuning) == len(expected), 'cluster: missing/duplicate tuning candidates')
    require(len(tuning) == len(result) and np.isfinite(tuning.silhouette).all(), 'cluster: invalid tuning scores')
    for name in config['model']['algorithms']:
        model = joblib.load(directory / f'{name}.joblib')
        labels = model.predict(X)
        best = tuning.loc[tuning.model.eq(name)].sort_values(['silhouette', 'k'], ascending=[False, True]).iloc[0]
        require(getattr(model, 'n_clusters', getattr(model, 'n_components', None)) == best.k, f'{name}: saved model K differs from tuning winner')
        require(len(labels) == len(frame) and 2 <= len(np.unique(labels)) < len(frame), f'{name}: invalid cluster labels')


def validate_evaluate(result, config):
    directory = Path(config['evaluation']['output_dir'])
    metrics = json.loads((directory / 'model_metrics.json').read_text())
    require(metrics == result, 'evaluate: saved metrics differ from result')
    algorithms = set(config['model']['algorithms'])
    require(set(metrics) == algorithms, 'evaluate: missing model metrics')
    for name, values in metrics.items():
        require(set(values) == {'silhouette', 'calinski_harabasz', 'davies_bouldin'}, f'{name}: missing metrics')
        require(np.isfinite(list(values.values())).all(), f'{name}: non-finite metrics')
        require(-1 <= values['silhouette'] <= 1 and values['calinski_harabasz'] >= 0 and values['davies_bouldin'] >= 0, f'{name}: metrics out of range')
    sizes = pd.read_csv(directory / 'cluster_sizes.csv')
    profiles = pd.read_csv(directory / 'cluster_profiles.csv')
    require(set(sizes.model) == algorithms and set(profiles.model) == algorithms, 'evaluate: missing cluster outputs')
    model_dir = Path(config['model']['output_dir'])
    names = (model_dir / 'features.txt').read_text().splitlines()
    cohort = model_cohort(pd.read_csv(config['features']['output_path']), names)
    require(set(names) <= set(profiles.columns), 'evaluate: missing profile features')
    for name, group in sizes.groupby('model'):
        require(not group.cluster.duplicated().any() and group['count'].gt(0).all(), f'{name}: invalid cluster counts')
        require(group['count'].sum() == len(cohort), f'{name}: cluster counts differ from cohort')
        require(np.allclose(group.share, group['count'] / len(cohort)), f'{name}: invalid cluster shares')
        profile = profiles.loc[profiles.model.eq(name)]
        require(not profile.cluster.duplicated().any() and set(profile.cluster) == set(group.cluster), f'{name}: cluster profiles mismatch')
        require(np.isfinite(profile[names].to_numpy()).all(), f'{name}: non-finite profiles')


VALIDATORS = {'extract': validate_extract, 'features': validate_features,
              'cluster': validate_cluster, 'evaluate': validate_evaluate}
