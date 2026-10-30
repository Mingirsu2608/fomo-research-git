import copy
import logging
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.data.extract import extract
from src.evaluation.evaluate import evaluate_models
from src.features.build import CANDIDATE_32
from src.models.train import train_models
from src.pipelines.common import load_config, run_stage
from src.pipelines.validation import (
    validate_extract, validate_features, validate_cluster, validate_evaluate,
)


@pytest.fixture
def stage_outputs(tmp_path, monkeypatch):
    config = copy.deepcopy(load_config())
    config['data'].update(market_path=str(tmp_path / 'market.csv'),
                          trades_path=str(tmp_path / 'trades.csv'),
                          manifest_path=str(tmp_path / 'manifest.json'))
    config['features']['output_path'] = str(tmp_path / 'features.csv')
    config['model'].update(output_dir=str(tmp_path / 'models'), k_candidates=[2, 3], n_init=1)
    config['evaluation']['output_dir'] = str(tmp_path / 'evaluation')
    config['tracking'].update(uri=f'sqlite:///{tmp_path}/mlflow.db', experiment='checks')
    mlflow.set_tracking_uri(config['tracking']['uri'])
    mlflow.create_experiment('checks', artifact_location=str(tmp_path / 'mlartifacts'))

    def download(_url, output):
        if str(output) == config['data']['market_path']:
            pd.DataFrame({'timestamp': pd.date_range('2025-01-01', periods=3),
                          'open': 100., 'high': 105., 'low': 95.,
                          'close': 101., 'volume': 1000.}).to_csv(output, index=False)
        else:
            pd.DataFrame({'order_id': ['a'], 'timestamp': ['2025-01-01'],
                          'investor_id': ['a'], 'side': ['BUY'], 'price': [100.],
                          'quantity': [1.], 'amount': [100.], 'fee': [1.],
                          'cash_after': [1000.]}).to_csv(output, index=False)
        return Path(output)

    monkeypatch.setattr('src.data.extract.download', download)
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(100, 32)), columns=CANDIDATE_32)
    frame['investor_id'] = [f'investor-{i}' for i in range(len(frame))]
    frame['window_start'] = '2025-01-01'
    frame['window_end'] = '2025-06-29'
    frame['observed_days'] = 180
    frame['n_buys'] = 10
    frame['buy_days'] = 8
    # Exercise complete-case selection in both training and evaluation.
    frame.loc[:4, CANDIDATE_32[0]] = np.nan
    frame.to_csv(config['features']['output_path'], index=False)
    return config, frame


def test_stages_validate_saved_outputs_and_log_pass(stage_outputs, caplog):
    config, frame = stage_outputs
    logger = logging.getLogger('test.pipeline')
    with caplog.at_level(logging.INFO):
        run_stage(logger, 'extract', extract, config)
        run_stage(logger, 'features', lambda _: frame, config)
        tuning = run_stage(logger, 'cluster', train_models, config)
        metrics = run_stage(logger, 'evaluate', evaluate_models, config)
    for name in ['extract', 'features', 'cluster', 'evaluate']:
        assert f'stage={name} status=PASS' in caplog.text
    for name in ['feature_funnel', 'complete_case', 'scaling', 'pca']:
        assert f'step={name} status=PASS' in caplog.text
    assert len(tuning) == 6
    assert set(metrics) == {'kmeans', 'minibatch_kmeans', 'gmm'}


@pytest.mark.parametrize('stage', ['extract', 'features', 'cluster', 'evaluate'])
def test_stage_output_corruption_raises(stage_outputs, stage):
    config, frame = stage_outputs
    if stage == 'extract':
        result = extract(config)
        Path(config['data']['market_path']).write_text('corrupted')
        with pytest.raises(ValueError, match='SHA256'):
            validate_extract(result, config)
    elif stage == 'features':
        frame.loc[0, CANDIDATE_32[1]] = np.inf
        frame.to_csv(config['features']['output_path'], index=False)
        with pytest.raises(ValueError, match='infinite'):
            validate_features(frame, config)
    else:
        tuning = train_models(config)
        if stage == 'cluster':
            report_path = Path(config['model']['output_dir']) / 'pca_variance.csv'
            report = pd.read_csv(report_path)
            report['cumulative_variance'] = 0.
            report.to_csv(report_path, index=False)
            with pytest.raises(ValueError, match='variance below target'):
                validate_cluster(tuning, config)
        else:
            metrics = evaluate_models(config)
            sizes_path = Path(config['evaluation']['output_dir']) / 'cluster_sizes.csv'
            sizes = pd.read_csv(sizes_path)
            sizes.loc[0, 'count'] += 1
            sizes.to_csv(sizes_path, index=False)
            with pytest.raises(ValueError, match='counts differ'):
                validate_evaluate(metrics, config)


def test_all_stops_after_validation_failure(monkeypatch, caplog):
    from src.pipelines import all as pipeline
    from src.pipelines.validation import VALIDATORS
    executed = []
    monkeypatch.setattr(pipeline, 'load_config', lambda: {'as_of_date': '2026-10-29', 'seed': 42})
    monkeypatch.setattr(pipeline, 'extract', lambda _: executed.append('extract'))
    monkeypatch.setattr(pipeline, 'build_features', lambda _: executed.append('features'))

    def reject(*args):
        raise ValueError('Invalid raw schema')

    monkeypatch.setitem(VALIDATORS, 'extract', reject)
    with caplog.at_level(logging.INFO), pytest.raises(ValueError, match='Invalid raw schema'):
        pipeline.main()
    assert executed == ['extract']
    assert 'stage=extract status=FAIL' in caplog.text
    assert 'stage=extract status=PASS' not in caplog.text


def test_pca_validation_failure_is_logged(caplog):
    from src.models.preprocess import fit_preprocessing
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(100, 32)), columns=CANDIDATE_32)
    # Inject an invalid numerical result and verify the PCA guard rejects it.
    from sklearn.decomposition import PCA
    from unittest.mock import patch
    config = load_config()
    with patch.object(PCA, 'fit_transform', return_value=np.full((100, 1), np.nan)):
        with caplog.at_level(logging.INFO), pytest.raises(ValueError, match='PCA produced non-finite'):
            fit_preprocessing(frame, config)
    assert 'step=pca status=FAIL' in caplog.text


@pytest.mark.parametrize('stage,function_name', [
    ('extract', 'extract'), ('features', 'build_features'),
    ('cluster', 'train_models'), ('evaluate', 'evaluate_models'),
])
def test_individual_entrypoints_use_stage_validation(monkeypatch, caplog, stage, function_name):
    import importlib
    from src.pipelines.validation import VALIDATORS
    module = importlib.import_module(f'src.pipelines.{stage}')
    result = object()
    config = object()
    seen = []
    monkeypatch.setattr(module, 'load_config', lambda: config)
    monkeypatch.setattr(module, function_name, lambda cfg: result)
    monkeypatch.setitem(VALIDATORS, stage, lambda output, cfg: seen.append((output, cfg)))
    with caplog.at_level(logging.INFO):
        module.main()
    assert seen == [(result, config)]
    assert f'stage={stage} status=PASS' in caplog.text
