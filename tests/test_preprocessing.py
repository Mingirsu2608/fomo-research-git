import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.build import CANDIDATE_32
from src.models.preprocess import fit_preprocessing, model_cohort, select_features


def fixture_frame():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(0., index=range(120), columns=CANDIDATE_32)
    frame[CANDIDATE_32[0]] = rng.normal(size=120)
    frame[CANDIDATE_32[1]] = frame[CANDIDATE_32[0]] * 2
    frame[CANDIDATE_32[2]] = rng.normal(size=120)
    frame.loc[:4, CANDIDATE_32[2]] = np.nan
    frame[CANDIDATE_32[3]] = rng.normal(size=120)
    frame.loc[:49, CANDIDATE_32[3]] = np.nan
    return frame


def settings():
    return dict(min_coverage=.8, min_std=.02, min_iqr=.02,
                max_spearman=.7, pca_variance_target=.9,
                lower_priority_features=[CANDIDATE_32[1]])


def test_funnel_filters_missing_constant_and_correlated_features():
    frame = fixture_frame()
    selected, report = select_features(frame, settings())
    assert selected == [CANDIDATE_32[0], CANDIDATE_32[2]]
    assert report.loc[CANDIDATE_32[1], 'redundant_with'] == CANDIDATE_32[0]
    assert not report.loc[CANDIDATE_32[3], 'passed_coverage']
    assert not report.loc[CANDIDATE_32[4], 'passed_dispersion']
    # Missing values in features removed by the funnel do not remove rows.
    assert len(model_cohort(frame, selected)) == 115


def test_saved_scaling_pca_reuses_training_transform(tmp_path):
    config = {'seed': 42, 'preprocessing': settings()}
    frame = fixture_frame()
    cohort, features, _, pipeline, X = fit_preprocessing(frame, config)
    np.testing.assert_allclose(pipeline['scaler'].transform(cohort[features]).mean(axis=0), 0, atol=1e-12)
    assert pipeline['pca'].explained_variance_ratio_.sum() >= .9
    joblib.dump(pipeline, tmp_path / 'preprocessor.joblib')
    restored = joblib.load(tmp_path / 'preprocessor.joblib')
    np.testing.assert_allclose(restored.transform(model_cohort(frame, features)[features]), X, atol=1e-12)


def test_empty_funnel_has_clear_error():
    frame = pd.DataFrame(0., index=range(10), columns=CANDIDATE_32)
    with pytest.raises(ValueError, match='removed all features'):
        select_features(frame, settings())
