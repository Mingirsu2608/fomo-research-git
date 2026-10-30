import pandas as pd
import pytest

from src.features.build import CANDIDATE_32, build_features


def test_feature_job_rejects_invalid_schema(tmp_path):
    source = tmp_path / "trades.csv"
    pd.DataFrame({"investor_id": ["a"]}).to_csv(source, index=False)
    config = {
        "data": {"market_path": str(source), "trades_path": str(source)},
        "features": {"min_trades": 1, "output_path": str(tmp_path / "out.csv")},
    }
    with pytest.raises(ValueError, match="Market data missing columns"):
        build_features(config)


def test_feature_contract_contains_32_unique_features():
    assert len(CANDIDATE_32) == 32
    assert len(set(CANDIDATE_32)) == 32
