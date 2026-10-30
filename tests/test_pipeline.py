from src.pipelines.common import load_config


def test_config_has_reproducibility_parameters():
    config = load_config()
    assert config["as_of_date"]
    assert config["seed"] == 42
    assert config["features"]["feature_set"] == "fomo_32_v1"
    assert config["model"]["k_candidates"]
