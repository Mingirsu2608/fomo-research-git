import logging
import time

from src.data.extract import extract
from src.evaluation.evaluate import evaluate_models
from src.features.build import build_features
from src.models.train import train_models
from src.pipelines.common import load_config, setup_runtime, run_stage


def main():
    setup_runtime()
    logger = logging.getLogger(__name__)
    config = load_config()
    pipeline_started = time.perf_counter()
    try:
        logger.info(
            "pipeline event=started as_of_date=%s seed=%s",
            config["as_of_date"], config["seed"],
        )
        run_stage(logger, "extract", extract, config)
        run_stage(logger, "features", build_features, config)
        run_stage(logger, "cluster", train_models, config)
        run_stage(logger, "evaluate", evaluate_models, config)
        logger.info(
            "pipeline status=PASS event=completed duration_seconds=%.2f",
            time.perf_counter() - pipeline_started,
        )
    except Exception:
        logger.error(
            "pipeline status=FAIL event=failed duration_seconds=%.2f",
            time.perf_counter() - pipeline_started,
        )
        raise


if __name__ == "__main__":
    main()
