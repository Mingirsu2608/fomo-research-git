import logging
import os
import warnings
from pathlib import Path

import yaml


def load_config(path="params.yaml"):
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_runtime():
    warnings.filterwarnings("ignore")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    for name in ["alembic", "mlflow", "optuna", "sqlalchemy", "urllib3"]:
        logging.getLogger(name).setLevel(logging.WARNING)



def run_stage(logger, name, function, config):
    """Run a stage, validate its saved outputs, and fail before the next stage."""
    import time
    from src.pipelines.validation import VALIDATORS

    started = time.perf_counter()
    logger.info("stage=%s event=started", name)
    try:
        result = function(config)
        VALIDATORS[name](result, config)
    except Exception:
        logger.exception("stage=%s status=FAIL event=failed", name)
        raise
    logger.info("stage=%s status=PASS event=completed duration_seconds=%.2f",
                name, time.perf_counter() - started)
    return result


def checked_step(name, operation, validate):
    """Log explicit PASS/FAIL for a preprocessing substep."""
    logger = logging.getLogger(__name__)
    try:
        result = operation()
        validate(result)
    except Exception:
        logger.exception("step=%s status=FAIL", name)
        raise
    logger.info("step=%s status=PASS", name)
    return result
