import logging

from src.evaluation.evaluate import evaluate_models
from src.pipelines.common import load_config, setup_runtime, run_stage


def main():
    setup_runtime()
    run_stage(logging.getLogger(__name__), "evaluate", evaluate_models, load_config())


if __name__ == "__main__":
    main()
