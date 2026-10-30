import logging

from src.models.train import train_models
from src.pipelines.common import load_config, setup_runtime, run_stage


def main():
    setup_runtime()
    run_stage(logging.getLogger(__name__), "cluster", train_models, load_config())


if __name__ == "__main__":
    main()
