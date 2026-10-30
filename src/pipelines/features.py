import logging

from src.features.build import build_features
from src.pipelines.common import load_config, setup_runtime, run_stage


def main():
    setup_runtime()
    run_stage(logging.getLogger(__name__), "features", build_features, load_config())


if __name__ == "__main__":
    main()
