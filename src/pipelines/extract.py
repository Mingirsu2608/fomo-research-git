import logging

from src.data.extract import extract
from src.pipelines.common import load_config, setup_runtime, run_stage


def main():
    setup_runtime()
    run_stage(logging.getLogger(__name__), "extract", extract, load_config())


if __name__ == "__main__":
    main()
