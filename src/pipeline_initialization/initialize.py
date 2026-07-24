import os
from pathlib import Path
import logging
import argparse
import yaml


class InitializePipeline:

    def __init__(self, run_output:str, logging: logging.Logger) -> None:
        self.run_output = run_output
        self.logging = logging

    def create_output_dir(self, run_output: str) -> bool:
        return Path(run_output).mkdir(parents=True, exist_ok=True)

    def run_step(self) -> bool:
        self.create_output_dir(self.run_output)
        self.logging.info(f'Initialization complete')
        return True


def main(config_path: str):
    """
    - Reads the configuration file
    - Initializes logging
    - Calls the InitializePipeline class

    Args:
        config_path (str): path to the configuration file for the run
    Returns:
        None
    """

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    run_output_top_level = config['run_output']['top_level']
    run_output_name = config['name']
    run_output = run_output_top_level + run_output_name

    log_dir = Path(run_output + "/logs")
    log_file = log_dir / "production.log"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Configuration file loaded successfully...")
    logging.info("Now calling initialization...")

    initialize = InitializePipeline(
        run_output = run_output,
        logging = logging
    )
    initialize.run_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='for calling script in cli')
    parser.add_argument("-c", "--config", required=True, help="Path to the config file")
    args = parser.parse_args()
    print('got here')
    print(args)

    print(Path(args.config).resolve())
    config_path = Path(args.config).resolve()
    print(config_path)

    if not config_path.exists():
        raise ValueError(f"Error: Configuration file not found at {config_path}")

    main(config_path)