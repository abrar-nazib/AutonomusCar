from __future__ import annotations

import argparse
from pathlib import Path

from .app import AutonomousCarApp
from .config import DEFAULT_CONFIG_PATH, AppConfig
from .logging_setup import configure as configure_logging


def main() -> None:
    parser = argparse.ArgumentParser(prog="autocar", description="Run the autonomous car loop.")
    parser.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to YAML config."
    )
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config)
    configure_logging(config.logging)
    AutonomousCarApp(config, config_path=args.config).run()


if __name__ == "__main__":
    main()
