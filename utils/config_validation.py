"""Minimal config validation utilities."""

from typing import Any, Dict


REQUIRED_TOP_LEVEL = ("model", "training", "data")


def validate_training_config(config: Dict[str, Any]) -> None:
    missing = [k for k in REQUIRED_TOP_LEVEL if k not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {missing}")

    model_name = config["model"].get("name")
    if not model_name:
        raise ValueError("model.name is required")

    batch_size = config["training"].get("batch_size")
    if batch_size is None or int(batch_size) <= 0:
        raise ValueError("training.batch_size must be > 0")
