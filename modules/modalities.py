from typing import Any, Mapping


def resolve_modalities(training_cfg: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return whether discrete and continuous audio targets are enabled."""
    if "discrete" not in training_cfg or "continuous" not in training_cfg:
        raise ValueError(
            "Set both training.discrete and training.continuous in the config."
        )

    discrete = bool(training_cfg["discrete"])
    continuous = bool(training_cfg["continuous"])

    if not (discrete or continuous):
        raise ValueError(
            "At least one of training.discrete or training.continuous must be true."
        )

    return discrete, continuous
