"""Inspect action tensor shapes for a training config's dataset."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import tyro

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def _collect_action_shapes(dataset, limit: int) -> list[tuple[int, ...]]:
    """Gather the action tensor shapes from the first ``limit`` samples."""
    shapes: list[tuple[int, ...]] = []
    sample_count = min(limit, len(dataset))
    for idx in range(sample_count):
        sample = dataset[idx]
        if "actions" not in sample:
            continue
        actions = np.asarray(sample["actions"])
        shapes.append(actions.shape)
    return shapes


def _format_summary(label: str, shapes: Iterable[tuple[int, ...]]) -> str:
    counter = Counter(shapes)
    if not counter:
        return f"{label}: no action tensors found"
    parts = [f"{count}×{shape}" for shape, count in sorted(counter.items())]
    return f"{label}: {', '.join(parts)}"


def main(config_name: str, num_samples: int = 8) -> None:
    """Print the action tensor shapes before and after transforms.

    Args:
        config_name: Name of the training config registered in ``openpi.training.config``.
        num_samples: Number of samples to inspect from the dataset.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be positive")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    raw_dataset = _data_loader.create_dataset(data_config, config.model)
    raw_shapes = _collect_action_shapes(raw_dataset, num_samples)
    print(raw_dataset[0]["action"].shape, "raw dataset sample 0")
    print(raw_dataset[0]["action"][:5], "raw dataset sample 0")
    print(raw_dataset[0]['action_is_pad'], "raw dataset sample 0 action_is_pad")
    print(_format_summary("raw dataset", raw_shapes))

    transformed_dataset = _data_loader.transform_dataset(
        raw_dataset,
        data_config,
        skip_norm_stats=True,
    )
    transformed_shapes = _collect_action_shapes(transformed_dataset, num_samples)
    print(_format_summary("after transforms", transformed_shapes))


if __name__ == "__main__":
    tyro.cli(main)