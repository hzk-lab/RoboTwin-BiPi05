"""Verify that PaliGemma LLM weights are restored by the configured loader."""

from __future__ import annotations

import argparse
import dataclasses
from typing import Any, Mapping

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np

import openpi.training.config as config_lib
import openpi.training.weight_loaders as weight_loaders


def _to_value_tree(state: nnx.State) -> Mapping[str, Any]:
    """Convert a nested pytree of nnx.Params to bare arrays."""

    def is_param(x: Any) -> bool:
        return hasattr(x, "value")

    return jax.tree_util.tree_map(
        lambda x: x.value if is_param(x) else x,
        state.to_pure_dict(),
        is_leaf=is_param,
    )


def _flatten(tree: Mapping[str, Any]) -> dict[str, jnp.ndarray]:
    flat = traverse_util.flatten_dict(tree, sep="/")
    result: dict[str, jnp.ndarray] = {}
    for key, value in flat.items():
        if isinstance(key, tuple):
            key_str = "/".join(str(part) for part in key)
        else:
            key_str = str(key)
        result[key_str] = jnp.asarray(jax.device_get(value))
    return result


@dataclasses.dataclass(frozen=True)
class LLMCheckResult:
    total_llm_keys: int
    loaded_keys: int
    unchanged_keys: int
    max_abs_diffs: dict[str, float]

    def pretty(self) -> str:
        if self.total_llm_keys == 0:
            return "No PaliGemma/llm parameters found in the model state."

        lines = [
            f"PaliGemma/llm parameter groups: {self.total_llm_keys}",
            f"  > restored from checkpoint: {self.loaded_keys}",
            f"  > unchanged (likely random init): {self.unchanged_keys}",
        ]
        if self.max_abs_diffs:
            sample = next(iter(self.max_abs_diffs.items()))
            lines.append(
                "  > sample key: {} (max |Δ| = {:.6f})".format(sample[0], sample[1])
            )
        return "\n".join(lines)


def _summarise_diffs(
    initial: Mapping[str, jnp.ndarray],
    restored: Mapping[str, jnp.ndarray],
    *,
    tol: float,
) -> LLMCheckResult:
    llm_keys = [k for k in restored if k.startswith("PaliGemma/llm/")]
    loaded = 0
    unchanged = 0
    max_abs = {}
    for key in llm_keys:
        if key not in initial:
            continue
        init = np.asarray(initial[key])
        new = np.asarray(restored[key])
        diff = np.max(np.abs(init - new))
        max_abs[key] = float(diff)
        if diff > tol:
            loaded += 1
        else:
            unchanged += 1
    return LLMCheckResult(
        total_llm_keys=len(llm_keys),
        loaded_keys=loaded,
        unchanged_keys=unchanged,
        max_abs_diffs=max_abs,
    )


def run_check(config_name: str, *, seed: int, tol: float) -> LLMCheckResult:
    cfg = dataclasses.replace(config_lib.get_config(config_name))
    rng = jax.random.key(seed)
    model = cfg.model.create(rng)
    initial_state = nnx.state(model)
    initial_params = _to_value_tree(initial_state)

    loader = cfg.weight_loader
    if isinstance(loader, weight_loaders.NoOpWeightLoader):
        raise ValueError("Configured weight loader is NoOp; there is nothing to verify.")

    restored = loader.load(initial_params)
    flat_init = _flatten(initial_params)
    flat_restored = _flatten(restored)
    return _summarise_diffs(flat_init, flat_restored, tol=tol)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True, help="Training config to inspect")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for model initialisation (only influences the random baseline)",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Tolerance for considering a parameter unchanged after loading.",
    )
    args = parser.parse_args()

    result = run_check(args.config_name, seed=args.seed, tol=args.tol)
    print(result.pretty())


if __name__ == "__main__":
    main()