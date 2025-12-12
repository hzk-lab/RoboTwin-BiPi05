import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):

    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):

    def load(self, params: at.Params) -> at.Params:
        return params


def _has_allowed_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    """Returns ``True`` if ``path`` starts with any of ``prefixes``."""
    for prefix in prefixes:
        if path == prefix or path.startswith(prefix + "/"):
            return True
    return False


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    The loader optionally filters the restored state by matching flattened parameter
    paths against ``allowed_param_prefixes``.  When filtering is enabled we intentionally
    drop everything else from the checkpoint (for example, action-expert parameters) so
    that those parts of the model keep their randomly initialised values.  After the
    subset is restored we merge the filtered weights back with ``params``.  This merge is
    what prevents shape/key mismatches: any parameter that is not present in the filtered
    checkpoint simply keeps the value that the model created during initialisation.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "s3://openpi-assets/checkpoints/<model>/params"

    Args:
        params_path: Path to the parameters checkpoint.
        allowed_param_prefixes: If provided, only parameters whose flattened path begins
            with one of the prefixes will be restored from the checkpoint.
    """

    params_path: str
    allowed_param_prefixes: tuple[str, ...] | None = None

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_params = flax.traverse_util.flatten_dict(loaded_params, sep="/")
        filtered = False
        if self.allowed_param_prefixes is not None:
            flat_params = {
                k: v for k, v in flat_params.items() if _has_allowed_prefix(k, self.allowed_param_prefixes)
            }
            filtered = True
        if filtered:
            loaded_params = flax.traverse_util.unflatten_dict(flat_params, sep="/")
        missing_regex = ".*" if filtered else ".*lora.*"
        # When ``missing_regex`` is ``.*`` we merge back any weights that were intentionally
        # dropped (e.g. action expert parameters) from the randomly initialized ``params``.
        return _merge_params(loaded_params, params, missing_regex=missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
            gs={"token": "anon"},
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
