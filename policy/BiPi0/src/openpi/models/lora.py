import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


def _lora_param_shapes(shape: tuple[int, ...], config: LoRAConfig) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shape_a, shape_b = list(shape), list(shape)
    shape_a[config.axes[1]] = config.rank
    shape_b[config.axes[0]] = config.rank
    return tuple(shape_a), tuple(shape_b)


def _make_lora_eqns(eqn: str, config: LoRAConfig) -> tuple[str, str]:
    if "L" in eqn:
        raise ValueError(f"L already in eqn: {eqn}")
    if not (m := re.match("(.*),(.*)->(.*)", eqn)):
        raise ValueError(f"Unsupported einsum eqn: {eqn}")
    lhs, rhs, out = m.groups()

    a_label, b_label = (rhs[x] for x in config.axes)
    label = config.label

    a_rhs = rhs.replace(b_label, label)
    a_out = out.replace(b_label, label)
    eqn_a = f"{lhs},{a_rhs}->{a_out}"

    b_rhs = rhs.replace(a_label, label)
    eqn_b = f"{a_out},{b_rhs}->{out}"

    return eqn_a, eqn_b


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = _lora_param_shapes(self.shape, config)
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = _make_lora_eqns(eqn, config)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result


class EinsumLoRAAdapter(nn.Module):
    """LoRA-only einsum that shares the base weight with another module."""

    # Shape of the weight that LoRA would be applied to.
    shape: tuple[int, ...]
    # LoRA configuration describing the adapter.
    lora_config: LoRAConfig

    def setup(self):
        shape_a, shape_b = _lora_param_shapes(self.shape, self.lora_config)
        self.w_a = self.param("lora_a", self.lora_config.init_fn, shape_a)
        self.w_b = self.param("lora_b", self.lora_config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # original dtype, could be half-precision
        eqn_a, eqn_b = _make_lora_eqns(eqn, self.lora_config)
        lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
        lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
        return lora * self.lora_config.scaling_value


class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, )),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param("gating_einsum_lora_b", self.lora_config.init_fn,
                           (2, self.lora_config.rank, self.hidden_dim)),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
