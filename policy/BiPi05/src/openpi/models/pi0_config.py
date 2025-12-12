import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    
    # Independent left/right arm action expert variants (only used when dual_arm=True)
    # If None, falls back to action_expert_variant for both arms
    # Use these to configure independent LoRA(L) and LoRA(R) for each arm
    action_expert_left_variant: _gemma.Variant | None = None
    action_expert_right_variant: _gemma.Variant | None = None

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # Dual-arm mode: split action expert into left/right arm experts with cross-attention
    dual_arm: bool = False
    
    # Task Decomposition settings (only used when dual_arm=True)
    # Enables VLM to automatically decompose unified prompt into left/right arm specific prompts
    task_decomposition_enabled: bool = True
    # Number of learnable query tokens per arm for task decomposition
    task_decomposition_num_queries: int = 16
    # Number of attention heads for task decomposition cross-attention
    task_decomposition_num_heads: int = 8
    
    # PECA (Predictive Error for Cooperative Action) settings (only used when dual_arm=True)
    # Enables learning a gate network that predicts when to enable/disable cross-arm attention
    peca_enabled: bool = False
    # PECA loss weight: L_PECA = peca_lambda * (L_on - L_off)_sg * alpha_t
    peca_lambda: float = 0.1
    # Gate network MLP hidden dimension
    gate_mlp_hidden: int = 256
    # L1 regularization weight for gate sparsity
    l1_lambda: float = 0.01
    # Stickiness loss weight for temporal continuity of gate values
    sticky_lambda: float = 0.01
    # Threshold for converting soft gate to hard 0/1 during inference
    gate_threshold: float = 0.5

    @property
    def half_action_dim(self) -> int:
        """Returns half of action_dim for dual-arm mode (e.g., 16 for each arm)."""
        return self.action_dim // 2

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config.
        
        For dual-arm mode with independent LoRA:
        - Freezes VLM (PaliGemma) main weights
        - Freezes left/right arm expert main weights
        - Only trains LoRA(L) and LoRA(R) parameters
        """
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        
        if self.dual_arm:
            # For dual-arm mode, we have two action experts: left and right
            action_expert_left_params_filter = nnx_utils.PathRegex(".*expert_left.*")
            action_expert_right_params_filter = nnx_utils.PathRegex(".*expert_right.*")
            action_expert_params_filter = nnx.Any(
                action_expert_left_params_filter, action_expert_right_params_filter
            )
            
            # Get effective variants for left/right arms
            left_variant = self.action_expert_left_variant or self.action_expert_variant
            right_variant = self.action_expert_right_variant or self.action_expert_variant
            
            # Check if either arm uses LoRA
            left_has_lora = "lora" in left_variant
            right_has_lora = "lora" in right_variant
            
            if "lora" in self.paligemma_variant:
                filters.append(gemma_params_filter)
                # If VLM has LoRA but arms don't, exclude arm params from freeze
                if not left_has_lora and not right_has_lora:
                    filters.append(nnx.Not(action_expert_params_filter))
                has_lora = True
            
            # Handle independent LoRA for left/right arms
            if left_has_lora or right_has_lora:
                # Freeze both arm expert main weights (non-LoRA)
                filters.append(action_expert_params_filter)
                has_lora = True
        else:
            action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
            
            if "lora" in self.paligemma_variant:
                filters.append(
                    gemma_params_filter,
                )
                if "lora" not in self.action_expert_variant:
                    # If only freeze gemma params, exclude action expert params.
                    filters.append(
                        nnx.Not(action_expert_params_filter),
                    )
                has_lora = True
            elif "lora" in self.action_expert_variant:
                filters.append(
                    action_expert_params_filter,
                )
                has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params from freezing.
            # This allows LoRA(L) and LoRA(R) to be trained independently.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        
        # If PECA is enabled, ensure gate network params are not frozen
        if self.peca_enabled:
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*gate_network.*")),
            )
        
        # Also ensure task decomposition params are not frozen
        if self.task_decomposition_enabled and self.dual_arm:
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*task_decomposition.*")),
            )
        
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
