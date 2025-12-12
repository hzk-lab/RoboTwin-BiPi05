import dataclasses
import logging
from typing import Literal

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.lora as _lora
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float,
                  max_period: float) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period)**fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    action_query_group_mode: Literal["shared", "independent", "lora"] = "shared"
    action_query_group_lora_rank: int = 16
    action_query_group_lora_alpha: float = 16.0

    # Set the model specific defaults.
    action_dim: int = 14
    action_horizon: int = 50
    max_token_len: int = 56
    cross_expert_alpha: float = 0.0

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
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
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(gemma_params_filter, )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(nnx.Not(action_expert_params_filter), )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter, )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(nnx.Not(nnx_utils.PathRegex(".*lora.*")), )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


@dataclasses.dataclass(frozen=True)
class _SuffixEmbeddings:
    """Container with per-expert and cross-expert suffix embeddings."""

    per_expert_tokens: at.Float[at.Array, "b experts s emb"]
    per_expert_mask: at.Bool[at.Array, "b experts s"]
    per_expert_ar_mask: at.Bool[at.Array, " s"]
    cross_tokens: at.Float[at.Array, "b s emb"]
    cross_mask: at.Bool[at.Array, "b s"]
    cross_ar_mask: at.Bool[at.Array, " s"]


class Pi0(_model.BaseModel):

    def __init__(self, config: Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        num_action_experts = 2
        if config.action_query_group_mode not in {"shared", "independent", "lora"}:
            raise ValueError("Invalid action_query_group_mode")
        if config.action_query_group_mode == "independent":
            query_group_config = _gemma.QueryGroupConfig(num_groups=num_action_experts, mode="independent")
            action_expert_config = dataclasses.replace(action_expert_config, query_group_config=query_group_config)
        elif config.action_query_group_mode == "lora":
            lora_config = _lora.LoRAConfig(rank=config.action_query_group_lora_rank,
                                           alpha=config.action_query_group_lora_alpha)
            query_group_config = _gemma.QueryGroupConfig(num_groups=num_action_experts,
                                                         mode="lora",
                                                         lora_config=lora_config)
            action_expert_config = dataclasses.replace(action_expert_config, query_group_config=query_group_config)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
            ))
        llm.lazy_init(rngs=rngs, method="init")
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            ))
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        if config.action_dim % 2 != 0:
            raise ValueError("action_dim must be even to split between left/right experts")
        if not 0.0 <= config.cross_expert_alpha <= 1.0:
            raise ValueError("cross_expert_alpha must lie in [0, 1]")
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        half_action_dim = config.action_dim // 2
        self.left_action_in_proj = nnx.Linear(half_action_dim, action_expert_config.width, rngs=rngs)
        self.left_action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.left_action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.left_action_out_proj = nnx.Linear(action_expert_config.width, half_action_dim, rngs=rngs)
        self.right_action_in_proj = nnx.Linear(half_action_dim, action_expert_config.width, rngs=rngs)
        self.right_action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.right_action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.right_action_out_proj = nnx.Linear(action_expert_config.width, half_action_dim, rngs=rngs)
        self._cross_expert_alpha = config.cross_expert_alpha
        self._num_action_experts = num_action_experts

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            ))
            # ar_mask controls how make_attn_mask partitions the sequence into
            # autoregressive blocks. False means "stay in the current block" so
            # setting all prefix entries to False allows the visual tokens to
            # attend to one another freely.
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # Language tokens also stay in the same block as the image tokens
            # so that the prefix attends bidirectionally while remaining
            # invisible to the suffix block that contains states and actions.
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> _SuffixEmbeddings:
        # add a single state token that both experts can read
        state_token = self.state_proj(obs.state)[:, None, :]

        # split actions for the left and right experts
        left_actions, right_actions = jnp.split(noisy_actions, 2, axis=-1)

        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(
            timestep,
            self.left_action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)

        def _expert_tokens(actions, in_proj, mlp_in, mlp_out):
            action_tokens = in_proj(actions)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = mlp_out(action_time_tokens)
            return action_time_tokens

        left_action_time_tokens = _expert_tokens(
            left_actions,
            self.left_action_in_proj,
            self.left_action_time_mlp_in,
            self.left_action_time_mlp_out,
        )
        right_action_time_tokens = _expert_tokens(
            right_actions,
            self.right_action_in_proj,
            self.right_action_time_mlp_in,
            self.right_action_time_mlp_out,
        )

        # Build per-expert sequences `[state, action_horizon]` so that each
        # expert attends independently to the prefix + state.
        left_sequence = jnp.concatenate([state_token, left_action_time_tokens], axis=1)
        right_sequence = jnp.concatenate([state_token, right_action_time_tokens], axis=1)
        per_expert_tokens = jnp.stack([left_sequence, right_sequence], axis=1)
        per_expert_mask = jnp.ones(per_expert_tokens.shape[:3], dtype=jnp.bool_)
        per_expert_ar_mask = jnp.array([True, True] + [False] * max(self.action_horizon - 1, 0))

        # A combined sequence that keeps the left expert causal while letting
        # the right expert optionally view the left expert when alpha > 0.
        cross_tokens = jnp.concatenate([state_token, left_action_time_tokens, right_action_time_tokens], axis=1)
        cross_mask = jnp.ones(cross_tokens.shape[:2], dtype=jnp.bool_)
        cross_ar_mask = jnp.array(
            [True]
            + [True]  # first left token
            + [False] * max(self.action_horizon - 1, 0)
            + [True]  # first right token
            + [False] * max(self.action_horizon - 1, 0)
        )

        return _SuffixEmbeddings(
            per_expert_tokens=per_expert_tokens,
            per_expert_mask=per_expert_mask,
            per_expert_ar_mask=per_expert_ar_mask,
            cross_tokens=cross_tokens,
            cross_mask=cross_mask,
            cross_ar_mask=cross_ar_mask,
        )

    @override
    def compute_loss(self,
                     rng: at.KeyArrayLike,
                     observation: _model.Observation,
                     actions: _model.Actions,
                     *,
                     train: bool = False) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix = self.embed_suffix(observation, x_t, time)
        num_experts = self._num_action_experts
        if suffix.per_expert_tokens.shape[1] != num_experts:
            raise ValueError("embed_suffix returned an unexpected number of experts")
            
        suffix_tokens = einops.rearrange(suffix.per_expert_tokens, "b e s emb -> (b e) s emb")
        suffix_mask = einops.rearrange(suffix.per_expert_mask, "b e s -> (b e) s")
        prefix_tokens_tiled = einops.repeat(prefix_tokens, "b s emb -> (b e) s emb", e=num_experts)
        prefix_mask_tiled = einops.repeat(prefix_mask, "b s -> (b e) s", e=num_experts)
        input_mask = jnp.concatenate([prefix_mask_tiled, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix.per_expert_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm([prefix_tokens_tiled, suffix_tokens],
                                                mask=attn_mask,
                                                positions=positions)
        suffix_out = einops.rearrange(suffix_out, "(b e) s emb -> b e s emb", e=num_experts)
        left_tokens = suffix_out[:, 0, 1:1 + self.action_horizon]
        right_tokens = suffix_out[:, 1, 1:1 + self.action_horizon]

        alpha = self._cross_expert_alpha
        if alpha:
            cross_input_mask = jnp.concatenate([prefix_mask, suffix.cross_mask], axis=1)
            cross_ar_mask = jnp.concatenate([prefix_ar_mask, suffix.cross_ar_mask], axis=0)
            cross_attn_mask = make_attn_mask(cross_input_mask, cross_ar_mask)
            cross_positions = jnp.cumsum(cross_input_mask, axis=1) - 1
            (_, cross_suffix_out), _ = self.PaliGemma.llm([prefix_tokens, suffix.cross_tokens],
                                                          mask=cross_attn_mask,
                                                          positions=cross_positions)
            cross_right_tokens = cross_suffix_out[:, 1 + self.action_horizon:1 + 2 * self.action_horizon]
            right_tokens = (1 - alpha) * right_tokens + alpha * cross_right_tokens

        left_v = self.left_action_out_proj(left_tokens)
        right_v = self.right_action_out_proj(right_tokens)
        v_t = jnp.concatenate([left_v, right_v], axis=-1)

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        num_experts = self._num_action_experts
        # Duplicate the cached prefix context so each expert receives its own
        # batch entry when we expand the suffix stream below. Repeating along
        # the batch axis keeps Gemma's cached shape contract intact during
        # rollout.
        key_cache, value_cache = kv_cache
        kv_cache_per_expert = (
            jnp.repeat(key_cache, num_experts, axis=1),
            jnp.repeat(value_cache, num_experts, axis=1),
        )

        def step(carry):
            x_t, time = carry
            suffix = self.embed_suffix(observation, x_t, jnp.broadcast_to(time, batch_size))
            suffix_tokens = einops.rearrange(suffix.per_expert_tokens, "b e s emb -> (b e) s emb")
            suffix_mask = einops.rearrange(suffix.per_expert_mask, "b e s -> (b e) s")
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix.per_expert_ar_mask)
            prefix_mask_tiled = einops.repeat(prefix_mask, "b p -> (b e) p", e=num_experts)
            prefix_attn_mask = einops.repeat(prefix_mask_tiled, "b p -> b s p", s=suffix_tokens.shape[1])
            positions = jnp.sum(prefix_mask_tiled, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)

            positions = jnp.sum(prefix_mask_tiled, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1


            (_, suffix_out), _ = self.PaliGemma.llm([None, suffix_tokens],
                                                    mask=full_attn_mask,
                                                    positions=positions,
                                                    kv_cache=kv_cache_per_expert)
            suffix_out = einops.rearrange(suffix_out, "(b e) s emb -> b e s emb", e=num_experts)
            left_tokens = suffix_out[:, 0, 1:1 + self.action_horizon]
            right_tokens = suffix_out[:, 1, 1:1 + self.action_horizon]

            alpha = self._cross_expert_alpha
            if alpha:
                suffix_attn_mask_cross = make_attn_mask(suffix.cross_mask, suffix.cross_ar_mask)
                prefix_attn_mask_cross = einops.repeat(prefix_mask, "b p -> b s p", s=suffix.cross_tokens.shape[1])
                full_attn_mask_cross = jnp.concatenate([prefix_attn_mask_cross, suffix_attn_mask_cross], axis=-1)
                positions_cross = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix.cross_mask, axis=-1) - 1
                (_, cross_suffix_out), _ = self.PaliGemma.llm([None, suffix.cross_tokens],
                                                              mask=full_attn_mask_cross,
                                                              positions=positions_cross,
                                                              kv_cache=kv_cache)
                cross_right_tokens = cross_suffix_out[:, 1 + self.action_horizon:1 + 2 * self.action_horizon]
                right_tokens = (1 - alpha) * right_tokens + alpha * cross_right_tokens

            left_v = self.left_action_out_proj(left_tokens)
            right_v = self.right_action_out_proj(right_tokens)
            v_t = jnp.concatenate([left_v, right_v], axis=-1)
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
