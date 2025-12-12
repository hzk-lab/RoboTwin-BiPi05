import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel, PaliGemmaWithDualExpertModel
from openpi.models_pytorch.task_decomposition import TaskDecompositionModule
from openpi.models_pytorch.gate_network import GateNetwork, compute_gate_bce_loss, compute_peca_loss, compute_stickiness_loss
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


def make_dual_arm_att_masks(pad_masks_left, pad_masks_right, att_masks_left, att_masks_right, action_horizon):
    """Create attention masks for dual-arm model where left and right arms can see each other.
    
    The attention pattern is:
    - Prefix tokens can only see themselves (bidirectional within prefix)
    - Left arm tokens can see prefix + all left tokens + all right tokens at same or earlier timesteps
    - Right arm tokens can see prefix + all right tokens + all left tokens at same or earlier timesteps
    
    Args:
        pad_masks_left: Padding mask for left arm [batch, left_len]
        pad_masks_right: Padding mask for right arm [batch, right_len]
        att_masks_left: Attention mask for left arm [batch, left_len]
        att_masks_right: Attention mask for right arm [batch, right_len]
        action_horizon: Number of action timesteps
    
    Returns:
        Combined 2D attention mask [batch, total_len, total_len]
    """
    batch_size = pad_masks_left.shape[0]
    left_len = pad_masks_left.shape[1]
    right_len = pad_masks_right.shape[1]
    total_len = left_len + right_len
    device = pad_masks_left.device
    
    # Create individual 2D masks
    left_2d = make_att_2d_masks(pad_masks_left, att_masks_left)
    right_2d = make_att_2d_masks(pad_masks_right, att_masks_right)
    
    # Create cross-attention masks (left can see right, right can see left within same timestep block)
    # For dual-arm, we want tokens at the same timestep to see each other freely
    cross_left_to_right = torch.zeros(batch_size, left_len, right_len, dtype=torch.bool, device=device)
    cross_right_to_left = torch.zeros(batch_size, right_len, left_len, dtype=torch.bool, device=device)
    
    # Allow cross-attention within same timestep (assuming actions are aligned by timestep)
    for t in range(action_horizon):
        cross_left_to_right[:, t, t] = True
        cross_right_to_left[:, t, t] = True
        # Also allow causal cross-attention to previous timesteps
        for prev_t in range(t):
            cross_left_to_right[:, t, prev_t] = True
            cross_right_to_left[:, t, prev_t] = True
    
    # Apply padding masks
    cross_left_to_right = cross_left_to_right & pad_masks_left[:, :, None] & pad_masks_right[:, None, :]
    cross_right_to_left = cross_right_to_left & pad_masks_right[:, :, None] & pad_masks_left[:, None, :]
    
    # Combine into full attention mask
    # [left_2d, cross_left_to_right]
    # [cross_right_to_left, right_2d]
    top_row = torch.cat([left_2d, cross_left_to_right], dim=2)
    bottom_row = torch.cat([cross_right_to_left, right_2d], dim=2)
    full_2d_mask = torch.cat([top_row, bottom_row], dim=1)
    
    return full_2d_mask


class PI0DualArmPytorch(nn.Module):
    """Dual-arm variant of PI0Pytorch with separate left/right action experts.
    
    This model splits the action space into left arm (first half) and right arm (second half),
    using separate action experts with cross-attention for coordinated bimanual control.
    
    Each arm can have its own LoRA configuration via:
    - config.action_expert_left_variant: LoRA(L) for left arm
    - config.action_expert_right_variant: LoRA(R) for right arm
    
    If not specified, both arms fall back to config.action_expert_variant.
    
    PECA (Predictive Error for Cooperative Action) training:
    When config.peca_enabled=True, the model learns a gate network that predicts
    when to enable/disable cross-arm attention based on the task requirements.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.half_action_dim = config.half_action_dim  # 16

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        
        # Get independent configs for left and right arm action experts
        # Fall back to action_expert_variant if arm-specific variant is not specified
        left_variant = getattr(config, 'action_expert_left_variant', None) or config.action_expert_variant
        right_variant = getattr(config, 'action_expert_right_variant', None) or config.action_expert_variant
        
        action_expert_left_config = _gemma.get_config(left_variant)
        action_expert_right_config = _gemma.get_config(right_variant)
        
        # Store action expert width for projection layers (use left config as reference)
        action_expert_width = action_expert_left_config.width

        # Use dual expert model with independent left/right configs
        # This allows LoRA(L) and LoRA(R) to be trained independently
        self.paligemma_with_expert = PaliGemmaWithDualExpertModel(
            paligemma_config,
            action_expert_left_config,    # Left arm expert config (may include LoRA(L))
            action_expert_right_config,   # Right arm expert config (may include LoRA(R))
            use_adarms=[False, True, True] if self.pi05 else [False, False, False],
            precision=config.dtype,
        )
        
        # Task Decomposition Module for generating left/right arm specific prefixes
        self.task_decomposition_enabled = getattr(config, 'task_decomposition_enabled', True)
        if self.task_decomposition_enabled:
            self.task_decomposition = TaskDecompositionModule(
                embed_dim=paligemma_config.width,  # VLM hidden dimension
                num_heads=getattr(config, 'task_decomposition_num_heads', 8),
                num_query_tokens=getattr(config, 'task_decomposition_num_queries', 16),
                dropout=0.0,
                use_layer_norm=True,
                precision=config.dtype,
            )
        
        # PECA (Predictive Error for Cooperative Action) Gate Network
        self.peca_enabled = getattr(config, 'peca_enabled', False)
        if self.peca_enabled:
            self.gate_network = GateNetwork(
                hidden_dim=action_expert_width,
                mlp_hidden=getattr(config, 'gate_mlp_hidden', 256),
                dropout=0.0,
                precision=config.dtype,
            )
            # PECA hyperparameters
            self.peca_lambda = getattr(config, 'peca_lambda', 0.1)
            self.l1_lambda = getattr(config, 'l1_lambda', 0.01)
            self.sticky_lambda = getattr(config, 'sticky_lambda', 0.01)
            self.gate_threshold = getattr(config, 'gate_threshold', 0.5)

        # Separate projection layers for left and right arms
        self.action_in_proj_left = nn.Linear(self.half_action_dim, action_expert_width)
        self.action_in_proj_right = nn.Linear(self.half_action_dim, action_expert_width)
        self.action_out_proj_left = nn.Linear(action_expert_width, self.half_action_dim)
        self.action_out_proj_right = nn.Linear(action_expert_width, self.half_action_dim)

        if self.pi05:
            # Shared time MLP for both arms (time conditioning is the same)
            self.time_mlp_in = nn.Linear(action_expert_width, action_expert_width)
            self.time_mlp_out = nn.Linear(action_expert_width, action_expert_width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_width)
            self.action_time_mlp_in_left = nn.Linear(2 * action_expert_width, action_expert_width)
            self.action_time_mlp_out_left = nn.Linear(action_expert_width, action_expert_width)
            self.action_time_mlp_in_right = nn.Linear(2 * action_expert_width, action_expert_width)
            self.action_time_mlp_out_right = nn.Linear(action_expert_width, action_expert_width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert_left.model.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert_right.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0DualArmPytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert_left.model.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert_right.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0DualArmPytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_prefix_decomposed(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed prefix and decompose into left/right arm specific prefixes.
        
        This method first computes the unified prefix embedding from images and language,
        then uses the TaskDecompositionModule to generate arm-specific prefixes.
        
        Returns:
            left_prefix: Left arm specific prefix [batch, num_query_tokens, embed_dim]
            right_prefix: Right arm specific prefix [batch, num_query_tokens, embed_dim]
            unified_prefix_embs: Original unified prefix (for shared attention with experts)
            prefix_pad_masks: Padding masks for unified prefix
            prefix_att_masks: Attention masks for unified prefix
        """
        # Get unified prefix embedding
        unified_prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        if self.task_decomposition_enabled:
            # Decompose into left/right arm specific prefixes
            left_prefix, right_prefix = self.task_decomposition(
                unified_prefix_embs, prefix_pad_masks
            )
        else:
            # If task decomposition is disabled, use the unified prefix for both arms
            # This maintains backward compatibility
            left_prefix = unified_prefix_embs
            right_prefix = unified_prefix_embs
        
        return left_prefix, right_prefix, unified_prefix_embs, prefix_pad_masks, prefix_att_masks

    def embed_suffix_dual(self, state, noisy_actions_left, noisy_actions_right, timestep):
        """Embed state and noisy actions for left and right arms separately.
        
        Returns:
            left_embs, left_pad_masks, left_att_masks: Embeddings and masks for left arm
            right_embs, right_pad_masks, right_att_masks: Embeddings and masks for right arm
            adarms_cond: Time conditioning for adaRMS (shared between arms)
        """
        bsize = noisy_actions_left.shape[0]
        device = noisy_actions_left.device

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj_left.out_features, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Project left arm actions
        def action_proj_left_func(noisy_actions):
            return self.action_in_proj_left(noisy_actions)

        left_action_emb = self._apply_checkpoint(action_proj_left_func, noisy_actions_left)

        # Project right arm actions
        def action_proj_right_func(noisy_actions):
            return self.action_in_proj_right(noisy_actions)

        right_action_emb = self._apply_checkpoint(action_proj_right_func, noisy_actions_right)

        if not self.pi05:
            # Fuse timestep + action information using MLPs for each arm
            time_emb_expanded = time_emb[:, None, :].expand_as(left_action_emb)
            
            # Left arm MLP
            left_action_time_emb = torch.cat([left_action_emb, time_emb_expanded], dim=2)
            def mlp_left_func(action_time_emb):
                x = self.action_time_mlp_in_left(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out_left(x)
            left_embs = self._apply_checkpoint(mlp_left_func, left_action_time_emb)
            
            # Right arm MLP
            right_action_time_emb = torch.cat([right_action_emb, time_emb_expanded], dim=2)
            def mlp_right_func(action_time_emb):
                x = self.action_time_mlp_in_right(action_time_emb)
                x = F.silu(x)
                return self.action_time_mlp_out_right(x)
            right_embs = self._apply_checkpoint(mlp_right_func, right_action_time_emb)
            
            adarms_cond = None
        else:
            # Time MLP for adaRMS conditioning (shared)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            left_embs = left_action_emb
            right_embs = right_action_emb
            adarms_cond = time_emb

        # Create padding and attention masks for both arms
        action_horizon = self.config.action_horizon
        left_pad_masks = torch.ones(bsize, action_horizon, dtype=torch.bool, device=device)
        right_pad_masks = torch.ones(bsize, action_horizon, dtype=torch.bool, device=device)
        
        # Attention masks: first token starts new block, subsequent tokens in same block
        left_att_masks = torch.tensor([1] + [0] * (action_horizon - 1), dtype=left_embs.dtype, device=device)
        left_att_masks = left_att_masks[None, :].expand(bsize, action_horizon)
        right_att_masks = torch.tensor([1] + [0] * (action_horizon - 1), dtype=right_embs.dtype, device=device)
        right_att_masks = right_att_masks[None, :].expand(bsize, action_horizon)

        return (left_embs, left_pad_masks, left_att_masks,
                right_embs, right_pad_masks, right_att_masks,
                adarms_cond)

    def forward(self, observation, actions, noise=None, time=None, force_alpha=None) -> Tensor:
        """Do a full training forward pass and compute the loss for dual-arm model.
        
        Args:
            observation: Observation containing images, state, and prompt
            actions: Full action tensor [batch, horizon, action_dim] (32 dims: 16 left + 16 right)
            noise: Optional noise tensor
            time: Optional time tensor
            force_alpha: If specified, force alpha_t to this value (0.0=cross-attn, 1.0=independent)
                        If None and peca_enabled, use GateNetwork prediction
            
        Returns:
            MSE loss tensor (if peca_enabled=False or force_alpha is specified)
            dict with 'loss', 'alpha_t', 'left_hidden', 'right_hidden' (if peca_enabled=True and force_alpha=None)
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        # Split actions into left and right arms
        actions_left = actions[..., :self.half_action_dim]
        actions_right = actions[..., self.half_action_dim:]

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        # Split noise similarly
        noise_left = noise[..., :self.half_action_dim]
        noise_right = noise[..., self.half_action_dim:]

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        
        # Flow matching interpolation for both arms
        x_t_left = time_expanded * noise_left + (1 - time_expanded) * actions_left
        x_t_right = time_expanded * noise_right + (1 - time_expanded) * actions_right
        u_t_left = noise_left - actions_left
        u_t_right = noise_right - actions_right

        # Embed prefix with task decomposition (VLM processing + arm-specific decomposition)
        (left_task_prefix, right_task_prefix, 
         prefix_embs, prefix_pad_masks, prefix_att_masks) = self.embed_prefix_decomposed(
            images, img_masks, lang_tokens, lang_masks
        )
        
        # Embed suffix for both arms
        (left_embs, left_pad_masks, left_att_masks,
         right_embs, right_pad_masks, right_att_masks,
         adarms_cond) = self.embed_suffix_dual(state, x_t_left, x_t_right, time)

        # Convert to bfloat16 if needed
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            left_embs = left_embs.to(dtype=torch.bfloat16)
            right_embs = right_embs.to(dtype=torch.bfloat16)
            if self.task_decomposition_enabled:
                left_task_prefix = left_task_prefix.to(dtype=torch.bfloat16)
                right_task_prefix = right_task_prefix.to(dtype=torch.bfloat16)

        # Prepend task-specific prefix to arm embeddings if task decomposition is enabled
        # This gives each arm expert its own task-specific conditioning
        if self.task_decomposition_enabled:
            # Concatenate: [task_prefix, action_embs] for each arm
            left_embs = torch.cat([left_task_prefix, left_embs], dim=1)
            right_embs = torch.cat([right_task_prefix, right_embs], dim=1)
            
            # Update masks accordingly
            bsize = left_pad_masks.shape[0]
            device = left_pad_masks.device
            num_task_tokens = left_task_prefix.shape[1]
            
            # Task prefix tokens are always valid
            task_prefix_pad_mask = torch.ones(bsize, num_task_tokens, dtype=torch.bool, device=device)
            left_pad_masks = torch.cat([task_prefix_pad_mask, left_pad_masks], dim=1)
            right_pad_masks = torch.cat([task_prefix_pad_mask, right_pad_masks], dim=1)
            
            # Task prefix tokens can see each other (bidirectional within task prefix)
            task_prefix_att_mask = torch.zeros(bsize, num_task_tokens, dtype=left_embs.dtype, device=device)
            task_prefix_att_mask[:, 0] = 1  # First token starts new block
            left_att_masks = torch.cat([task_prefix_att_mask, left_att_masks], dim=1)
            right_att_masks = torch.cat([task_prefix_att_mask, right_att_masks], dim=1)

        # Combine masks for three-branch attention
        # Full sequence: [prefix, left_arm (with task_prefix), right_arm (with task_prefix)]
        pad_masks = torch.cat([prefix_pad_masks, left_pad_masks, right_pad_masks], dim=1)
        
        # Create attention masks
        # Prefix: bidirectional within prefix
        # Left/Right arms: can see prefix + cross-attention between arms
        prefix_len = prefix_pad_masks.shape[1]
        left_len = left_pad_masks.shape[1]
        right_len = right_pad_masks.shape[1]
        bsize = prefix_pad_masks.shape[0]
        device = prefix_pad_masks.device
        
        # Build combined attention mask
        # Row i can attend to column j if mask[i,j] = True
        
        # Prefix-to-prefix: bidirectional
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        
        # Left-to-prefix: can see all prefix
        left_to_prefix = prefix_pad_masks[:, None, :].expand(bsize, left_len, prefix_len)
        
        # Right-to-prefix: can see all prefix
        right_to_prefix = prefix_pad_masks[:, None, :].expand(bsize, right_len, prefix_len)
        
        # Prefix-to-left/right: cannot see (causal)
        prefix_to_left = torch.zeros(bsize, prefix_len, left_len, dtype=torch.bool, device=device)
        prefix_to_right = torch.zeros(bsize, prefix_len, right_len, dtype=torch.bool, device=device)
        
        # Left-to-left: causal within arm
        left_att_2d = make_att_2d_masks(left_pad_masks, left_att_masks)
        
        # Right-to-right: causal within arm
        right_att_2d = make_att_2d_masks(right_pad_masks, right_att_masks)
        
        # Cross-attention between arms (bidirectional within same timestep, causal across timesteps)
        left_to_right = make_dual_arm_att_masks(
            left_pad_masks, right_pad_masks, 
            left_att_masks, right_att_masks, 
            self.config.action_horizon
        )[:, :left_len, left_len:]
        
        right_to_left = make_dual_arm_att_masks(
            left_pad_masks, right_pad_masks,
            left_att_masks, right_att_masks,
            self.config.action_horizon
        )[:, left_len:, :left_len]
        
        # Assemble full attention mask
        # [prefix_att_2d, prefix_to_left, prefix_to_right]
        # [left_to_prefix, left_att_2d, left_to_right]
        # [right_to_prefix, right_to_left, right_att_2d]
        row1 = torch.cat([prefix_att_2d, prefix_to_left, prefix_to_right], dim=2)
        row2 = torch.cat([left_to_prefix, left_att_2d, left_to_right], dim=2)
        row3 = torch.cat([right_to_prefix, right_to_left, right_att_2d], dim=2)
        att_2d_masks = torch.cat([row1, row2, row3], dim=1)
        
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks for transformer
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Determine alpha_t for gated cross-attention
        alpha_t = None
        if force_alpha is not None:
            # Force alpha to specific value (used for PECA on/off paths)
            alpha_t = torch.full((bsize, 1), force_alpha, device=device, dtype=left_embs.dtype)
        elif self.peca_enabled:
            # Will predict alpha_t after getting hidden states
            # For now, do a forward pass with full cross-attention to get hidden states
            alpha_t = None  # Will be computed after first pass

        # Forward pass through dual expert model
        def forward_func(prefix_embs, left_embs, right_embs, att_2d_masks_4d, position_ids, adarms_cond, alpha_t):
            (_, left_out, right_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, left_embs, right_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond, adarms_cond],
                alpha_t=alpha_t,
            )
            return left_out, right_out

        left_out, right_out = self._apply_checkpoint(
            forward_func, prefix_embs, left_embs, right_embs, att_2d_masks_4d, position_ids, adarms_cond, alpha_t
        )

        # Extract action outputs (hidden states for PECA gate prediction)
        left_hidden = left_out[:, -self.config.action_horizon:]
        right_hidden = right_out[:, -self.config.action_horizon:]
        left_hidden_float = left_hidden.to(dtype=torch.float32)
        right_hidden_float = right_hidden.to(dtype=torch.float32)

        # Project to action space
        def action_out_proj_func(left_out, right_out):
            v_left = self.action_out_proj_left(left_out)
            v_right = self.action_out_proj_right(right_out)
            return v_left, v_right

        v_t_left, v_t_right = self._apply_checkpoint(action_out_proj_func, left_hidden_float, right_hidden_float)

        # Compute loss for both arms
        loss_left = F.mse_loss(u_t_left, v_t_left, reduction="none")
        loss_right = F.mse_loss(u_t_right, v_t_right, reduction="none")
        
        # Combine losses (concatenate along action dimension)
        loss = torch.cat([loss_left, loss_right], dim=-1)
        
        # If PECA is enabled and we're not forcing alpha, return additional info for PECA loss computation
        if self.peca_enabled and force_alpha is None:
            # Predict alpha_t from hidden states
            alpha_t = self.gate_network(left_hidden_float, right_hidden_float)
            return {
                'loss': loss,
                'alpha_t': alpha_t,
                'left_hidden': left_hidden_float,
                'right_hidden': right_hidden_float,
            }
        
        return loss

    def forward_with_peca(self, observation, actions, noise=None, time=None) -> dict:
        """Do PECA training forward pass with three paths: on, off, and predicted.
        
        This method performs:
        1. Forward with alpha=0 (cross-attention ON) -> L_BC^on
        2. Forward with alpha=1 (cross-attention OFF) -> L_BC^off
        3. Forward with predicted alpha -> L_BC + gate prediction
        4. Compute all PECA losses
        
        Args:
            observation: Observation containing images, state, and prompt
            actions: Full action tensor [batch, horizon, action_dim]
            noise: Optional noise tensor (will be reused for all paths)
            time: Optional time tensor (will be reused for all paths)
            
        Returns:
            dict with 'total_loss', 'loss_bc', 'loss_gate', 'loss_peca', 'loss_l1', 'loss_sticky', 'alpha_t'
        """
        if not self.peca_enabled:
            raise ValueError("PECA is not enabled. Set config.peca_enabled=True to use PECA training.")
        
        bsize = actions.shape[0]
        device = actions.device
        
        # Sample noise and time once and reuse for all paths
        if noise is None:
            noise = self.sample_noise(actions.shape, device)
        if time is None:
            time = self.sample_time(bsize, device)
        
        # Path 1: alpha=0 (cross-attention ON)
        loss_on = self.forward(observation, actions, noise=noise, time=time, force_alpha=0.0)
        loss_on_mean = loss_on.mean()
        
        # Path 2: alpha=1 (cross-attention OFF)
        loss_off = self.forward(observation, actions, noise=noise, time=time, force_alpha=1.0)
        loss_off_mean = loss_off.mean()
        
        # Path 3: predicted alpha (main path)
        outputs = self.forward(observation, actions, noise=noise, time=time, force_alpha=None)
        loss_bc = outputs['loss']
        alpha_t = outputs['alpha_t']
        
        loss_bc_mean = loss_bc.mean()
        
        # Compute Gate BCE loss (learn to predict when cooperation helps)
        loss_gate, alpha_label = compute_gate_bce_loss(alpha_t, loss_on_mean, loss_off_mean)
        
        # Compute PECA loss
        loss_peca = compute_peca_loss(alpha_t, loss_on_mean, loss_off_mean, self.peca_lambda)
        
        # Compute L1 regularization (encourage sparsity)
        loss_l1 = self.l1_lambda * alpha_t.mean()
        
        # Compute stickiness loss (encourage temporal continuity)
        # Note: For single-timestep prediction, this will be 0
        loss_sticky = self.sticky_lambda * compute_stickiness_loss(alpha_t.squeeze(-1))
        
        # Total loss
        total_loss = loss_bc_mean + loss_gate + loss_peca + loss_l1 + loss_sticky
        
        return {
            'total_loss': total_loss,
            'loss_bc': loss_bc_mean,
            'loss_bc_on': loss_on_mean,
            'loss_bc_off': loss_off_mean,
            'loss_gate': loss_gate,
            'loss_peca': loss_peca,
            'loss_l1': loss_l1,
            'loss_sticky': loss_sticky,
            'alpha_t': alpha_t,
            'alpha_label': alpha_label,
        }

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action for dual-arm model.
        
        Returns:
            Combined action tensor [batch, horizon, action_dim] (32 dims: 16 left + 16 right)
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # Split noise into left and right
        noise_left = noise[..., :self.half_action_dim]
        noise_right = noise[..., self.half_action_dim:]

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # Get decomposed prefix (left/right task-specific + unified)
        (left_task_prefix, right_task_prefix, 
         prefix_embs, prefix_pad_masks, prefix_att_masks) = self.embed_prefix_decomposed(
            images, img_masks, lang_tokens, lang_masks
        )
        
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t_left = noise_left
        x_t_right = noise_right
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t_left, v_t_right = self.denoise_step_dual(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t_left,
                x_t_right,
                expanded_time,
                left_task_prefix,
                right_task_prefix,
            )

            # Euler step for both arms
            x_t_left = x_t_left + dt * v_t_left
            x_t_right = x_t_right + dt * v_t_right
            time += dt
        
        # Combine left and right actions
        return torch.cat([x_t_left, x_t_right], dim=-1)

    def denoise_step_dual(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t_left,
        x_t_right,
        timestep,
        left_task_prefix=None,
        right_task_prefix=None,
    ):
        """Apply one denoising step for dual-arm model.
        
        Args:
            state: Robot state
            prefix_pad_masks: Padding masks for cached prefix
            past_key_values: Cached key-values from VLM prefix
            x_t_left: Current left arm noise/action
            x_t_right: Current right arm noise/action
            timestep: Current timestep
            left_task_prefix: Left arm task-specific prefix from TaskDecomposition
            right_task_prefix: Right arm task-specific prefix from TaskDecomposition
        """
        (left_embs, left_pad_masks, left_att_masks,
         right_embs, right_pad_masks, right_att_masks,
         adarms_cond) = self.embed_suffix_dual(state, x_t_left, x_t_right, timestep)

        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        device = prefix_pad_masks.device

        # Prepend task-specific prefix if task decomposition is enabled
        if self.task_decomposition_enabled and left_task_prefix is not None:
            # Convert to appropriate dtype
            if left_embs.dtype == torch.bfloat16:
                left_task_prefix = left_task_prefix.to(dtype=torch.bfloat16)
                right_task_prefix = right_task_prefix.to(dtype=torch.bfloat16)
            
            # Concatenate: [task_prefix, action_embs] for each arm
            left_embs = torch.cat([left_task_prefix, left_embs], dim=1)
            right_embs = torch.cat([right_task_prefix, right_embs], dim=1)
            
            # Update masks
            num_task_tokens = left_task_prefix.shape[1]
            task_prefix_pad_mask = torch.ones(batch_size, num_task_tokens, dtype=torch.bool, device=device)
            left_pad_masks = torch.cat([task_prefix_pad_mask, left_pad_masks], dim=1)
            right_pad_masks = torch.cat([task_prefix_pad_mask, right_pad_masks], dim=1)
            
            # Task prefix attention masks
            task_prefix_att_mask = torch.zeros(batch_size, num_task_tokens, dtype=left_embs.dtype, device=device)
            task_prefix_att_mask[:, 0] = 1
            left_att_masks = torch.cat([task_prefix_att_mask, left_att_masks], dim=1)
            right_att_masks = torch.cat([task_prefix_att_mask, right_att_masks], dim=1)

        left_len = left_pad_masks.shape[1]
        right_len = right_pad_masks.shape[1]

        # Build attention masks for cached prefix + suffix
        # Left can see prefix
        left_to_prefix_2d = prefix_pad_masks[:, None, :].expand(batch_size, left_len, prefix_len)
        # Right can see prefix
        right_to_prefix_2d = prefix_pad_masks[:, None, :].expand(batch_size, right_len, prefix_len)
        
        # Self-attention within each arm
        left_att_2d = make_att_2d_masks(left_pad_masks, left_att_masks)
        right_att_2d = make_att_2d_masks(right_pad_masks, right_att_masks)
        
        # Cross-attention between arms (need to handle task prefix tokens)
        # For simplicity, we compute cross mask for action tokens only, task prefix uses self-attention
        action_horizon = self.config.action_horizon
        num_task_tokens = left_len - action_horizon if self.task_decomposition_enabled and left_task_prefix is not None else 0
        
        if num_task_tokens > 0:
            # Task prefix tokens can see each other across arms
            left_task_to_right_task = torch.ones(batch_size, num_task_tokens, num_task_tokens, dtype=torch.bool, device=device)
            right_task_to_left_task = torch.ones(batch_size, num_task_tokens, num_task_tokens, dtype=torch.bool, device=device)
            
            # Task prefix to action: can see all task prefix
            left_task_to_right_action = torch.zeros(batch_size, num_task_tokens, action_horizon, dtype=torch.bool, device=device)
            right_task_to_left_action = torch.zeros(batch_size, num_task_tokens, action_horizon, dtype=torch.bool, device=device)
            
            # Action to task prefix: can see task prefix
            left_action_to_right_task = torch.ones(batch_size, action_horizon, num_task_tokens, dtype=torch.bool, device=device)
            right_action_to_left_task = torch.ones(batch_size, action_horizon, num_task_tokens, dtype=torch.bool, device=device)
            
            # Action to action: use dual arm cross mask
            action_left_pad = torch.ones(batch_size, action_horizon, dtype=torch.bool, device=device)
            action_right_pad = torch.ones(batch_size, action_horizon, dtype=torch.bool, device=device)
            action_left_att = torch.tensor([1] + [0] * (action_horizon - 1), dtype=left_embs.dtype, device=device)
            action_left_att = action_left_att[None, :].expand(batch_size, action_horizon)
            action_right_att = action_left_att.clone()
            
            cross_mask_action = make_dual_arm_att_masks(
                action_left_pad, action_right_pad,
                action_left_att, action_right_att,
                action_horizon
            )
            left_action_to_right_action = cross_mask_action[:, :action_horizon, action_horizon:]
            right_action_to_left_action = cross_mask_action[:, action_horizon:, :action_horizon]
            
            # Assemble cross attention
            left_to_right_top = torch.cat([left_task_to_right_task, left_task_to_right_action], dim=2)
            left_to_right_bottom = torch.cat([left_action_to_right_task, left_action_to_right_action], dim=2)
            left_to_right = torch.cat([left_to_right_top, left_to_right_bottom], dim=1)
            
            right_to_left_top = torch.cat([right_task_to_left_task, right_task_to_left_action], dim=2)
            right_to_left_bottom = torch.cat([right_action_to_left_task, right_action_to_left_action], dim=2)
            right_to_left = torch.cat([right_to_left_top, right_to_left_bottom], dim=1)
        else:
            # No task decomposition, use original cross mask
            cross_mask = make_dual_arm_att_masks(
                left_pad_masks, right_pad_masks,
                left_att_masks, right_att_masks,
                self.config.action_horizon
            )
            left_to_right = cross_mask[:, :left_len, left_len:]
            right_to_left = cross_mask[:, left_len:, :left_len]
        
        # Combine: [prefix | left | right]
        row_left = torch.cat([left_to_prefix_2d, left_att_2d, left_to_right], dim=2)
        row_right = torch.cat([right_to_prefix_2d, right_to_left, right_att_2d], dim=2)
        full_att_2d_masks = torch.cat([row_left, row_right], dim=1)

        suffix_pad_masks = torch.cat([left_pad_masks, right_pad_masks], dim=1)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert_left.model.config._attn_implementation = "eager"  # noqa: SLF001
        self.paligemma_with_expert.gemma_expert_right.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, left_embs, right_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond, adarms_cond],
        )

        left_out = outputs_embeds[1]
        right_out = outputs_embeds[2]
        
        # Extract only action outputs (skip task prefix tokens)
        left_out = left_out[:, -self.config.action_horizon:]
        right_out = right_out[:, -self.config.action_horizon:]
        left_out = left_out.to(dtype=torch.float32)
        right_out = right_out.to(dtype=torch.float32)
        
        return self.action_out_proj_left(left_out), self.action_out_proj_right(right_out)
