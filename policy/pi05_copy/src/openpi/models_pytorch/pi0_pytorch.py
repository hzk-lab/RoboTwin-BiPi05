import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
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


class PI0TripleVLMPytorch(nn.Module):
    """Triple VLM architecture for dual-arm control.
    
    This architecture consists of:
    - 1 Skill Selector VLM (frozen): generates left/right arm prompts
    - 2 Per-Arm VLMs (LoRA fine-tuned): process arm-specific prompts + observations
    - 2 Action Experts (frozen): generate actions, with cross-attention between arms
    - 1 Gate Network (trained): controls cross-attention level between arms
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        self.half_action_dim = config.half_action_dim
        self.actual_arm_dim = getattr(config, 'actual_arm_dim', 7)
        
        # Get configs for each component
        skill_selector_variant = getattr(config, 'skill_selector_variant', 'gemma_2b')
        per_arm_vlm_left_variant = getattr(config, 'per_arm_vlm_left_variant', 'gemma_2b_lora')
        per_arm_vlm_right_variant = getattr(config, 'per_arm_vlm_right_variant', 'gemma_2b_lora')
        ae_left_variant = getattr(config, 'action_expert_left_variant', None) or config.action_expert_variant
        ae_right_variant = getattr(config, 'action_expert_right_variant', None) or config.action_expert_variant
        
        skill_selector_config = _gemma.get_config(skill_selector_variant)
        per_arm_vlm_left_config = _gemma.get_config(per_arm_vlm_left_variant)
        per_arm_vlm_right_config = _gemma.get_config(per_arm_vlm_right_variant)
        ae_left_config = _gemma.get_config(ae_left_variant)
        ae_right_config = _gemma.get_config(ae_right_variant)
        
        # ===== Initialize Components =====
        
        # 1. Skill Selector (frozen VLM with trainable query tokens)
        from openpi.models_pytorch.skill_selector import SkillSelector
        self.skill_selector = SkillSelector(
            vlm_config=skill_selector_config,
            num_query_tokens=getattr(config, 'task_decomposition_num_queries', 16),
            num_heads=getattr(config, 'task_decomposition_num_heads', 8),
            freeze_vlm=True,
            freeze_queries=False,  # Query tokens are trainable
            precision=config.dtype,
        )
        
        # 2. Per-Arm VLMs (LoRA fine-tuned)
        from openpi.models_pytorch.per_arm_vlm import PerArmVLM
        self.vlm_left = PerArmVLM(
            vlm_config=per_arm_vlm_left_config,
            use_lora="lora" in per_arm_vlm_left_variant,
            precision=config.dtype,
        )
        self.vlm_right = PerArmVLM(
            vlm_config=per_arm_vlm_right_config,
            use_lora="lora" in per_arm_vlm_right_variant,
            precision=config.dtype,
        )
        
        # 3. Dual Action Experts (frozen)
        from openpi.models_pytorch.gemma_pytorch import DualActionExpert
        self.dual_ae = DualActionExpert(
            ae_left_config=ae_left_config,
            ae_right_config=ae_right_config,
            use_adarms=self.pi05,
            precision=config.dtype,
        )
        # Freeze AE parameters
        for param in self.dual_ae.parameters():
            param.requires_grad = False
        
        # Store dimensions
        self.vlm_dim = skill_selector_config.width
        self.ae_dim = ae_left_config.width
        
        # 4. Gate Network (trained)
        self.peca_enabled = getattr(config, 'peca_enabled', True)
        if self.peca_enabled:
            from openpi.models_pytorch.gate_network import GateNetwork
            self.gate_network = GateNetwork(
                hidden_dim=self.ae_dim,
                mlp_hidden=getattr(config, 'gate_mlp_hidden', 256),
                precision=config.dtype,
            )
        
        # 5. Action projections (per arm)
        self.action_in_proj_left = nn.Linear(self.half_action_dim, self.ae_dim)
        self.action_in_proj_right = nn.Linear(self.half_action_dim, self.ae_dim)
        self.action_out_proj_left = nn.Linear(self.ae_dim, self.half_action_dim)
        self.action_out_proj_right = nn.Linear(self.ae_dim, self.half_action_dim)
        
        # 6. Time MLP for adaRMSNorm (if pi05)
        if self.pi05:
            self.time_mlp_in = nn.Linear(self.ae_dim, self.ae_dim)
            self.time_mlp_out = nn.Linear(self.ae_dim, self.ae_dim)
        
        # Store configs
        self.gate_threshold = getattr(config, 'gate_threshold', 0.5)
        self.peca_lambda = getattr(config, 'peca_lambda', 0.1)
        self.l1_lambda = getattr(config, 'l1_lambda', 0.01)
        self.sticky_lambda = getattr(config, 'sticky_lambda', 0.01)
        
        torch.set_float32_matmul_precision("high")
        self.gradient_checkpointing_enabled = False
        
        logging.info("Initialized PI0TripleVLMPytorch with:")
        logging.info(f"  - Skill Selector: {skill_selector_variant} (frozen)")
        logging.info(f"  - Per-Arm VLMs: L={per_arm_vlm_left_variant}, R={per_arm_vlm_right_variant}")
        logging.info(f"  - Action Experts: L={ae_left_variant}, R={ae_right_variant} (frozen)")
        logging.info(f"  - PECA enabled: {self.peca_enabled}")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        logging.info("Enabled gradient checkpointing for PI0TripleVLMPytorch")
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        logging.info("Disabled gradient checkpointing for PI0TripleVLMPytorch")
    
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
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)
    
    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)
    
    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Embed images and language tokens using Skill Selector's VLM."""
        embs = []
        pad_masks = []
        att_masks = []
        
        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self.skill_selector.embed_image(img)
            bsize, num_img_embs = img_emb.shape[:2]
            
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs
        
        # Process language tokens
        lang_emb = self.skill_selector.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        
        return embs, pad_masks, att_masks
    
    def embed_suffix_per_arm(self, noisy_actions_left, noisy_actions_right, timestep):
        """Embed actions for each arm separately."""
        # Time embedding
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.ae_dim, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)
        
        if self.pi05:
            # Time MLP for adaRMS
            time_emb = self.time_mlp_in(time_emb)
            time_emb = F.silu(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            adarms_cond = F.silu(time_emb)
        else:
            adarms_cond = None
        
        # Project actions
        action_emb_left = self.action_in_proj_left(noisy_actions_left)
        action_emb_right = self.action_in_proj_right(noisy_actions_right)
        
        return action_emb_left, action_emb_right, adarms_cond
    
    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Full training forward pass with PECA loss computation."""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        
        # Split actions into left/right (first half and second half of action_dim)
        actions_left = actions[..., :self.half_action_dim]
        actions_right = actions[..., self.half_action_dim:]
        noise_left = noise[..., :self.half_action_dim]
        noise_right = noise[..., self.half_action_dim:]
        
        # Flow matching: x_t = t * noise + (1-t) * actions
        time_expanded = time[:, None, None]
        x_t_left = time_expanded * noise_left + (1 - time_expanded) * actions_left
        x_t_right = time_expanded * noise_right + (1 - time_expanded) * actions_right
        u_t_left = noise_left - actions_left
        u_t_right = noise_right - actions_right
        
        # === Step 1: Skill Selector generates arm-specific prompts ===
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        left_prompt, right_prompt = self.skill_selector(prefix_embs, prefix_pad_masks)
        
        # === Step 2: Per-Arm VLMs process prompts + observations ===
        vlm_left_out = self.vlm_left(left_prompt, prefix_embs, prefix_pad_masks)
        vlm_right_out = self.vlm_right(right_prompt, prefix_embs, prefix_pad_masks)
        
        # === Step 3: Embed action suffixes ===
        suffix_left, suffix_right, adarms_cond = self.embed_suffix_per_arm(
            x_t_left, x_t_right, time
        )
        
        # === Step 4: Action Experts with Cross-Attention ===
        batch_size = actions.shape[0]
        seq_len = vlm_left_out.shape[1] + suffix_left.shape[1]
        
        # Create attention masks (causal for action tokens)
        att_mask_4d = torch.ones(batch_size, 1, seq_len, seq_len, device=actions.device)
        att_mask_4d = torch.tril(att_mask_4d)
        att_mask_4d = torch.where(att_mask_4d.bool(), 0.0, -2.3819763e38)
        
        position_ids = torch.arange(seq_len, device=actions.device).unsqueeze(0).expand(batch_size, -1)
        
        if self.peca_enabled:
            # Three-path forward for PECA
            # Path 1: Full cross-attention (alpha=0)
            output_left_on, output_right_on = self.dual_ae.forward_with_cross_attention(
                vlm_left_out, vlm_right_out,
                suffix_left, suffix_right,
                att_mask_4d, att_mask_4d,
                position_ids,
                alpha_t=torch.zeros(batch_size, 1, device=actions.device),
                adarms_cond_left=adarms_cond,
                adarms_cond_right=adarms_cond,
            )
            
            # Path 2: No cross-attention (alpha=1)
            output_left_off, output_right_off = self.dual_ae.forward_with_cross_attention(
                vlm_left_out, vlm_right_out,
                suffix_left, suffix_right,
                att_mask_4d, att_mask_4d,
                position_ids,
                alpha_t=torch.ones(batch_size, 1, device=actions.device),
                adarms_cond_left=adarms_cond,
                adarms_cond_right=adarms_cond,
            )
            
            # Compute BC losses for both paths
            v_t_left_on = self.action_out_proj_left(output_left_on.float())
            v_t_right_on = self.action_out_proj_right(output_right_on.float())
            v_t_left_off = self.action_out_proj_left(output_left_off.float())
            v_t_right_off = self.action_out_proj_right(output_right_off.float())
            
            loss_left_on = F.mse_loss(u_t_left, v_t_left_on, reduction='none').mean(dim=(1, 2))
            loss_right_on = F.mse_loss(u_t_right, v_t_right_on, reduction='none').mean(dim=(1, 2))
            loss_left_off = F.mse_loss(u_t_left, v_t_left_off, reduction='none').mean(dim=(1, 2))
            loss_right_off = F.mse_loss(u_t_right, v_t_right_off, reduction='none').mean(dim=(1, 2))
            
            loss_on = loss_left_on + loss_right_on
            loss_off = loss_left_off + loss_right_off
            
            # Compute gate prediction
            # Use hidden states from the "on" path
            alpha_t = self.gate_network(output_left_on, output_right_on)
            
            # Compute gate label: 0 if cooperation helps, 1 if it hurts
            from openpi.models_pytorch.gate_network import (
                compute_gate_bce_loss, compute_l1_regularization, 
                compute_peca_loss, compute_gate_label
            )
            
            alpha_label = compute_gate_label(loss_on, loss_off)
            gate_bce_loss = compute_gate_bce_loss(alpha_t, alpha_label)
            l1_loss = compute_l1_regularization(alpha_t, self.l1_lambda)
            peca_loss = compute_peca_loss(loss_on, loss_off, alpha_t, self.peca_lambda)
            
            # Path 3: Actual gated output using predicted alpha
            output_left, output_right = self.dual_ae.forward_with_cross_attention(
                vlm_left_out, vlm_right_out,
                suffix_left, suffix_right,
                att_mask_4d, att_mask_4d,
                position_ids,
                alpha_t=alpha_t.detach(),  # Detach to avoid double backward
                adarms_cond_left=adarms_cond,
                adarms_cond_right=adarms_cond,
            )
            
            v_t_left = self.action_out_proj_left(output_left.float())
            v_t_right = self.action_out_proj_right(output_right.float())
            
            bc_loss_left = F.mse_loss(u_t_left, v_t_left, reduction='none')
            bc_loss_right = F.mse_loss(u_t_right, v_t_right, reduction='none')
            bc_loss = bc_loss_left.mean() + bc_loss_right.mean()
            
            total_loss = bc_loss + gate_bce_loss + l1_loss + peca_loss
            return total_loss
            
        else:
            # Simple forward without PECA
            output_left, output_right = self.dual_ae.forward_with_cross_attention(
                vlm_left_out, vlm_right_out,
                suffix_left, suffix_right,
                att_mask_4d, att_mask_4d,
                position_ids,
                alpha_t=None,  # Default to independent
                adarms_cond_left=adarms_cond,
                adarms_cond_right=adarms_cond,
            )
            
            v_t_left = self.action_out_proj_left(output_left.float())
            v_t_right = self.action_out_proj_right(output_right.float())
            
            loss_left = F.mse_loss(u_t_left, v_t_left, reduction='none')
            loss_right = F.mse_loss(u_t_right, v_t_right, reduction='none')
            
            return loss_left.mean() + loss_right.mean()
    
    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Generate actions using the denoising process."""
        bsize = observation.state.shape[0]
        action_shape = (bsize, self.config.action_horizon, self.config.action_dim)
        
        if noise is None:
            noise = self.sample_noise(action_shape, device)
        
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        
        # === Step 1: Skill Selector generates arm-specific prompts ===
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        left_prompt, right_prompt = self.skill_selector(prefix_embs, prefix_pad_masks)
        
        # === Step 2: Per-Arm VLMs process prompts + observations ===
        vlm_left_out = self.vlm_left(left_prompt, prefix_embs, prefix_pad_masks)
        vlm_right_out = self.vlm_right(right_prompt, prefix_embs, prefix_pad_masks)
        
        # Denoising loop
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            
            # Split noise for each arm
            x_t_left = x_t[..., :self.half_action_dim]
            x_t_right = x_t[..., self.half_action_dim:]
            
            # Embed action suffixes
            suffix_left, suffix_right, adarms_cond = self.embed_suffix_per_arm(
                x_t_left, x_t_right, expanded_time
            )
            
            # Create attention masks
            seq_len = vlm_left_out.shape[1] + suffix_left.shape[1]
            att_mask_4d = torch.ones(bsize, 1, seq_len, seq_len, device=device)
            att_mask_4d = torch.tril(att_mask_4d)
            att_mask_4d = torch.where(att_mask_4d.bool(), 0.0, -2.3819763e38)
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsize, -1)
            
            # Compute alpha (hard gate during inference)
            if self.peca_enabled:
                # First pass to get hidden states for gate
                output_left_temp, output_right_temp = self.dual_ae.forward_with_cross_attention(
                    vlm_left_out, vlm_right_out,
                    suffix_left, suffix_right,
                    att_mask_4d, att_mask_4d,
                    position_ids,
                    alpha_t=torch.ones(bsize, 1, device=device),  # Independent pass
                    adarms_cond_left=adarms_cond,
                    adarms_cond_right=adarms_cond,
                )
                alpha_t_soft = self.gate_network(output_left_temp, output_right_temp)
                alpha_t = self.gate_network.get_hard_gate(alpha_t_soft, self.gate_threshold)
            else:
                alpha_t = torch.ones(bsize, 1, device=device)
            
            # Forward with computed alpha
            output_left, output_right = self.dual_ae.forward_with_cross_attention(
                vlm_left_out, vlm_right_out,
                suffix_left, suffix_right,
                att_mask_4d, att_mask_4d,
                position_ids,
                alpha_t=alpha_t,
                adarms_cond_left=adarms_cond,
                adarms_cond_right=adarms_cond,
            )
            
            v_t_left = self.action_out_proj_left(output_left.float())
            v_t_right = self.action_out_proj_right(output_right.float())
            
            # Combine and Euler step
            v_t = torch.cat([v_t_left, v_t_right], dim=-1)
            x_t = x_t + dt * v_t
            time += dt
        
        # Extract actual action dimensions from padded output
        # x_t shape: [batch, horizon, 32]
        # Left: x_t[..., :16][..., :7], Right: x_t[..., 16:][..., :7]
        left_actions = x_t[..., :self.half_action_dim][..., :self.actual_arm_dim]
        right_actions = x_t[..., self.half_action_dim:][..., :self.actual_arm_dim]
        actions = torch.cat([left_actions, right_actions], dim=-1)
        
        return actions
