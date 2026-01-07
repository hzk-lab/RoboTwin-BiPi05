"""Per-Arm VLM module for Triple-VLM dual-arm architecture.

Each Per-Arm VLM is a PaliGemma with LoRA that processes arm-specific prompts
and observations to generate hidden states for the Action Expert to attend to.
"""

from typing import Literal, Optional

import torch
from torch import nn

from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING


class PerArmVLM(nn.Module):
    """VLM with LoRA for per-arm reasoning.
    
    This module:
    1. Takes arm-specific prompt (from Skill Selector) + observation as input
    2. Processes them through a PaliGemma with LoRA
    3. Outputs hidden states for the Action Expert to attend to
    
    The VLM backbone is frozen, only LoRA parameters are trainable.
    """
    
    def __init__(
        self,
        vlm_config,
        use_lora: bool = True,
        freeze_backbone: bool = True,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        """Initialize Per-Arm VLM.
        
        Args:
            vlm_config: Configuration for PaliGemma (may include LoRA settings)
            use_lora: Whether this VLM uses LoRA (should be True for per-arm VLMs)
            freeze_backbone: Whether to freeze non-LoRA parameters
            precision: Model precision
        """
        super().__init__()
        
        self.use_lora = use_lora
        
        # Create VLM config
        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = False
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"
        
        # Initialize PaliGemma
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        
        self.embed_dim = vlm_config.width
        
        # Freeze backbone if specified (LoRA params will be unfrozen later)
        if freeze_backbone:
            for param in self.paligemma.parameters():
                param.requires_grad = False
            
            # If using LoRA, unfreeze LoRA parameters
            if use_lora:
                self._unfreeze_lora_params()
        
        self.to_precision(precision)
    
    def _unfreeze_lora_params(self):
        """Unfreeze LoRA parameters if they exist."""
        for name, param in self.paligemma.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
    
    def to_precision(self, precision: Literal["bfloat16", "float32"]):
        """Set model precision."""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")
    
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed image using VLM's vision encoder."""
        return self.paligemma.model.get_image_features(image)
    
    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using VLM's language model."""
        return self.paligemma.language_model.embed_tokens(tokens)
    
    def forward(
        self,
        arm_prompt_embs: torch.Tensor,
        obs_embs: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process arm-specific prompt and observation.
        
        Args:
            arm_prompt_embs: Arm-specific prompt embeddings from Skill Selector
                            [batch, num_query_tokens, embed_dim]
            obs_embs: Observation embeddings (images + state)
                     [batch, obs_seq_len, embed_dim]
            pad_mask: Padding mask for the combined sequence
            attention_mask: Attention mask (optional)
        
        Returns:
            hidden_states: VLM hidden states [batch, seq_len, embed_dim]
                          for Action Expert to attend to
        """
        # Concatenate arm prompt with observation
        # [arm_prompt, obs] -> VLM
        combined_embs = torch.cat([arm_prompt_embs, obs_embs], dim=1)
        
        # Build attention mask if not provided
        batch_size = combined_embs.shape[0]
        seq_len = combined_embs.shape[1]
        
        if attention_mask is None:
            # Causal attention mask
            attention_mask = torch.ones(
                batch_size, 1, seq_len, seq_len,
                dtype=combined_embs.dtype,
                device=combined_embs.device
            )
            # Make it causal
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=combined_embs.device),
                diagonal=1
            ).bool()
            attention_mask = attention_mask.masked_fill(causal_mask, float('-inf'))
        
        # Forward through VLM language model
        outputs = self.paligemma.language_model.model(
            inputs_embeds=combined_embs,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Return the last hidden state
        hidden_states = outputs.last_hidden_state
        
        return hidden_states
    
    def forward_with_images(
        self,
        arm_prompt_embs: torch.Tensor,
        images: torch.Tensor,
        img_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with image inputs.
        
        Args:
            arm_prompt_embs: Arm-specific prompt from Skill Selector
            images: Input images [batch, num_images, C, H, W]
            img_masks: Image validity mask [batch, num_images]
        
        Returns:
            hidden_states: VLM output hidden states
        """
        batch_size = images.shape[0]
        num_images = images.shape[1]
        
        # Embed images
        images_flat = images.view(-1, *images.shape[2:])
        image_embs = self.embed_image(images_flat)
        image_embs = image_embs.view(batch_size, num_images, -1, image_embs.shape[-1])
        
        # Flatten image embeddings
        # [batch, num_images, num_patches, dim] -> [batch, num_images * num_patches, dim]
        obs_embs = image_embs.view(batch_size, -1, image_embs.shape[-1])
        
        # Apply image masks if provided
        if img_masks is not None:
            # Expand mask to match patch dimensions
            num_patches_per_image = image_embs.shape[2]
            expanded_mask = img_masks.unsqueeze(-1).expand(-1, -1, num_patches_per_image)
            expanded_mask = expanded_mask.reshape(batch_size, -1).unsqueeze(-1)
            obs_embs = obs_embs * expanded_mask.float()
        
        return self.forward(arm_prompt_embs, obs_embs)


class DualPerArmVLM(nn.Module):
    """Container for left and right Per-Arm VLMs.
    
    Manages two independent VLMs with their own LoRA weights.
    """
    
    def __init__(
        self,
        vlm_config_left,
        vlm_config_right,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        """Initialize dual Per-Arm VLMs.
        
        Args:
            vlm_config_left: Config for left arm VLM (with LoRA_L)
            vlm_config_right: Config for right arm VLM (with LoRA_R)
            precision: Model precision
        """
        super().__init__()
        
        self.vlm_left = PerArmVLM(vlm_config_left, use_lora=True, precision=precision)
        self.vlm_right = PerArmVLM(vlm_config_right, use_lora=True, precision=precision)
    
    def forward(
        self,
        left_prompt_embs: torch.Tensor,
        right_prompt_embs: torch.Tensor,
        left_obs_embs: torch.Tensor,
        right_obs_embs: torch.Tensor,
    ):
        """Forward pass for both arms.
        
        Args:
            left_prompt_embs: Left arm prompt from Skill Selector
            right_prompt_embs: Right arm prompt from Skill Selector
            left_obs_embs: Observation embeddings for left arm
            right_obs_embs: Observation embeddings for right arm
        
        Returns:
            left_hidden: Hidden states from left VLM
            right_hidden: Hidden states from right VLM
        """
        left_hidden = self.vlm_left(left_prompt_embs, left_obs_embs)
        right_hidden = self.vlm_right(right_prompt_embs, right_obs_embs)
        
        return left_hidden, right_hidden

