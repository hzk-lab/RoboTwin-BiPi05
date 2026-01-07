"""Per-Arm VLM module for arm-specific reasoning.

This module implements a VLM with LoRA adapters that receives arm-specific
prompts from the Skill Selector and observation embeddings, producing
hidden states for the Action Expert to attend to.
"""

from typing import Literal, Tuple

import torch
from torch import nn
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING


class PerArmVLM(nn.Module):
    """VLM with LoRA for per-arm reasoning.
    
    Takes arm-specific prompt embeddings and observation embeddings,
    processes them through a VLM (with LoRA fine-tuning), and outputs
    hidden states for the Action Expert to attend to.
    """
    
    def __init__(
        self,
        vlm_config,
        use_lora: bool = True,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        """Initialize the Per-Arm VLM.
        
        Args:
            vlm_config: Configuration for the PaliGemma VLM.
            use_lora: Whether to use LoRA (indicated by variant name).
            precision: Model precision ("bfloat16" or "float32").
        """
        super().__init__()
        
        # Build HuggingFace config
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
        vlm_config_hf.text_config.adarms_cond_dim = None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"
        
        # Check if LoRA is configured (from variant name)
        self.use_lora = use_lora
        if hasattr(vlm_config, 'lora_rank') and vlm_config.lora_rank > 0:
            vlm_config_hf.text_config.lora_rank = vlm_config.lora_rank
        
        # Initialize the PaliGemma model
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        
        # Store config
        self.embed_dim = vlm_config.width
        
        # Apply precision
        self._apply_precision(precision)
    
    def _apply_precision(self, precision: Literal["bfloat16", "float32"]):
        """Apply the specified precision to the model."""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
            # Keep some params in float32 for stability
            params_to_keep_float32 = [
                "vision_tower.vision_model.embeddings.patch_embedding.weight",
                "vision_tower.vision_model.embeddings.patch_embedding.bias",
                "vision_tower.vision_model.embeddings.position_embedding.weight",
                "input_layernorm",
                "post_attention_layernorm",
                "model.norm",
            ]
            for name, param in self.named_parameters():
                if any(selector in name for selector in params_to_keep_float32):
                    param.data = param.data.to(dtype=torch.float32)
        elif precision == "float32":
            self.to(dtype=torch.float32)
        else:
            raise ValueError(f"Invalid precision: {precision}")
    
    def forward(
        self,
        arm_prompt_embs: torch.Tensor,
        obs_embs: torch.Tensor,
        obs_pad_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process arm prompt and observations through VLM.
        
        Args:
            arm_prompt_embs: Arm-specific prompt embeddings [batch, num_query_tokens, embed_dim]
            obs_embs: Observation embeddings [batch, obs_len, embed_dim]
            obs_pad_masks: Padding mask for observations [batch, obs_len], True for valid tokens
        
        Returns:
            vlm_output_embs: VLM hidden states [batch, prompt_len + obs_len, embed_dim]
        """
        # Concatenate arm prompt and observation embeddings
        # arm_prompt_embs: [B, N_prompt, D]
        # obs_embs: [B, N_obs, D]
        combined_embs = torch.cat([arm_prompt_embs, obs_embs], dim=1)
        
        batch_size = combined_embs.shape[0]
        seq_len = combined_embs.shape[1]
        
        # Create attention mask: all tokens can attend to all tokens (bidirectional for prefix)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=combined_embs.device)
        
        # If we have padding masks for observations, apply them
        if obs_pad_masks is not None:
            prompt_len = arm_prompt_embs.shape[1]
            prompt_mask = torch.ones(batch_size, prompt_len, dtype=torch.bool, device=combined_embs.device)
            attention_mask = torch.cat([prompt_mask, obs_pad_masks], dim=1)
        
        # Create position ids
        position_ids = torch.arange(seq_len, device=combined_embs.device).unsqueeze(0).expand(batch_size, -1)
        
        # Create 2D attention mask (bidirectional - all can see all valid tokens)
        att_2d_mask = attention_mask.unsqueeze(1) & attention_mask.unsqueeze(2)
        att_2d_mask_4d = att_2d_mask.unsqueeze(1)  # [B, 1, S, S]
        att_2d_mask_4d = torch.where(att_2d_mask_4d, 0.0, -2.3819763e38)
        
        # Forward through language model (not full PaliGemma, as we already have embeddings)
        outputs = self.paligemma.language_model(
            inputs_embeds=combined_embs,
            attention_mask=att_2d_mask_4d,
            position_ids=position_ids,
            use_cache=False,
        )
        
        return outputs.last_hidden_state

