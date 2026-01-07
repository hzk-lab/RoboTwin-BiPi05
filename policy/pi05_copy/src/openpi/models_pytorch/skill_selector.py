"""Skill Selector module for task decomposition into left/right arm prompts.

This module implements a frozen VLM that takes observation and instruction,
and outputs separate prompt embeddings for left and right arms using
learnable query tokens and cross-attention.
"""

from typing import Literal, Tuple

import torch
from torch import nn
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING


class SkillSelector(nn.Module):
    """Frozen high-level VLM for decomposing tasks into left/right arm sub-tasks.
    
    The VLM backbone is frozen, but the query tokens and cross-attention
    projections are trainable.
    """
    
    def __init__(
        self,
        vlm_config,
        num_query_tokens: int = 16,
        num_heads: int = 8,
        freeze_vlm: bool = True,
        freeze_queries: bool = False,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        """Initialize the Skill Selector.
        
        Args:
            vlm_config: Configuration for the PaliGemma VLM.
            num_query_tokens: Number of query tokens for each arm.
            num_heads: Number of attention heads for cross-attention.
            freeze_vlm: Whether to freeze the VLM backbone.
            freeze_queries: Whether to freeze the query tokens.
            precision: Model precision ("bfloat16" or "float32").
        """
        super().__init__()
        
        # Build HuggingFace config from our config
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
        
        # Initialize the PaliGemma model
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        
        # Freeze VLM backbone if requested
        if freeze_vlm:
            for param in self.paligemma.parameters():
                param.requires_grad = False
        
        # Get embedding dimension
        self.embed_dim = vlm_config.width
        self.num_query_tokens = num_query_tokens
        
        # Learnable query tokens for each arm
        self.left_queries = nn.Parameter(
            torch.randn(1, num_query_tokens, self.embed_dim) * 0.02,
            requires_grad=not freeze_queries
        )
        self.right_queries = nn.Parameter(
            torch.randn(1, num_query_tokens, self.embed_dim) * 0.02,
            requires_grad=not freeze_queries
        )
        
        # Cross-attention layers for extracting arm-specific prompts
        self.left_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.right_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        
        # Layer normalization and projection
        self.left_norm = nn.LayerNorm(self.embed_dim)
        self.right_norm = nn.LayerNorm(self.embed_dim)
        self.left_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.right_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
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
    
    def embed_image(self, image: torch.Tensor) -> torch.Tensor:
        """Embed image using VLM's vision encoder."""
        return self.paligemma.model.get_image_features(image)
    
    def embed_language_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed language tokens using VLM's language model."""
        return self.paligemma.language_model.embed_tokens(tokens)
    
    def forward(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate left and right arm prompt embeddings.
        
        Args:
            prefix_embs: VLM prefix embeddings [batch, seq_len, embed_dim]
                        (images + instruction combined)
            prefix_pad_masks: Padding mask [batch, seq_len], True for valid tokens
        
        Returns:
            left_prompt_embs: Left arm prompt embeddings [batch, num_query_tokens, embed_dim]
            right_prompt_embs: Right arm prompt embeddings [batch, num_query_tokens, embed_dim]
        """
        batch_size = prefix_embs.shape[0]
        
        # Expand query tokens to batch size
        left_queries = self.left_queries.expand(batch_size, -1, -1)
        right_queries = self.right_queries.expand(batch_size, -1, -1)
        
        # Convert padding mask for attention (True means masked in MultiheadAttention)
        # Our prefix_pad_masks is True for valid tokens, so we invert
        attn_mask = ~prefix_pad_masks
        
        # Left arm: queries attend to VLM hidden states
        left_prompt, _ = self.left_cross_attn(
            query=left_queries,
            key=prefix_embs,
            value=prefix_embs,
            key_padding_mask=attn_mask,
        )
        left_prompt = self.left_norm(left_prompt)
        left_prompt = self.left_proj(left_prompt)
        
        # Right arm: queries attend to VLM hidden states
        right_prompt, _ = self.right_cross_attn(
            query=right_queries,
            key=prefix_embs,
            value=prefix_embs,
            key_padding_mask=attn_mask,
        )
        right_prompt = self.right_norm(right_prompt)
        right_prompt = self.right_proj(right_prompt)
        
        return left_prompt, right_prompt

