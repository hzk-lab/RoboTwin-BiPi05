"""Skill Selector module for Triple-VLM dual-arm architecture.

The Skill Selector is a frozen VLM that takes observation and instruction as input,
and outputs left/right arm-specific prompts through learned query tokens and cross-attention.
"""

from typing import Literal, Tuple

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING


class SkillSelector(nn.Module):
    """Frozen high-level VLM that decomposes task into left/right arm sub-task prompts.
    
    This module:
    1. Uses a frozen PaliGemma VLM to process images and instruction
    2. Extracts left/right arm-specific prompt embeddings via cross-attention
       with learnable query tokens
    
    The VLM backbone is completely frozen. Only the query tokens and cross-attention
    layers are trainable (though typically we freeze these too after pre-training).
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
        """Initialize Skill Selector.
        
        Args:
            vlm_config: Configuration for the PaliGemma VLM
            num_query_tokens: Number of learnable query tokens per arm
            num_heads: Number of attention heads for cross-attention
            freeze_vlm: Whether to freeze VLM backbone (should be True)
            freeze_queries: Whether to freeze query tokens (True for inference)
            precision: Model precision
        """
        super().__init__()
        
        self.num_query_tokens = num_query_tokens
        self.num_heads = num_heads
        
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
        
        embed_dim = vlm_config.width
        
        # Learnable query tokens for left and right arms
        self.left_queries = nn.Parameter(torch.randn(1, num_query_tokens, embed_dim) * 0.02)
        self.right_queries = nn.Parameter(torch.randn(1, num_query_tokens, embed_dim) * 0.02)
        
        # Cross-attention layers to extract arm-specific prompts
        self.left_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.right_cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.0, batch_first=True
        )
        
        # Layer norms
        self.left_norm = nn.LayerNorm(embed_dim)
        self.right_norm = nn.LayerNorm(embed_dim)
        
        # Output projections
        self.left_proj = nn.Linear(embed_dim, embed_dim)
        self.right_proj = nn.Linear(embed_dim, embed_dim)
        
        # Freeze VLM backbone
        if freeze_vlm:
            for param in self.paligemma.parameters():
                param.requires_grad = False
        
        # Optionally freeze query tokens (for inference or after pre-training)
        if freeze_queries:
            self.left_queries.requires_grad = False
            self.right_queries.requires_grad = False
            for param in self.left_cross_attn.parameters():
                param.requires_grad = False
            for param in self.right_cross_attn.parameters():
                param.requires_grad = False
            for param in self.left_norm.parameters():
                param.requires_grad = False
            for param in self.right_norm.parameters():
                param.requires_grad = False
            for param in self.left_proj.parameters():
                param.requires_grad = False
            for param in self.right_proj.parameters():
                param.requires_grad = False
        
        self.to_precision(precision)
    
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
    
    def forward_with_vlm(
        self,
        images: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        img_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass including VLM encoding.
        
        Args:
            images: Input images [batch, num_images, C, H, W]
            lang_tokens: Language tokens [batch, seq_len]
            lang_masks: Language mask [batch, seq_len]
            img_masks: Image mask [batch, num_images]
        
        Returns:
            left_prompt_embs: Left arm prompt embeddings
            right_prompt_embs: Right arm prompt embeddings
            vlm_hidden: VLM hidden states (for potential reuse)
            vlm_pad_mask: VLM padding mask
        """
        # Embed images
        batch_size = images.shape[0]
        num_images = images.shape[1]
        
        # Flatten images for VLM
        images_flat = images.view(-1, *images.shape[2:])  # [B*N, C, H, W]
        image_embs = self.embed_image(images_flat)  # [B*N, num_patches, dim]
        image_embs = image_embs.view(batch_size, num_images, -1, image_embs.shape[-1])
        
        # Embed language tokens
        lang_embs = self.embed_language_tokens(lang_tokens)
        
        # Combine image and language embeddings
        # Interleave based on image positions (simplified: images first, then text)
        prefix_parts = []
        prefix_masks = []
        
        for i in range(num_images):
            if img_masks is not None:
                valid_img = img_masks[:, i:i+1].unsqueeze(-1)  # [B, 1, 1]
                prefix_parts.append(image_embs[:, i] * valid_img.float())
                prefix_masks.append(img_masks[:, i:i+1].expand(-1, image_embs.shape[2]))
            else:
                prefix_parts.append(image_embs[:, i])
                prefix_masks.append(torch.ones(batch_size, image_embs.shape[2], 
                                               dtype=torch.bool, device=images.device))
        
        prefix_parts.append(lang_embs)
        prefix_masks.append(lang_masks)
        
        prefix_embs = torch.cat(prefix_parts, dim=1)
        prefix_pad_masks = torch.cat(prefix_masks, dim=1)
        
        # Get arm-specific prompts
        left_prompt, right_prompt = self.forward(prefix_embs, prefix_pad_masks)
        
        return left_prompt, right_prompt, prefix_embs, prefix_pad_masks

