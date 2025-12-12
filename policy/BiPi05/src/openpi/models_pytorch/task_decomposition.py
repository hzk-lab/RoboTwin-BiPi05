"""Task Decomposition Module for dual-arm bimanual control.

This module decomposes a unified VLM prefix embedding into separate left/right arm
specific embeddings using Cross-Attention mechanism.
"""

import math
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class TaskDecompositionModule(nn.Module):
    """Decomposes unified VLM prefix into left/right arm specific prefixes.
    
    Uses learnable query tokens and Cross-Attention to extract arm-specific
    information from the shared VLM representation.
    
    Architecture:
        prefix_embs (unified) 
            │
            ├──> left_cross_attn(left_queries, prefix_embs) ──> left_prefix
            │
            └──> right_cross_attn(right_queries, prefix_embs) ──> right_prefix
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_query_tokens: int = 16,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        """Initialize TaskDecompositionModule.
        
        Args:
            embed_dim: Dimension of embeddings (should match VLM hidden dim)
            num_heads: Number of attention heads for Cross-Attention
            num_query_tokens: Number of learnable query tokens per arm
            dropout: Dropout probability
            use_layer_norm: Whether to apply layer normalization
            precision: Model precision
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_query_tokens = num_query_tokens
        
        # Learnable arm-specific query tokens
        # These will learn to extract relevant information for each arm
        self.left_arm_queries = nn.Parameter(
            torch.randn(1, num_query_tokens, embed_dim) * 0.02
        )
        self.right_arm_queries = nn.Parameter(
            torch.randn(1, num_query_tokens, embed_dim) * 0.02
        )
        
        # Cross-Attention layers for each arm
        self.left_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.right_cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.left_norm = nn.LayerNorm(embed_dim)
            self.right_norm = nn.LayerNorm(embed_dim)
            self.prefix_norm = nn.LayerNorm(embed_dim)
        
        # Output projection with residual-style MLP
        self.left_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.right_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        
        # Set precision
        self._set_precision(precision)
        
        # Initialize weights
        self._init_weights()
    
    def _set_precision(self, precision: Literal["bfloat16", "float32"]):
        """Set model precision."""
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
        
        # Keep layer norm in float32 for stability
        if self.use_layer_norm:
            self.left_norm.to(dtype=torch.float32)
            self.right_norm.to(dtype=torch.float32)
            self.prefix_norm.to(dtype=torch.float32)
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in [self.left_cross_attn, self.right_cross_attn]:
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                nn.init.xavier_uniform_(module.out_proj.weight)
        
        for proj in [self.left_proj, self.right_proj]:
            for layer in proj:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompose unified prefix into left/right arm specific prefixes.
        
        Args:
            prefix_embs: Unified prefix embeddings from VLM [batch, seq_len, embed_dim]
            prefix_pad_mask: Padding mask for prefix [batch, seq_len], True for valid tokens
            
        Returns:
            left_prefix: Left arm specific prefix [batch, num_query_tokens, embed_dim]
            right_prefix: Right arm specific prefix [batch, num_query_tokens, embed_dim]
        """
        batch_size = prefix_embs.shape[0]
        
        # Normalize prefix if enabled
        if self.use_layer_norm:
            prefix_embs_normed = self.prefix_norm(prefix_embs.float()).to(prefix_embs.dtype)
        else:
            prefix_embs_normed = prefix_embs
        
        # Expand queries to batch size
        left_queries = self.left_arm_queries.expand(batch_size, -1, -1)
        right_queries = self.right_arm_queries.expand(batch_size, -1, -1)
        
        # Convert padding mask to attention mask format for MultiheadAttention
        # key_padding_mask: True means ignore (opposite of our convention)
        key_padding_mask = None
        if prefix_pad_mask is not None:
            key_padding_mask = ~prefix_pad_mask  # Invert: True -> ignore
        
        # Cross-Attention for left arm
        # Query: left_queries, Key/Value: prefix_embs
        left_attn_out, _ = self.left_cross_attn(
            query=left_queries,
            key=prefix_embs_normed,
            value=prefix_embs_normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        
        # Cross-Attention for right arm
        right_attn_out, _ = self.right_cross_attn(
            query=right_queries,
            key=prefix_embs_normed,
            value=prefix_embs_normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        
        # Apply layer norm and projection with residual connection
        if self.use_layer_norm:
            left_normed = self.left_norm(left_attn_out.float()).to(left_attn_out.dtype)
            right_normed = self.right_norm(right_attn_out.float()).to(right_attn_out.dtype)
        else:
            left_normed = left_attn_out
            right_normed = right_attn_out
        
        # Project with residual
        left_prefix = left_attn_out + self.left_proj(left_normed)
        right_prefix = right_attn_out + self.right_proj(right_normed)
        
        return left_prefix, right_prefix
    
    def get_num_query_tokens(self) -> int:
        """Return the number of query tokens per arm."""
        return self.num_query_tokens


class TaskDecompositionModuleV2(nn.Module):
    """Alternative implementation using shared attention with arm-specific projections.
    
    This version uses a single shared attention mechanism followed by 
    arm-specific projection heads, which may be more parameter efficient.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_query_tokens: int = 16,
        dropout: float = 0.0,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_query_tokens = num_query_tokens
        
        # Shared learnable queries (will be split or differentiated by position)
        self.shared_queries = nn.Parameter(
            torch.randn(1, num_query_tokens * 2, embed_dim) * 0.02
        )
        
        # Shared Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Arm-specific projection heads
        self.left_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.right_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self._set_precision(precision)
    
    def _set_precision(self, precision: Literal["bfloat16", "float32"]):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
    
    def forward(
        self,
        prefix_embs: torch.Tensor,
        prefix_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = prefix_embs.shape[0]
        
        # Expand queries
        queries = self.shared_queries.expand(batch_size, -1, -1)
        
        # Convert padding mask
        key_padding_mask = None
        if prefix_pad_mask is not None:
            key_padding_mask = ~prefix_pad_mask
        
        # Shared attention
        attn_out, _ = self.cross_attn(
            query=queries,
            key=prefix_embs,
            value=prefix_embs,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        
        # Split and project
        left_out = attn_out[:, :self.num_query_tokens]
        right_out = attn_out[:, self.num_query_tokens:]
        
        left_prefix = left_out + self.left_head(left_out)
        right_prefix = right_out + self.right_head(right_out)
        
        return left_prefix, right_prefix
    
    def get_num_query_tokens(self) -> int:
        return self.num_query_tokens

