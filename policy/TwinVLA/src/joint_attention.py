"""
Joint Attention Module for TwinVLA

实现 TwinVLA 的核心 Joint Attention 机制，让左右臂在 Transformer 层互相交流。

Attention Mask 设计:
Token 顺序: [Common Input] [Left Arm] [Right Arm]

Mask 矩阵 (1=可见, 0=不可见):
              Common  Left  Right
Common          1      0      0     # Common 只看自己 (双向)
Left            1      1      1*    # Left 看 Common + 自己 + Right(因果)
Right           1      1*     1     # Right 看 Common + Left(因果) + 自己

*: 因果约束 - 只能看到当前时间步及之前的 token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def create_causal_joint_mask(
    common_len: int,
    left_len: int,
    right_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """
    创建 TwinVLA 的因果 Joint Attention Mask。
    
    Args:
        common_len: Common input tokens 数量 (ego view + language)
        left_len: Left arm tokens 数量 (wrist cam + proprio + action)
        right_len: Right arm tokens 数量
        device: 设备
        dtype: 数据类型
        
    Returns:
        attention_mask: [total_len, total_len] 的 mask，True 表示可以 attend
    """
    total_len = common_len + left_len + right_len
    
    # 初始化全 False (不可见)
    mask = torch.zeros(total_len, total_len, dtype=dtype, device=device)
    
    # 1. Common tokens 内部双向 attention
    mask[:common_len, :common_len] = True
    
    # 2. Left arm tokens 可以看 Common
    mask[common_len:common_len+left_len, :common_len] = True
    
    # 3. Left arm tokens 内部因果 attention
    left_start = common_len
    left_end = common_len + left_len
    for i in range(left_len):
        mask[left_start + i, left_start:left_start + i + 1] = True
    
    # 4. Right arm tokens 可以看 Common
    mask[common_len+left_len:, :common_len] = True
    
    # 5. Right arm tokens 内部因果 attention
    right_start = common_len + left_len
    for i in range(right_len):
        mask[right_start + i, right_start:right_start + i + 1] = True
    
    # 6. 左右臂之间的交互 (因果)
    # Left 可以看 Right 的历史 token (因果约束)
    for i in range(left_len):
        # Left token i 可以看到 Right 的前 i 个 token
        if i > 0:
            mask[left_start + i, right_start:right_start + i] = True
    
    # Right 可以看 Left 的历史 token (因果约束)  
    for i in range(right_len):
        # Right token i 可以看到 Left 的前 i+1 个 token (包括当前)
        mask[right_start + i, left_start:left_start + min(i + 1, left_len)] = True
    
    return mask


def create_bimanual_attention_mask(
    batch_size: int,
    common_len: int,
    left_len: int,
    right_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    创建批量的 4D attention mask，用于 Transformer。
    
    Returns:
        mask: [batch_size, 1, total_len, total_len]
              True/0.0 表示可以 attend, False/-inf 表示 mask 掉
    """
    # 创建基础 mask
    base_mask = create_causal_joint_mask(common_len, left_len, right_len, device)
    
    # 扩展到 batch 维度 [B, 1, S, S]
    mask = base_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
    
    # 转换为 attention 权重格式 (True -> 0, False -> -inf)
    attention_mask = torch.where(mask, 0.0, -2.3819763e38)
    
    return attention_mask


class JointAttentionLayer(nn.Module):
    """
    Joint Attention Layer for TwinVLA.
    
    在单个 Transformer 层中实现三分支联合 attention:
    - Common branch (共享的 ego view + language)
    - Left arm branch
    - Right arm branch
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int = 1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.scaling = head_dim ** -0.5
        
        # Q/K/V projections for each branch
        # Common branch
        self.common_q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.common_k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.common_v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.common_o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Left arm branch
        self.left_q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.left_k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.left_v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.left_o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
        # Right arm branch
        self.right_q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.right_k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.right_v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.right_o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        
    def forward(
        self,
        common_hidden: torch.Tensor,  # [B, common_len, D]
        left_hidden: torch.Tensor,    # [B, left_len, D]
        right_hidden: torch.Tensor,   # [B, right_len, D]
        attention_mask: torch.Tensor, # [B, 1, total_len, total_len]
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行联合 attention 计算。
        
        Returns:
            common_out, left_out, right_out: 各分支的输出
        """
        batch_size = common_hidden.shape[0]
        common_len = common_hidden.shape[1]
        left_len = left_hidden.shape[1]
        right_len = right_hidden.shape[1]
        
        # 计算 Q, K, V
        # Common
        common_q = self.common_q_proj(common_hidden).view(batch_size, common_len, self.num_heads, self.head_dim).transpose(1, 2)
        common_k = self.common_k_proj(common_hidden).view(batch_size, common_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        common_v = self.common_v_proj(common_hidden).view(batch_size, common_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Left
        left_q = self.left_q_proj(left_hidden).view(batch_size, left_len, self.num_heads, self.head_dim).transpose(1, 2)
        left_k = self.left_k_proj(left_hidden).view(batch_size, left_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        left_v = self.left_v_proj(left_hidden).view(batch_size, left_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Right
        right_q = self.right_q_proj(right_hidden).view(batch_size, right_len, self.num_heads, self.head_dim).transpose(1, 2)
        right_k = self.right_k_proj(right_hidden).view(batch_size, right_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        right_v = self.right_v_proj(right_hidden).view(batch_size, right_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 应用 RoPE (如果提供)
        if cos is not None and sin is not None:
            common_q, common_k = self._apply_rope(common_q, common_k, cos, sin, 0)
            left_q, left_k = self._apply_rope(left_q, left_k, cos, sin, common_len)
            right_q, right_k = self._apply_rope(right_q, right_k, cos, sin, common_len + left_len)
        
        # 拼接所有 Q, K, V
        all_q = torch.cat([common_q, left_q, right_q], dim=2)  # [B, H, total_len, head_dim]
        all_k = torch.cat([common_k, left_k, right_k], dim=2)
        all_v = torch.cat([common_v, left_v, right_v], dim=2)
        
        # 扩展 K, V 以匹配 Q 的头数 (GQA)
        if self.num_kv_heads != self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            all_k = all_k.repeat_interleave(n_rep, dim=1)
            all_v = all_v.repeat_interleave(n_rep, dim=1)
        
        # 计算 attention scores
        attn_weights = torch.matmul(all_q, all_k.transpose(-2, -1)) * self.scaling
        
        # 应用 attention mask
        attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(all_q.dtype)
        
        # 计算 attention output
        attn_output = torch.matmul(attn_weights, all_v)  # [B, H, total_len, head_dim]
        
        # 分割回各分支
        common_attn = attn_output[:, :, :common_len]
        left_attn = attn_output[:, :, common_len:common_len+left_len]
        right_attn = attn_output[:, :, common_len+left_len:]
        
        # 重塑并应用 output projection
        common_attn = common_attn.transpose(1, 2).reshape(batch_size, common_len, -1)
        left_attn = left_attn.transpose(1, 2).reshape(batch_size, left_len, -1)
        right_attn = right_attn.transpose(1, 2).reshape(batch_size, right_len, -1)
        
        common_out = self.common_o_proj(common_attn)
        left_out = self.left_o_proj(left_attn)
        right_out = self.right_o_proj(right_attn)
        
        return common_out, left_out, right_out
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 Rotary Position Embedding."""
        seq_len = q.shape[2]
        cos = cos[:, offset:offset+seq_len]
        sin = sin[:, offset:offset+seq_len]
        
        # 应用 RoPE
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


class JointTransformerBlock(nn.Module):
    """
    完整的 Joint Transformer Block，包含:
    - Joint Attention
    - LayerNorm
    - FFN (MLP)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        num_kv_heads: int = 1,
    ):
        super().__init__()
        
        # Layer norms for each branch
        self.common_input_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.left_input_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.right_input_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        
        self.common_post_attn_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.left_post_attn_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.right_post_attn_norm = nn.RMSNorm(hidden_size, eps=1e-6)
        
        # Joint Attention
        self.joint_attention = JointAttentionLayer(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )
        
        # FFN for each branch (可以共享或独立)
        self.common_mlp = self._create_mlp(hidden_size, intermediate_size)
        self.left_mlp = self._create_mlp(hidden_size, intermediate_size)
        self.right_mlp = self._create_mlp(hidden_size, intermediate_size)
        
    def _create_mlp(self, hidden_size: int, intermediate_size: int) -> nn.Module:
        """创建 GeGLU MLP."""
        return nn.Sequential(
            nn.Linear(hidden_size, intermediate_size * 2, bias=False),
            GeGLU(),
            nn.Linear(intermediate_size, hidden_size, bias=False),
        )
    
    def forward(
        self,
        common_hidden: torch.Tensor,
        left_hidden: torch.Tensor,
        right_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the joint transformer block."""
        
        # Pre-norm
        common_normed = self.common_input_norm(common_hidden)
        left_normed = self.left_input_norm(left_hidden)
        right_normed = self.right_input_norm(right_hidden)
        
        # Joint Attention
        common_attn, left_attn, right_attn = self.joint_attention(
            common_normed, left_normed, right_normed,
            attention_mask, position_ids, cos, sin
        )
        
        # Residual connection
        common_hidden = common_hidden + common_attn
        left_hidden = left_hidden + left_attn
        right_hidden = right_hidden + right_attn
        
        # Post-attention norm + FFN + Residual
        common_hidden = common_hidden + self.common_mlp(self.common_post_attn_norm(common_hidden))
        left_hidden = left_hidden + self.left_mlp(self.left_post_attn_norm(left_hidden))
        right_hidden = right_hidden + self.right_mlp(self.right_post_attn_norm(right_hidden))
        
        return common_hidden, left_hidden, right_hidden


class GeGLU(nn.Module):
    """GeGLU activation function."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

