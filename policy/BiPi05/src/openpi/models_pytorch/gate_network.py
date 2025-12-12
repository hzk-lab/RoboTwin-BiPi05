"""Gate Network for PECA (Predictive Error for Cooperative Action) training.

This module defines the GateNetwork class that predicts the α_t collaboration switch
(0=cooperate with cross-attention, 1=independent) based on left/right arm hidden states.
"""

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class GateNetwork(nn.Module):
    """Predicts α_t collaboration switch based on left/right arm hidden states.
    
    α_t is a binary gate that controls cross-arm attention:
    - α_t = 0: Full cross-attention (collaboration mode)
    - α_t = 1: Independent attention (no cross-arm communication)
    
    During training, α_t is a soft sigmoid output trained with BCE loss.
    During inference, α_t is thresholded at 0.5 to become a hard gate.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden: int = 256,
        dropout: float = 0.0,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp_hidden = mlp_hidden
        
        # MLP takes concatenated left and right hidden states
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, 1),
        )
        
        self._init_weights()
        self.to_precision(precision)
    
    def _init_weights(self):
        """Initialize weights with small values to start near 0.5."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def to_precision(self, precision: Literal["bfloat16", "float32"]):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
    
    def forward(
        self,
        left_hidden: torch.Tensor,
        right_hidden: torch.Tensor,
        return_logit: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Predict α_t from left and right arm hidden states.
        
        Args:
            left_hidden: [B, seq_len, hidden_dim]
            right_hidden: [B, seq_len, hidden_dim]
            return_logit: If True, also return raw logit
            
        Returns:
            alpha_t: [B, 1], range (0, 1)
        """
        h_L = left_hidden[:, -1, :]  # [B, hidden_dim]
        h_R = right_hidden[:, -1, :]  # [B, hidden_dim]
        
        h_concat = torch.cat([h_L, h_R], dim=-1)
        logit = self.mlp(h_concat)
        alpha_t = torch.sigmoid(logit)
        
        if return_logit:
            return alpha_t, logit
        return alpha_t
    
    def get_hard_gate(self, alpha_t: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        return (alpha_t > threshold).float()


def compute_stickiness_loss(alpha_sequence: torch.Tensor) -> torch.Tensor:
    """Compute stickiness loss for temporal continuity."""
    if alpha_sequence.dim() == 2 and alpha_sequence.shape[1] > 1:
        diff = alpha_sequence[:, 1:] - alpha_sequence[:, :-1]
        return (diff ** 2).mean()
    return torch.tensor(0.0, device=alpha_sequence.device, dtype=alpha_sequence.dtype)


def compute_gate_bce_loss(
    alpha_pred: torch.Tensor,
    loss_on: torch.Tensor,
    loss_off: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute BCE loss for gate prediction."""
    alpha_label = (loss_off < loss_on).float()
    
    if alpha_label.dim() == 0:
        alpha_label = alpha_label.unsqueeze(0)
    if alpha_pred.dim() == 2:
        alpha_label = alpha_label.unsqueeze(-1)
    
    bce_loss = F.binary_cross_entropy(alpha_pred, alpha_label, reduction='mean')
    return bce_loss, alpha_label


def compute_peca_loss(
    alpha_t: torch.Tensor,
    loss_on: torch.Tensor,
    loss_off: torch.Tensor,
    peca_lambda: float = 0.1,
) -> torch.Tensor:
    """Compute PECA loss: L_PECA = λ * (L_on - L_off)_sg * α_t"""
    loss_diff = (loss_on - loss_off).detach()
    
    if loss_diff.dim() == 0:
        loss_diff = loss_diff.unsqueeze(0)
    if alpha_t.dim() == 2 and loss_diff.dim() == 1:
        loss_diff = loss_diff.unsqueeze(-1)
    
    return peca_lambda * (loss_diff * alpha_t).mean()

