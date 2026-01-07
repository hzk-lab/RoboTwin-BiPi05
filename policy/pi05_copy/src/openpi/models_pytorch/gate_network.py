"""Gate Network module for controlling cross-arm attention.

This module implements a gate network that predicts the cooperation level (alpha)
between left and right arms based on their action expert hidden states.
"""

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class GateNetwork(nn.Module):
    """Predicts the alpha_t cooperation switch based on concatenated hidden states.
    
    alpha_t = 1 means independent (no cross-attention)
    alpha_t = 0 means full cooperation (full cross-attention)
    """
    
    def __init__(
        self,
        hidden_dim: int,
        mlp_hidden: int = 256,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        """Initialize the Gate Network.
        
        Args:
            hidden_dim: Dimension of the hidden states from each AE.
            mlp_hidden: Hidden dimension of the MLP.
            precision: Model precision ("bfloat16" or "float32").
        """
        super().__init__()
        
        # MLP: concat(h_L, h_R) -> alpha
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Linear(mlp_hidden // 2, 1),
        )
        
        # Apply precision
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
    
    def forward(
        self,
        left_hidden: torch.Tensor,
        right_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Compute alpha_t from AE hidden states.
        
        Args:
            left_hidden: Left AE hidden states [batch, seq_len, hidden_dim]
            right_hidden: Right AE hidden states [batch, seq_len, hidden_dim]
        
        Returns:
            alpha_t: Cooperation level [batch, 1], 0=full cooperation, 1=independent
        """
        # Use the last token's hidden state as input to the gate network
        h_L = left_hidden[:, -1, :]   # [batch, hidden_dim]
        h_R = right_hidden[:, -1, :]  # [batch, hidden_dim]
        
        # Concatenate and pass through MLP
        h_concat = torch.cat([h_L, h_R], dim=-1)  # [batch, hidden_dim * 2]
        logit = self.mlp(h_concat)  # [batch, 1]
        
        # Sigmoid to get alpha in [0, 1]
        alpha_t = torch.sigmoid(logit)
        
        return alpha_t
    
    def get_hard_gate(self, alpha_t: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert soft alpha to hard binary gate.
        
        Args:
            alpha_t: Soft alpha values [batch, 1]
            threshold: Threshold for binarization
        
        Returns:
            Hard binary gate [batch, 1], 0 or 1
        """
        return (alpha_t > threshold).float()


def compute_gate_bce_loss(
    alpha_pred: torch.Tensor,
    alpha_label: torch.Tensor,
) -> torch.Tensor:
    """Compute Binary Cross-Entropy loss for the gate network.
    
    Args:
        alpha_pred: Predicted alpha [batch, 1]
        alpha_label: Target alpha labels [batch, 1], 0 or 1
    
    Returns:
        BCE loss scalar
    """
    return F.binary_cross_entropy(alpha_pred, alpha_label)


def compute_l1_regularization(
    alpha_pred: torch.Tensor,
    l1_lambda: float = 0.01,
) -> torch.Tensor:
    """Compute L1 regularization for alpha_t to encourage sparsity.
    
    Args:
        alpha_pred: Predicted alpha [batch, 1]
        l1_lambda: L1 regularization coefficient
    
    Returns:
        L1 regularization loss scalar
    """
    return l1_lambda * alpha_pred.mean()


def compute_stickiness_loss(
    alpha_sequence: torch.Tensor,
    sticky_lambda: float = 0.01,
) -> torch.Tensor:
    """Compute stickiness loss to encourage temporal continuity of alpha.
    
    Args:
        alpha_sequence: Sequence of alpha values [batch, time]
        sticky_lambda: Stickiness regularization coefficient
    
    Returns:
        Stickiness loss scalar
    """
    if alpha_sequence.shape[1] < 2:
        return torch.tensor(0.0, device=alpha_sequence.device, dtype=alpha_sequence.dtype)
    
    # Penalize changes in alpha over time
    diff = alpha_sequence[:, 1:] - alpha_sequence[:, :-1]
    return sticky_lambda * (diff ** 2).mean()


def compute_peca_loss(
    loss_on: torch.Tensor,
    loss_off: torch.Tensor,
    alpha_pred: torch.Tensor,
    peca_lambda: float = 0.1,
) -> torch.Tensor:
    """Compute PECA (Predictive Error for Cooperative Action) loss.
    
    This loss teaches the gate network when to enable/disable cross-attention
    based on the difference in BC loss with/without cross-attention.
    
    If loss_on < loss_off: cooperation helps -> alpha should be low (0)
    If loss_on > loss_off: cooperation hurts -> alpha should be high (1)
    
    Args:
        loss_on: BC loss with cross-attention enabled
        loss_off: BC loss with cross-attention disabled
        alpha_pred: Predicted alpha [batch, 1]
        peca_lambda: PECA loss coefficient
    
    Returns:
        PECA loss scalar
    """
    # Stop gradient on the loss difference - only train alpha_pred
    loss_diff = (loss_on - loss_off).detach()
    
    # When loss_diff > 0 (cooperation hurts), alpha should be high (1)
    # When loss_diff < 0 (cooperation helps), alpha should be low (0)
    # This formulation encourages alpha to match the sign of the loss difference
    return peca_lambda * (loss_diff * alpha_pred).mean()


def compute_gate_label(
    loss_on: torch.Tensor,
    loss_off: torch.Tensor,
) -> torch.Tensor:
    """Compute binary gate labels based on BC loss comparison.
    
    Args:
        loss_on: BC loss with cross-attention enabled [batch]
        loss_off: BC loss with cross-attention disabled [batch]
    
    Returns:
        Binary labels [batch, 1]: 0 if cooperation helps, 1 if it hurts
    """
    # If loss_on < loss_off, cooperation helps -> label = 0 (use cooperation)
    # If loss_on >= loss_off, cooperation hurts -> label = 1 (independent)
    return (loss_on >= loss_off).float().unsqueeze(-1)

