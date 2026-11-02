"""
Mixture of Experts (MoE) and Gated Fusion layers for task-aware processing.

This module provides gated fusion and MoE components for flexible multi-task learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class GatedFusion(nn.Module):
    """
    Gated fusion layer that learns to weight features.
    
    Args:
        input_dim: Input feature dimension
    """
    def __init__(self, input_dim: int):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Gated output of shape (batch, input_dim)
        """
        g = self.gate(x)
        return x * g


class Expert(nn.Module):
    """
    Single expert network in MoE.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        dropout: Dropout probability
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class MoELayer(nn.Module):
    """
    Mixture of Experts layer with soft routing.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        num_experts: Number of expert networks
        dropout: Dropout probability for experts
    """
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        num_experts: int = 4,
        dropout: float = 0.1
    ):
        super(MoELayer, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, output_dim, dropout) for _ in range(num_experts)
        ])
        self.router = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        # Compute routing weights: (batch, num_experts)
        weights = self.router(x)
        
        # Get expert outputs: (batch, num_experts, output_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        
        # Weighted sum of expert outputs
        out = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return out


class TaskAwareLayer(nn.Module):
    """
    Task-aware layer combining gated fusion and MoE for multi-task learning.
    
    Args:
        input_dim: Input feature dimension (typically 3*d_model for 3 modalities)
        hidden_dim: Hidden/output dimension
        num_experts: Number of experts in MoE
        dropout: Dropout probability
    """
    def __init__(
        self, 
        input_dim: int = 3*128, 
        hidden_dim: int = 256,
        num_experts: int = 4,
        dropout: float = 0.1
    ):
        super(TaskAwareLayer, self).__init__()
        self.gated_fusion = GatedFusion(input_dim)
        self.moe = MoELayer(input_dim, hidden_dim, num_experts=num_experts, dropout=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Output tensor of shape (batch, hidden_dim)
        """
        x = self.gated_fusion(x)
        x = self.moe(x)
        return x
