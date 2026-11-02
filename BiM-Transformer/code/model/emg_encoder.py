"""
EMG (Electromyogram) Encoder module with BiMamba layers.

Processes 2-channel EMG signals (zEMG and tEMG from DEAP dataset)
using 1D convolution with pooling and bidirectional Mamba layers.

Key improvement: Returns sequence output (B, n_tokens, d_model) for real cross-modal attention.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from code.mamba.bimamba import Mamba as BiMamba


class EMGEncoder(nn.Module):
    """
    EMG Encoder using 1D convolution with pooling and BiMamba layers.
    
    Returns (batch, n_tokens, d_model) sequence output.
    
    Args:
        d_model: Hidden dimension for BiMamba layers
        n_layers: Number of BiMamba layers
        dropout: Dropout probability
        in_channels: Number of input EMG channels
        conv_channels: Tuple of (first_conv_channels, second_conv_channels)
        kernel_sizes: Tuple of kernel sizes for the two conv layers
        pool_size: Size for max pooling
        n_tokens: Number of output tokens after temporal downsampling
    """
    def __init__(
        self, 
        d_model: int = 128, 
        n_layers: int = 4,
        dropout: float = 0.1,
        in_channels: int = 2,
        conv_channels: tuple = (32, 64),
        kernel_sizes: tuple = (25, 15),
        pool_size: int = 2,
        n_tokens: int = 16
    ):
        super(EMGEncoder, self).__init__()
        self.n_tokens = n_tokens
        self.in_channels = in_channels
        
        # EMG muscle activity features
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2),
            nn.BatchNorm1d(conv_channels[0]),
            nn.ELU(),
            nn.MaxPool1d(pool_size),
            nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2),
            nn.BatchNorm1d(conv_channels[1]),
            nn.ELU(),
            nn.MaxPool1d(pool_size)
        )
        
        self.input_proj = nn.Linear(conv_channels[1], d_model)
        
        # Pre-normalization for each BiMamba layer
        self.pre_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_layers)
        ])
        
        self.bimamba_layers = nn.ModuleList([
            BiMamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 2, time)
        Returns:
            (batch, n_tokens, d_model)
        """
        x = torch.clamp(x, min=-1e4, max=1e4)
        
        # Convolution with pooling: (batch, 2, time) -> (batch, 64, time/4)
        x = self.conv1d(x)
        
        # Reshape for BiMamba: (batch, time/4, 64)
        x = x.transpose(1, 2)
        x = self.input_proj(x)  # (batch, time/4, d_model)
        
        # BiMamba layers with pre-norm and residual connections
        for pre_norm, layer in zip(self.pre_norms, self.bimamba_layers):
            residual = x
            x = pre_norm(x)
            x = layer(x)
            x = self.dropout(x)
            x = x + residual
            
        x = self.norm(x)
        
        # Temporal downsampling: (batch, time/4, d_model) -> (batch, n_tokens, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, time/4)
        x = F.adaptive_avg_pool1d(x, self.n_tokens)  # (batch, d_model, n_tokens)
        x = x.transpose(1, 2)  # (batch, n_tokens, d_model)
        
        return x
