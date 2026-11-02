"""
EEG Encoder module with BiMamba layers for temporal modeling.

This module processes multi-channel EEG signals using spatial convolution,
time-frequency decomposition, and bidirectional Mamba layers.

Key improvements:
- Returns sequence output (B, n_tokens, d_model) instead of pooled (B, d_model)
- Temporal downsampling via adaptive average pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from code.mamba.bimamba import Mamba as BiMamba


class SpatialConv(nn.Module):
    """
    Spatial convolution layer that compresses electrode dimension.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_electrodes: Number of EEG electrodes
    """
    def __init__(self, in_channels: int, out_channels: int, num_electrodes: int):
        super(SpatialConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(num_electrodes, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, electrodes, time)
        Returns:
            (batch, out_channels, 1, time)
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class TimeFrequencyDecomposition(nn.Module):
    """
    TF decomposition using multiple 1D convolutions with different kernel sizes
    to capture multi-scale temporal patterns (analogous to frequency bands).
    
    Args:
        in_channels: Number of input channels
        kernel_sizes: Tuple of kernel sizes for each frequency band
    """
    def __init__(self, in_channels: int, kernel_sizes: tuple = (64, 32, 16, 8, 4)):
        super(TimeFrequencyDecomposition, self).__init__()
        self.conv_delta = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        self.conv_theta = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_sizes[1], padding=kernel_sizes[1]//2)
        self.conv_alpha = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_sizes[2], padding=kernel_sizes[2]//2)
        self.conv_beta  = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_sizes[3], padding=kernel_sizes[3]//2)
        self.conv_gamma = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_sizes[4], padding=kernel_sizes[4]//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, in_channels, time)
        Returns:
            (batch, 5, channels, time)
        """
        seq_len = x.shape[-1]
        d = self.conv_delta(x)[:, :, :seq_len]
        t = self.conv_theta(x)[:, :, :seq_len]
        a = self.conv_alpha(x)[:, :, :seq_len]
        b = self.conv_beta(x)[:, :, :seq_len]
        g = self.conv_gamma(x)[:, :, :seq_len]
        
        x = torch.stack([d, t, a, b, g], dim=1)
        return x


class EEGEncoder(nn.Module):
    """
    EEG Encoder using spatial convolution, time-frequency decomposition, and BiMamba layers.
    
    Key change: Returns (batch, n_tokens, d_model) sequence output instead of pooled vector,
    enabling real cross-modal attention in the fusion layer.
    
    Args:
        d_model: Hidden dimension for BiMamba layers
        n_layers: Number of BiMamba layers
        num_electrodes: Number of EEG electrodes
        dropout: Dropout probability
        spatial_channels: Number of channels after spatial convolution
        n_tokens: Number of output tokens after temporal downsampling
    """
    def __init__(
        self, 
        d_model: int = 128, 
        n_layers: int = 6, 
        num_electrodes: int = 32,
        dropout: float = 0.1,
        spatial_channels: int = 16,
        n_tokens: int = 16
    ):
        super(EEGEncoder, self).__init__()
        self.n_tokens = n_tokens
        
        self.spatial_conv = SpatialConv(1, spatial_channels, num_electrodes)
        self.tf_decomp = TimeFrequencyDecomposition(spatial_channels)
        
        # Flatten TF bands into channels: spatial_channels * 5
        tf_dim = spatial_channels * 5
        self.input_proj = nn.Linear(tf_dim, d_model)
        
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
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, 1, electrodes, time)
            
        Returns:
            Output tensor of shape (batch, n_tokens, d_model)
        """
        # Input normalization
        x = torch.clamp(x, min=-1e4, max=1e4)
        
        # Spatial convolution: (batch, 1, electrodes, time) -> (batch, 16, 1, time)
        x = self.spatial_conv(x)
        x = x.squeeze(2)  # (batch, 16, time)
        
        # Time-frequency decomposition: (batch, 16, time) -> (batch, 5, 16, time)
        x = self.tf_decomp(x)
        
        # Reshape for BiMamba: (batch, time, 80)
        batch, bands, chans, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, bands * chans)
        x = self.input_proj(x)  # (batch, time, d_model)
        
        # BiMamba layers with pre-norm and residual connections
        for pre_norm, layer in zip(self.pre_norms, self.bimamba_layers):
            residual = x
            x = pre_norm(x)
            x = layer(x)
            x = self.dropout(x)
            x = x + residual
            
        x = self.norm(x)
        
        # Temporal downsampling: (batch, time, d_model) -> (batch, n_tokens, d_model)
        x = x.transpose(1, 2)  # (batch, d_model, time)
        x = F.adaptive_avg_pool1d(x, self.n_tokens)  # (batch, d_model, n_tokens)
        x = x.transpose(1, 2)  # (batch, n_tokens, d_model)
        
        return x
