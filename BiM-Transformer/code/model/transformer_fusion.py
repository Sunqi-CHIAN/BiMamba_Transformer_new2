"""
Cross-Modal Transformer Fusion for multimodal feature fusion.

This module implements sequence-level cross-modal fusion:
- Each modality provides a sequence of tokens (n_tokens, d_model)
- Concatenated into a joint sequence with modality embeddings
- Processed by Transformer encoder layers for real cross-modal attention
- Output via mean pooling to (batch, d_model)

This design fixes the critical issue in the previous version where
cross-attention on seq_len=1 tokens degenerated into linear transformations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class CrossModalTransformerFusion(nn.Module):
    """
    Sequence-level Cross-Modal Transformer Fusion.
    
    Concatenates token sequences from all modalities, adds learnable
    modality embeddings and positional embeddings, then applies
    Transformer self-attention for real cross-modal interaction.
    
    Args:
        d_model: Feature dimension for each modality
        nhead: Number of attention heads
        num_layers: Number of Transformer encoder layers
        dropout: Dropout probability
        n_tokens_per_modality: Number of tokens from each modality encoder
        num_modalities: Number of modalities (default: 3 for EEG, EOG, EMG)
        dim_feedforward: Hidden dimension of FFN (default: 4 * d_model)
    """
    def __init__(
        self,
        d_model: int = 96,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_tokens_per_modality: int = 16,
        num_modalities: int = 3,
        dim_feedforward: Optional[int] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_tokens_per_modality = n_tokens_per_modality
        self.num_modalities = num_modalities
        self.total_tokens = n_tokens_per_modality * num_modalities
        
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        
        # Learnable modality type embeddings (like token_type_embeddings in BERT)
        self.modality_embeddings = nn.Parameter(
            torch.randn(num_modalities, 1, d_model) * 0.02
        )
        
        # Learnable positional embeddings for the full concatenated sequence
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, self.total_tokens, d_model) * 0.02
        )
        
        # Pre-norm before transformer
        self.input_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder for joint self-attention across all modality tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
            norm_first=True  # Pre-norm style (better for training stability)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            enable_nested_tensor=False
        )
        
        # Modality-aware gating for final pooling
        # Learns to weight different modality regions differently
        self.pool_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        
    def forward(
        self,
        eeg_seq: torch.Tensor,
        eog_seq: torch.Tensor,
        emg_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for sequence-level cross-modal fusion.
        
        Args:
            eeg_seq: EEG token sequence (batch, n_tokens, d_model)
            eog_seq: EOG token sequence (batch, n_tokens, d_model)
            emg_seq: EMG token sequence (batch, n_tokens, d_model)
            
        Returns:
            Fused features (batch, d_model)
        """
        batch_size = eeg_seq.size(0)
        
        # Add modality type embeddings
        eeg_seq = eeg_seq + self.modality_embeddings[0]  # (B, T, D) + (1, D)
        eog_seq = eog_seq + self.modality_embeddings[1]
        emg_seq = emg_seq + self.modality_embeddings[2]
        
        # Concatenate all modality tokens: (B, 3*n_tokens, d_model)
        x = torch.cat([eeg_seq, eog_seq, emg_seq], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embeddings
        
        # Input normalization
        x = self.input_norm(x)
        
        # Transformer self-attention across all modality tokens
        # This is where real cross-modal attention happens
        x = self.transformer(x)  # (B, 3*n_tokens, d_model)
        
        # Gated mean pooling: learns to weight different positions
        gate = self.pool_gate(x)  # (B, 3*n_tokens, d_model)
        x = (x * gate).mean(dim=1)  # (B, d_model)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


# Legacy interface kept for backward compatibility
class CrossModalTransformer(nn.Module):
    """Legacy cross-modal transformer (vector inputs). Kept for compatibility."""
    
    def __init__(
        self, 
        d_model: int = 128, 
        nhead: int = 8, 
        num_layers: int = 2,
        dropout: float = 0.1,
        dim_feedforward: Optional[int] = None
    ):
        super(CrossModalTransformer, self).__init__()
        
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
            
        self.pos_embedding = nn.Parameter(torch.randn(1, 3, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(
        self, 
        eeg_feat: torch.Tensor, 
        eog_feat: torch.Tensor, 
        emg_feat: torch.Tensor
    ) -> torch.Tensor:
        x = torch.stack([eeg_feat, eog_feat, emg_feat], dim=1) 
        x = x + self.pos_embedding
        x = self.transformer(x)
        out = x.reshape(x.shape[0], -1)
        return out
