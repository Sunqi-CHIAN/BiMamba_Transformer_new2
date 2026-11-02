"""
BiMamba-Transformer Network for multimodal emotion recognition.

This is the main network that integrates EEG, EOG, and EMG encoders with
sequence-level cross-modal fusion and task-aware output layers.

Key improvements:
- Encoders output token sequences for real cross-modal attention
- Removed unused mental_health_head
- MLP classification heads with dropout for each VAD dimension
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Union
from .eeg_encoder import EEGEncoder
from .eog_encoder import EOGEncoder
from .emg_encoder import EMGEncoder
from .transformer_fusion import CrossModalTransformerFusion
from .moe_layer import TaskAwareLayer
from .grl_layer import GradientReversalLayer


class BiMambaTransformerNetwork(nn.Module):
    """
    BiMamba-Transformer Network for multimodal physiological signal analysis.
    
    This network processes EEG, EOG, and EMG signals through modality-specific
    encoders that output token sequences, fuses them with a joint cross-modal
    transformer, and outputs predictions for emotion recognition (VAD).
    
    Args:
        d_model: Hidden dimension for encoders and fusion
        n_classes_emotion: Number of emotion classes (for VAD dimensions)
        hidden_dim: Hidden dimension for task-aware layer
        dropout: Dropout probability for encoders
        eeg_layers: Number of BiMamba layers in EEG encoder
        peripheral_layers: Number of BiMamba layers in EOG/EMG encoders
        n_tokens: Number of temporal tokens per modality
    """
    def __init__(
        self, 
        d_model: int = 64, 
        n_classes_emotion: int = 2, 
        hidden_dim: int = 128,
        dropout: float = 0.25,
        eeg_layers: int = 2,
        peripheral_layers: int = 1,
        n_tokens: int = 16,
        fusion_layers: int = 1,
        num_experts: int = 2
    ):
        super(BiMambaTransformerNetwork, self).__init__()
        
        self.d_model = d_model
        self.n_tokens = n_tokens
        
        # Encoders — output (batch, n_tokens, d_model) token sequences
        self.eeg_encoder = EEGEncoder(
            d_model=d_model, n_layers=eeg_layers, 
            dropout=dropout, n_tokens=n_tokens
        )
        self.eog_encoder = EOGEncoder(
            d_model=d_model, n_layers=peripheral_layers, 
            dropout=dropout, n_tokens=n_tokens
        )
        self.emg_encoder = EMGEncoder(
            d_model=d_model, n_layers=peripheral_layers, 
            dropout=dropout, n_tokens=n_tokens
        )
        
        # Sequence-level Cross-Modal Fusion
        nhead = 4 if d_model <= 64 else 8
        dim_ff = d_model * 2 if fusion_layers == 1 else d_model * 4
        self.fusion = CrossModalTransformerFusion(
            d_model=d_model,
            nhead=nhead,
            num_layers=fusion_layers,
            dropout=dropout,
            n_tokens_per_modality=n_tokens,
            dim_feedforward=dim_ff
        )
        
        # Task Aware Layer (Gated Fusion + MoE)
        self.task_layer = TaskAwareLayer(
            input_dim=d_model, hidden_dim=hidden_dim,
            num_experts=num_experts, dropout=dropout * 0.5
        )
        
        # Task-specific projection layers
        # Each task gets its own LN + Linear + GELU + Dropout to specialize
        # features from the shared MoE representation
        self.valence_proj = self._make_task_proj(hidden_dim, dropout)
        self.arousal_proj = self._make_task_proj(hidden_dim, dropout)
        self.dominance_proj = self._make_task_proj(hidden_dim, dropout)
        
        # MLP Classification Heads (improved: 2-layer with dropout)
        self.valence_head = self._make_head(hidden_dim, n_classes_emotion, dropout)
        self.arousal_head = self._make_head(hidden_dim, n_classes_emotion, dropout)
        self.dominance_head = self._make_head(hidden_dim, n_classes_emotion, dropout)
        
        # DE (Differential Entropy) Feature Encoder (32 electrodes x 5 frequency bands = 160)
        # Heavily regularized + zero-initialized residual gate to prevent trial-fingerprint shortcut memorization
        self.de_encoder = nn.Sequential(
            nn.LayerNorm(160),
            nn.Dropout(0.35),
            nn.Linear(160, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.de_gate = nn.Parameter(torch.zeros(1))
        
        # Domain Adversarial Subject Discriminator (for 32 subjects)
        self.grl = GradientReversalLayer(alpha=1.0)
        self.domain_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 32)
        )
    
    @staticmethod
    def _make_task_proj(hidden_dim: int, dropout: float) -> nn.Module:
        """Create a task-specific projection layer with residual connection."""
        return nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),  # Very light dropout
        )
    
    @staticmethod
    def _make_head(hidden_dim: int, n_classes: int, dropout: float) -> nn.Module:
        """Create a 2-layer MLP classification head."""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # Lighter dropout in head
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(
        self, 
        eeg: torch.Tensor, 
        eog: torch.Tensor, 
        emg: torch.Tensor,
        de: Optional[torch.Tensor] = None,
        grl_alpha: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            eeg: EEG tensor of shape (batch, 1, electrodes, time)
            eog: EOG tensor of shape (batch, 2, time)
            emg: EMG tensor of shape (batch, 2, time)
            de: Optional Differential Entropy tensor of shape (batch, 32, 5) or (batch, 160)
            grl_alpha: Alpha parameter for Gradient Reversal Layer
            
        Returns:
            Dictionary containing:
                - valence: (batch, n_classes_emotion)
                - arousal: (batch, n_classes_emotion)
                - dominance: (batch, n_classes_emotion)
                - domain_pred: (batch, 32)
        """
        # Encode each modality -> (batch, n_tokens, d_model)
        eeg_seq = self.eeg_encoder(eeg)
        eog_seq = self.eog_encoder(eog)
        emg_seq = self.emg_encoder(emg)
        
        # Integrate DE (Differential Entropy) features via zero-initialized residual gate
        # Starts at 0 so BiMamba + SpatialConv learn full temporal-spatial representations without shortcut laziness
        if de is not None and hasattr(self, 'de_encoder'):
            de_flat = de.reshape(de.size(0), -1)  # (batch, 160)
            de_feat = self.de_encoder(de_flat)    # (batch, d_model)
            gate = torch.tanh(self.de_gate) if hasattr(self, 'de_gate') else 1.0
            eeg_seq = eeg_seq + gate * de_feat.unsqueeze(1)
        
        # Apply feature-level dropout during training
        if self.training:
            eeg_seq = F.dropout(eeg_seq, p=0.1, training=True)
            eog_seq = F.dropout(eog_seq, p=0.1, training=True)
            emg_seq = F.dropout(emg_seq, p=0.1, training=True)
        
        # Cross-modal fusion -> (batch, d_model)
        fused_feat = self.fusion(eeg_seq, eog_seq, emg_seq)
        
        # Task-aware processing -> (batch, hidden_dim)
        shared_feat = self.task_layer(fused_feat)
        
        # Task-specific projections with residual connections
        v_feat = shared_feat + self.valence_proj(shared_feat)
        a_feat = shared_feat + self.arousal_proj(shared_feat)
        d_feat = shared_feat + self.dominance_proj(shared_feat)
        
        # Output heads
        valence = self.valence_head(v_feat)
        arousal = self.arousal_head(a_feat)
        dominance = self.dominance_head(d_feat)
        
        # Domain adversarial head
        self.grl.alpha = grl_alpha
        domain_feat = self.grl(shared_feat)
        domain_pred = self.domain_head(domain_feat)
        
        return {
            "valence": valence,
            "arousal": arousal,
            "dominance": dominance,
            "domain_pred": domain_pred
        }
