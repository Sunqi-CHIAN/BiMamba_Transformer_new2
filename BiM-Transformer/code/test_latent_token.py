"""Test script for Latent Token fusion implementation."""
import torch
from model.bimamba_transformer_network import BiMambaTransformerNetwork

print("Testing Latent Token Fusion Implementation")
print("=" * 50)

# Create model with Latent Token fusion
model = BiMambaTransformerNetwork(
    d_model=96, 
    use_latent_token=True, 
    num_latent_tokens=8
)
print("Model created with Latent Token fusion")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Create dummy input
eeg = torch.randn(2, 1, 32, 512)
eog = torch.randn(2, 2, 512)
emg = torch.randn(2, 2, 512)

# Forward pass
with torch.no_grad():
    out = model(eeg, eog, emg)
    print(f"Valence shape: {out['valence'].shape}")
    print(f"Arousal shape: {out['arousal'].shape}")
    print(f"Mental health shape: {out['mental_health'].shape}")

print("=" * 50)
print("Forward pass successful!")
