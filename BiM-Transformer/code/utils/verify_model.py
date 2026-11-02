"""
Quick model verification script.
Tests that the model can perform a forward pass without errors.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from code.model import BiMambaTransformerNetwork


def verify():
    print("=" * 60)
    print("Model Verification")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model with default config parameters
    model = BiMambaTransformerNetwork(
        d_model=96,
        n_classes_emotion=2,
        hidden_dim=192,
        dropout=0.4,
        eeg_layers=3,
        peripheral_layers=2,
        n_tokens=16
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # Create dummy inputs (batch=2, matching DEAP dimensions)
    batch_size = 2
    eeg = torch.randn(batch_size, 1, 32, 512).to(device)   # (B, 1, 32 electrodes, 4s * 128Hz)
    eog = torch.randn(batch_size, 2, 512).to(device)        # (B, 2 channels, 4s * 128Hz)
    emg = torch.randn(batch_size, 2, 512).to(device)        # (B, 2 channels, 4s * 128Hz)
    
    print(f"\nInput shapes:")
    print(f"  EEG: {eeg.shape}")
    print(f"  EOG: {eog.shape}")
    print(f"  EMG: {emg.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(eeg, eog, emg)
    
    print(f"\nOutput shapes:")
    for key, val in outputs.items():
        print(f"  {key}: {val.shape}")
    
    # Verify no NaN
    all_finite = all(torch.isfinite(v).all().item() for v in outputs.values())
    print(f"\nAll outputs finite: {all_finite}")
    
    # Quick training mode test
    model.train()
    outputs = model(eeg, eog, emg)
    all_finite_train = all(torch.isfinite(v).all().item() for v in outputs.values())
    print(f"Training mode outputs finite: {all_finite_train}")
    
    # Test backward pass
    loss = sum(v.sum() for v in outputs.values())
    loss.backward()
    print(f"Backward pass: OK")
    
    # Check gradient flow
    has_grads = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"Parameters with gradients: {has_grads}/{total_params}")
    
    print("\n" + "=" * 60)
    if all_finite and all_finite_train:
        print("✓ Model verification PASSED")
    else:
        print("✗ Model verification FAILED")
    print("=" * 60)


if __name__ == "__main__":
    verify()
