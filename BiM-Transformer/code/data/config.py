"""
Configuration file for DEAP dataset preprocessing and training.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DEAPConfig:
    """
    Configuration for DEAP dataset preprocessing and training.
    
    Preprocessing parameters:
        data_dir: Path to raw DEAP data directory
        preprocessed_path: Path to save/load preprocessed data
        baseline_duration: Duration of baseline period in seconds
        segment_length: Length of each segment in seconds
        segment_overlap: Overlap between segments in seconds
        binary_labels: Whether to convert labels to binary (low/high)
        label_threshold_low: Labels below this are "low" class
        label_threshold_high: Labels at or above this are "high" class
        
    Training parameters:
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        weight_decay: Weight decay for optimizer
        
    Evaluation parameters:
        subject_independent: Whether to use subject-independent evaluation
        test_subjects: List of subject IDs for testing
        test_size: Fraction of data for testing (subject-dependent mode)
        val_size: Fraction of training data for validation
    """
    # Data paths
    data_dir: str = "data/DEAP/data_preprocessed_python"
    preprocessed_path: str = "data/DEAP/deap_preprocessed.npz"
    
    # Preprocessing parameters
    baseline_duration: float = 3.0
    segment_length: float = 4.0
    segment_overlap: float = 2.0  # 50% overlap for 2x more training samples
    binary_labels: bool = True
    norm_level: str = "trial"     # "trial" for raw waveforms (eliminates 60-min session impedance drift); DE always uses subject-norm
    extract_de: bool = True       # Extract baseline-corrected Differential Entropy features
    use_de: bool = True           # Whether to feed DE features into the model during training/eval
    # Label thresholds: exclude ambiguous labels in [threshold_low, threshold_high)
    label_threshold_low: float = 4.0   # < this -> class 0
    label_threshold_high: float = 6.0  # >= this -> class 1
    
    # Modality-specific filter parameters (EOG/EMG only; EEG already filtered in DEAP)
    eog_lowcut: float = 1.0
    eog_highcut: float = 45.0
    emg_lowcut: float = 10.0
    emg_highcut: float = 45.0
    
    # Model parameters (right-sized for cross-subject generalization)
    d_model: int = 64
    n_classes_emotion: int = 2
    hidden_dim: int = 128
    dropout: float = 0.25
    eeg_layers: int = 2
    peripheral_layers: int = 1
    fusion_layers: int = 1
    num_experts: int = 2
    n_tokens: int = 16  # Number of temporal tokens per modality after downsampling
    
    # Training parameters (anti-overfitting cross-subject regime)
    batch_size: int = 64
    num_epochs: int = 40
    learning_rate: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 5
    label_smoothing: float = 0.10
    
    # Regularization
    use_dann: bool = False           # Disabled by default: avoids target-shift conflict on imbalanced subject priors
    dann_weight: float = 0.10        # Max GRL reversal lambda when use_dann=True
    use_mixup: bool = True           # Prevents memorizing 50%-overlapping segments within trials
    mixup_alpha: float = 0.2
    
    # Loss configuration (clean dynamically-balanced CE by default)
    use_focal_loss: bool = False
    focal_gamma: float = 1.5
    
    # SWA (Stochastic Weight Averaging): validation metrics plateau from ~epoch 5 with +/-0.02 noise,
    # so averaging epochs 20-40 is more robust than picking a single noisy argmax checkpoint
    use_swa: bool = True
    swa_start_epoch: int = 20
    swa_lr: float = 2e-5
    
    # Evaluation parameters
    subject_independent: bool = True
    test_subjects: List[int] = field(default_factory=lambda: [1, 2, 3])
    val_subjects: Optional[List[int]] = None
    test_size: float = 0.2
    val_size: float = 0.20
    val_metric: str = "val_balanced_acc"  # Options: "val_balanced_acc", "val_auc", "val_macro_f1", "val_loss"
    
    # Other
    random_state: int = 42
    num_workers: int = 2
    device: str = "cuda"
    
    def __post_init__(self):
        """Validate configuration."""
        assert 0 < self.segment_length <= 60, "Segment length must be between 0 and 60 seconds"
        assert 0 <= self.segment_overlap < self.segment_length, "Overlap must be less than segment length"
        assert 0 < self.test_size < 1, "Test size must be between 0 and 1"
        assert 0 < self.val_size < 1, "Val size must be between 0 and 1"
        assert self.label_threshold_low < self.label_threshold_high, \
            "label_threshold_low must be < label_threshold_high"


# Default configuration
DEFAULT_CONFIG = DEAPConfig()


# Configuration for subject-independent evaluation (LOSO)
LOSO_CONFIG = DEAPConfig(
    subject_independent=True,
    test_subjects=[1, 2, 3, 4],
    batch_size=64,
    num_epochs=150
)


# Configuration for subject-dependent evaluation
SUBJECT_DEPENDENT_CONFIG = DEAPConfig(
    subject_independent=False,
    test_size=0.2,
    val_size=0.15,
    batch_size=32,
    num_epochs=100
)


# Quick debug configuration (smaller data for testing)
DEBUG_CONFIG = DEAPConfig(
    segment_length=2.0,
    segment_overlap=0.0,
    batch_size=16,
    num_epochs=5,
    num_workers=0
)
