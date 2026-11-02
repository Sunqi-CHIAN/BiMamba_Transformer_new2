"""
Training script for BiMamba-Transformer on DEAP dataset.

Optimized training pipeline including:
- Focal Loss for handling class imbalance
- Warmup + CosineAnnealing LR schedule
- Stochastic Weight Averaging (SWA)
- F1-score and AUC evaluation metrics
- Gradient accumulation support
- Early stopping with patience

Usage:
    # Step 1: Preprocess DEAP data
    python code/data/deap_dataset.py
    
    # Step 2: Train model
    python train.py --train --data_path data/DEAP/deap_preprocessed.npz
    
    # Step 3: Evaluate model
    python train.py --evaluate --checkpoint checkpoints/best_model.pth --data_path data/DEAP/deap_preprocessed.npz
"""
import os
import sys
import argparse
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Guard against stdlib module name collision: project uses package name "code".
existing_code_module = sys.modules.get('code')
if existing_code_module is not None and not hasattr(existing_code_module, '__path__'):
    del sys.modules['code']

from code.model import BiMambaTransformerNetwork
from code.data import create_deap_dataloaders
from code.data.config import DEAPConfig, DEFAULT_CONFIG
from code.utils.metrics import (
    compute_classification_metrics,
    find_optimal_threshold,
    aggregate_trial_predictions,
    format_metrics_table
)


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance (mathematically exact formulation).
    
    Computes pt from unweighted cross-entropy so class weights alpha_t scale
    the loss linearly rather than exponentiating inside pt = exp(-ce).
    """
    def __init__(
        self, 
        alpha: torch.Tensor = None, 
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Unweighted true-class log-prob for exact pt
        ce_raw = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_raw)
        
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha_t * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# Early Stopping
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
            
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop


# ============================================================================
# Metrics
# ============================================================================

class MetricsTracker:
    """Track training and validation metrics."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_valence_acc': [],
            'val_arousal_acc': [],
            'val_dominance_acc': [],
            'val_valence_bal_acc': [],
            'val_arousal_bal_acc': [],
            'val_dominance_bal_acc': [],
            'val_valence_f1': [],
            'val_arousal_f1': [],
            'val_dominance_f1': [],
            'val_mean_acc': [],
            'val_mean_bal_acc': [],
            'val_mean_f1': [],
            'val_mean_auc': [],
            'val_trial_mean_bal_acc': [],
            'val_trial_mean_auc': [],
            'learning_rate': []
        }
        
    def update(self, epoch_metrics: dict):
        for key, value in epoch_metrics.items():
            if key in self.history:
                self.history[key].append(value)
                
    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def load(self, path: str):
        with open(path, 'r') as f:
            self.history = json.load(f)


def compute_f1(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute macro F1-score for binary classification."""
    preds = preds.long()
    targets = targets.long()
    
    # Per-class F1
    f1_scores = []
    for cls in range(2):
        tp = ((preds == cls) & (targets == cls)).sum().float()
        fp = ((preds == cls) & (targets != cls)).sum().float()
        fn = ((preds != cls) & (targets == cls)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1.item())
    
    return np.mean(f1_scores)


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(
    model: nn.Module,
    train_loader,
    criterion_valence: nn.Module,
    criterion_arousal: nn.Module,
    criterion_dominance: nn.Module,
    criterion_domain: nn.Module,
    task_weights: dict,
    total_weight: float,
    optimizer: optim.Optimizer,
    device: torch.device,
    config,
    epoch: int,
    max_epochs: int,
    max_grad_norm: float = 1.0
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    nan_count = 0
    
    # Calculate GRL reversal lambda for DANN (anneals from 0 to config.dann_weight)
    p = float(epoch) / max(max_epochs, 1)
    grl_alpha = config.dann_weight * (2. / (1. + np.exp(-10 * p)) - 1)
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch in pbar:
        eeg = batch['eeg'].to(device)
        eog = batch['eog'].to(device)
        emg = batch['emg'].to(device)
        de = batch.get('de', None)
        if de is not None:
            de = de.to(device)
        labels = batch['labels']
        valid_mask = batch.get('valid_mask', None)
        participant_id = batch.get('participant_id', None)
        
        if participant_id is not None:
            participant_id = participant_id.to(device)
        
        if isinstance(labels, dict):
            valence = labels['valence'].to(device)
            arousal = labels['arousal'].to(device)
            dominance = labels['dominance'].to(device)
        else:
            valence = labels.to(device)
            arousal = valence
            dominance = valence
            
        if isinstance(valid_mask, dict):
            mask_v = valid_mask['valence'].to(device).bool()
            mask_a = valid_mask['arousal'].to(device).bool()
            mask_d = valid_mask['dominance'].to(device).bool()
        elif valid_mask is not None:
            mask_v = valid_mask.to(device).bool()
            mask_a = mask_v
            mask_d = mask_v
        else:
            mask_v = (valence != -1)
            mask_a = (arousal != -1)
            mask_d = (dominance != -1)
            
        # Mixup augmentation
        use_mixup = config.use_mixup and config.mixup_alpha > 0
        if use_mixup:
            lam = np.random.beta(config.mixup_alpha, config.mixup_alpha)
            batch_size = eeg.size(0)
            index = torch.randperm(batch_size).to(device)
            
            # Mix inputs
            eeg = lam * eeg + (1 - lam) * eeg[index, :]
            eog = lam * eog + (1 - lam) * eog[index, :]
            emg = lam * emg + (1 - lam) * emg[index, :]
            if de is not None:
                de = lam * de + (1 - lam) * de[index, :]
            
            # Target mixing index
            val_a, val_b = valence, valence[index]
            aro_a, aro_b = arousal, arousal[index]
            dom_a, dom_b = dominance, dominance[index]
            
            mask_v_a, mask_v_b = mask_v, mask_v[index]
            mask_a_a, mask_a_b = mask_a, mask_a[index]
            mask_d_a, mask_d_b = mask_d, mask_d[index]
            
            if participant_id is not None:
                pid_a, pid_b = participant_id, participant_id[index]
        
        optimizer.zero_grad()
        
        # Forward pass (if DANN is enabled, pass grl_alpha)
        outputs = model(eeg, eog, emg, de=de, grl_alpha=grl_alpha if config.use_dann else 0.0)
        
        # Calculate Task Loss with valid sample masking (reduction='none' criteria)
        if use_mixup:
            loss_v_a = criterion_valence(outputs['valence'], val_a) * mask_v_a.float()
            loss_v_b = criterion_valence(outputs['valence'], val_b) * mask_v_b.float()
            weight_v = lam * mask_v_a.float() + (1 - lam) * mask_v_b.float()
            loss_v = (lam * loss_v_a + (1 - lam) * loss_v_b).sum() / (weight_v.sum() + 1e-8)

            loss_a_a = criterion_arousal(outputs['arousal'], aro_a) * mask_a_a.float()
            loss_a_b = criterion_arousal(outputs['arousal'], aro_b) * mask_a_b.float()
            weight_a = lam * mask_a_a.float() + (1 - lam) * mask_a_b.float()
            loss_a = (lam * loss_a_a + (1 - lam) * loss_a_b).sum() / (weight_a.sum() + 1e-8)

            loss_d_a = criterion_dominance(outputs['dominance'], dom_a) * mask_d_a.float()
            loss_d_b = criterion_dominance(outputs['dominance'], dom_b) * mask_d_b.float()
            weight_d = lam * mask_d_a.float() + (1 - lam) * mask_d_b.float()
            loss_d = (lam * loss_d_a + (1 - lam) * loss_d_b).sum() / (weight_d.sum() + 1e-8)
        else:
            loss_v_raw = criterion_valence(outputs['valence'], valence) * mask_v.float()
            loss_v = loss_v_raw.sum() / (mask_v.float().sum() + 1e-8)

            loss_a_raw = criterion_arousal(outputs['arousal'], arousal) * mask_a.float()
            loss_a = loss_a_raw.sum() / (mask_a.float().sum() + 1e-8)

            loss_d_raw = criterion_dominance(outputs['dominance'], dominance) * mask_d.float()
            loss_d = loss_d_raw.sum() / (mask_d.float().sum() + 1e-8)
            
        task_loss = (task_weights['valence'] * loss_v + 
                     task_weights['arousal'] * loss_a + 
                     task_weights['dominance'] * loss_d) / total_weight
                     
        # Domain Adversarial Loss (discriminator trained at weight 1.0; reversal controlled via grl_alpha)
        if config.use_dann and 'domain_pred' in outputs and participant_id is not None and criterion_domain is not None:
            if use_mixup:
                domain_loss = lam * criterion_domain(outputs['domain_pred'], pid_a) + (1 - lam) * criterion_domain(outputs['domain_pred'], pid_b)
            else:
                domain_loss = criterion_domain(outputs['domain_pred'], participant_id)
            loss = task_loss + domain_loss
        else:
            loss = task_loss
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            optimizer.zero_grad()
            continue
        
        loss.backward()
        
        # Check for NaN gradients
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                has_nan_grad = True
                break
        
        if has_nan_grad:
            nan_count += 1
            optimizer.zero_grad()
            continue
        
        if config.use_dann and hasattr(model, 'domain_head'):
            domain_params = list(model.domain_head.parameters())
            domain_ids = {id(p) for p in domain_params}
            backbone_params = [p for p in model.parameters() if id(p) not in domain_ids]
            torch.nn.utils.clip_grad_norm_(backbone_params, max_norm=max_grad_norm)
            torch.nn.utils.clip_grad_norm_(domain_params, max_norm=max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        
        total_loss += task_loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{task_loss.item():.4f}'})
    
    if nan_count > 0:
        print(f"  Warning: {nan_count} batches skipped due to NaN/Inf")
    
    avg_loss = total_loss / n_batches if n_batches > 0 else float('nan')
    return {'train_loss': avg_loss}


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader,
    criterion_valence: nn.Module,
    criterion_arousal: nn.Module,
    criterion_dominance: nn.Module,
    task_weights: dict,
    total_weight: float,
    device: torch.device
) -> dict:
    """Validate the model with both segment-level and trial-level balanced metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    
    all_preds = {'valence': [], 'arousal': [], 'dominance': []}
    all_probs = {'valence': [], 'arousal': [], 'dominance': []}
    all_targets = {'valence': [], 'arousal': [], 'dominance': []}
    all_sub_ids = {'valence': [], 'arousal': [], 'dominance': []}
    all_tri_ids = {'valence': [], 'arousal': [], 'dominance': []}
    
    for batch in tqdm(val_loader, desc='Validating', leave=False):
        eeg = batch['eeg'].to(device)
        eog = batch['eog'].to(device)
        emg = batch['emg'].to(device)
        de = batch.get('de', None)
        if de is not None:
            de = de.to(device)
        labels = batch['labels']
        valid_mask = batch.get('valid_mask', None)
        
        sub_ids = None
        if 'subject_id' in batch:
            sub_ids = batch['subject_id'].cpu().numpy()
        elif 'participant_id' in batch:
            sub_ids = batch['participant_id'].cpu().numpy() + 1
        tri_ids = batch['trial_id'].cpu().numpy() if 'trial_id' in batch else None
        
        if isinstance(labels, dict):
            valence = labels['valence'].to(device)
            arousal = labels['arousal'].to(device)
            dominance = labels['dominance'].to(device)
        else:
            valence = labels.to(device)
            arousal = valence
            dominance = valence
            
        if isinstance(valid_mask, dict):
            mask_v = valid_mask['valence'].to(device).bool()
            mask_a = valid_mask['arousal'].to(device).bool()
            mask_d = valid_mask['dominance'].to(device).bool()
        elif valid_mask is not None:
            mask_v = valid_mask.to(device).bool()
            mask_a = mask_v
            mask_d = mask_v
        else:
            mask_v = (valence != -1)
            mask_a = (arousal != -1)
            mask_d = (dominance != -1)
        
        outputs = model(eeg, eog, emg, de=de)
        
        loss_v = (criterion_valence(outputs['valence'], valence) * mask_v.float()).sum() / (mask_v.float().sum() + 1e-8)
        loss_a = (criterion_arousal(outputs['arousal'], arousal) * mask_a.float()).sum() / (mask_a.float().sum() + 1e-8)
        loss_d = (criterion_dominance(outputs['dominance'], dominance) * mask_d.float()).sum() / (mask_d.float().sum() + 1e-8)

        loss = (
            task_weights['valence'] * loss_v +
            task_weights['arousal'] * loss_a +
            task_weights['dominance'] * loss_d
        ) / total_weight
        
        total_loss += loss.item()
        n_batches += 1
        
        task_masks = {'valence': mask_v, 'arousal': mask_a, 'dominance': mask_d}
        task_targets = {'valence': valence, 'arousal': arousal, 'dominance': dominance}
        for task in ['valence', 'arousal', 'dominance']:
            m = task_masks[task]
            if m.any():
                m_np = m.cpu().numpy().astype(bool)
                probs = F.softmax(outputs[task], dim=1)[m]
                preds = outputs[task].argmax(dim=1)[m]
                all_preds[task].append(preds.cpu())
                all_probs[task].append(probs.cpu())
                all_targets[task].append(task_targets[task][m].cpu())
                if sub_ids is not None:
                    all_sub_ids[task].append(sub_ids[m_np])
                if tri_ids is not None:
                    all_tri_ids[task].append(tri_ids[m_np])
    
    # Compute segment-level and trial-level validation metrics
    results = {'val_loss': total_loss / n_batches if n_batches > 0 else float('nan')}
    
    accs, bal_accs, f1s, aucs = [], [], [], []
    trial_bal_accs, trial_aucs = [], []
    optimal_thresholds = {}
    optimal_thresholds_trial = {}
    has_val_trials = False

    for task in ['valence', 'arousal', 'dominance']:
        if len(all_preds[task]) > 0:
            preds = torch.cat(all_preds[task]).numpy()
            probs = torch.cat(all_probs[task]).numpy()
            targets = torch.cat(all_targets[task]).numpy()
            task_m = compute_classification_metrics(targets, preds, probs)
            optimal_thresholds[task] = find_optimal_threshold(targets, probs)

            if len(all_sub_ids[task]) > 0 and len(all_tri_ids[task]) > 0:
                subjs = np.concatenate(all_sub_ids[task])
                tris = np.concatenate(all_tri_ids[task])
                t_m, t_data = aggregate_trial_predictions(subjs, tris, probs, targets, threshold=0.5)
                trial_lbls = t_data.get('labels', t_data.get('trial_labels'))
                trial_prbs = t_data.get('probs', t_data.get('trial_probs'))
                optimal_thresholds_trial[task] = find_optimal_threshold(trial_lbls, trial_prbs)
                results[f'val_trial_{task}_bal_acc'] = t_m['balanced_accuracy']
                results[f'val_trial_{task}_auc'] = t_m['auc_roc']
                trial_bal_accs.append(t_m['balanced_accuracy'])
                trial_aucs.append(t_m['auc_roc'])
                has_val_trials = True
            else:
                optimal_thresholds_trial[task] = optimal_thresholds[task]
        else:
            task_m = {'accuracy': 0.0, 'balanced_accuracy': 0.0, 'macro_f1': 0.0, 'auc_roc': 0.5}
            optimal_thresholds[task] = 0.50
            optimal_thresholds_trial[task] = 0.50
        
        results[f'val_{task}_acc'] = task_m['accuracy']
        results[f'val_{task}_bal_acc'] = task_m['balanced_accuracy']
        results[f'val_{task}_f1'] = task_m['macro_f1']
        results[f'val_{task}_auc'] = task_m['auc_roc']
        
        accs.append(task_m['accuracy'])
        bal_accs.append(task_m['balanced_accuracy'])
        f1s.append(task_m['macro_f1'])
        aucs.append(task_m['auc_roc'])
    
    results['val_mean_acc'] = float(np.mean(accs))
    results['val_mean_bal_acc'] = float(np.mean(bal_accs))
    results['val_mean_f1'] = float(np.mean(f1s))
    results['val_mean_auc'] = float(np.mean(aucs))
    results['val_trial_mean_bal_acc'] = float(np.mean(trial_bal_accs)) if has_val_trials else results['val_mean_bal_acc']
    results['val_trial_mean_auc'] = float(np.mean(trial_aucs)) if has_val_trials else results['val_mean_auc']
    results['optimal_thresholds'] = optimal_thresholds
    results['optimal_thresholds_trial'] = optimal_thresholds_trial
    
    return results


@torch.no_grad()
def update_bn_multimodal(
    loader,
    model: nn.Module,
    device: torch.device
) -> None:
    """
    Recompute BatchNorm running statistics for multimodal dict batches.
    """
    momenta = {}
    has_bn = False

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum
            module.momentum = None
            has_bn = True

    if not has_bn:
        return

    was_training = model.training
    model.train()

    for batch in loader:
        eeg = batch['eeg'].to(device)
        eog = batch['eog'].to(device)
        emg = batch['emg'].to(device)
        de = batch.get('de', None)
        if de is not None:
            de = de.to(device)
        model(eeg, eog, emg, de=de)

    for bn_module, momentum in momenta.items():
        bn_module.momentum = momentum

    model.train(was_training)


# ============================================================================
# Main Training Function
# ============================================================================

def train(
    config: DEAPConfig,
    train_loader,
    val_loader,
    checkpoint_dir: str = 'checkpoints',
    resume_from: str = None
):
    """
    Main training function with Focal Loss, SWA, and improved LR scheduling.
    """
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize model
    model = BiMambaTransformerNetwork(
        d_model=config.d_model,
        n_classes_emotion=config.n_classes_emotion,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        eeg_layers=config.eeg_layers,
        peripheral_layers=config.peripheral_layers,
        n_tokens=config.n_tokens,
        fusion_layers=getattr(config, 'fusion_layers', 1),
        num_experts=getattr(config, 'num_experts', 2)
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    
    # ---------- Loss Functions (Dynamic Balanced Class Weights from Training Split) ----------
    train_ds = train_loader.dataset
    dynamic_weights = {}
    print("  Training set class distribution & exact balanced class weights:")
    for task in ['valence', 'arousal', 'dominance']:
        if hasattr(train_ds, task) and hasattr(train_ds, 'valid_indices'):
            lbl_arr = getattr(train_ds, task)[train_ds.valid_indices]
            val_attr = getattr(train_ds, f'valid_{task}', None)
            if val_attr is not None:
                m = val_attr[train_ds.valid_indices].astype(bool)
            else:
                m = (lbl_arr != -1)
            valid_lbls = lbl_arr[m].astype(np.int64)
            n0 = int((valid_lbls == 0).sum())
            n1 = int((valid_lbls == 1).sum())
            ntot = max(n0 + n1, 1)
            w0 = ntot / (2.0 * max(n0, 1))
            w1 = ntot / (2.0 * max(n1, 1))
            dynamic_weights[task] = torch.tensor([w0, w1], dtype=torch.float32, device=device)
            print(f"    {task.capitalize():<10s}: Low={n0:5d} ({n0/ntot*100:4.1f}%), High={n1:5d} ({n1/ntot*100:4.1f}%) -> weights=[{w0:.3f}, {w1:.3f}]")
        else:
            dynamic_weights[task] = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
            
    valence_weights = dynamic_weights['valence']
    arousal_weights = dynamic_weights['arousal']
    dominance_weights = dynamic_weights['dominance']
    
    if config.use_focal_loss:
        print("Using Focal Loss (gamma={:.1f})".format(config.focal_gamma))
        criterion_valence = FocalLoss(
            alpha=valence_weights, gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            reduction='none'
        )
        criterion_arousal = FocalLoss(
            alpha=arousal_weights, gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            reduction='none'
        )
        criterion_dominance = FocalLoss(
            alpha=dominance_weights, gamma=config.focal_gamma,
            label_smoothing=config.label_smoothing,
            reduction='none'
        )
    else:
        print(f"Using Weighted Cross-Entropy Loss (label_smoothing={config.label_smoothing})")
        criterion_valence = nn.CrossEntropyLoss(
            weight=valence_weights, label_smoothing=config.label_smoothing,
            reduction='none'
        )
        criterion_arousal = nn.CrossEntropyLoss(
            weight=arousal_weights, label_smoothing=config.label_smoothing,
            reduction='none'
        )
        criterion_dominance = nn.CrossEntropyLoss(
            weight=dominance_weights, label_smoothing=config.label_smoothing,
            reduction='none'
        )
    
    # Domain Loss for DANN
    criterion_domain = None
    if config.use_dann:
        criterion_domain = nn.CrossEntropyLoss()
    
    # Equal task weights for unbiased multi-task learning
    task_weights = {'valence': 1.0, 'arousal': 1.0, 'dominance': 1.0}
    total_weight = sum(task_weights.values())
    
    # ---------- Optimizer ----------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # ---------- LR Scheduler ----------
    warmup_epochs = config.warmup_epochs
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.3,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.num_epochs - warmup_epochs),
        eta_min=1e-6
    )
    
    # ---------- SWA ----------
    swa_model = None
    swa_scheduler = None
    swa_updates = 0
    if config.use_swa:
        from torch.optim.swa_utils import AveragedModel, SWALR
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=config.swa_lr)
        print(f"SWA enabled from epoch {config.swa_start_epoch}")
    
    # ---------- Metrics & Early Stopping ----------
    metrics_tracker = MetricsTracker()
    
    val_metric_name = getattr(config, 'val_metric', 'val_balanced_acc')
    if val_metric_name == 'val_loss':
        es_mode = 'min'
        best_metric_val = float('inf')
        metric_label = 'Val Loss'
    elif val_metric_name == 'val_auc':
        es_mode = 'max'
        best_metric_val = -float('inf')
        metric_label = 'Val Mean AUC'
    elif val_metric_name in ['val_balanced_acc', 'val_bal_acc']:
        es_mode = 'max'
        best_metric_val = -float('inf')
        metric_label = 'Val Mean Bal-Acc'
    else:  # val_macro_f1
        es_mode = 'max'
        best_metric_val = -float('inf')
        metric_label = 'Val Mean Macro-F1'
        
    early_stopping = EarlyStopping(patience=15, mode=es_mode)
    
    # Resume from checkpoint
    start_epoch = 0
    best_val_acc = 0.0
    best_val_bal_acc = 0.0
    best_val_f1 = 0.0
    
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if missing:
            print(f"  Note: Missing keys in checkpoint (initialized fresh): {missing}")
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_metric_val = checkpoint.get('best_metric_val', checkpoint.get('best_val_acc', best_metric_val))
        
    # ---------- Training Loop ----------
    print(f"\nStarting training for {config.num_epochs} epochs (warmup: {warmup_epochs} epochs)...")
    print(f"  LR: {config.learning_rate}, WD: {config.weight_decay}")
    print(f"  Model Selection Criterion: {metric_label} ({val_metric_name})")
    print(f"  Focal Loss: {config.use_focal_loss}, Label Smoothing: {config.label_smoothing}")
    print(f"  DANN: {config.use_dann}, Mixup: {config.use_mixup}")
    print("-" * 70)
    
    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()
        
        # Determine if we're in SWA phase
        in_swa_phase = config.use_swa and epoch >= config.swa_start_epoch
        
        # Train
        train_metrics = train_epoch(
            model, train_loader,
            criterion_valence, criterion_arousal, criterion_dominance, criterion_domain,
            task_weights, total_weight,
            optimizer, device,
            config, epoch, config.num_epochs
        )
        
        # Validate
        val_metrics = validate(
            model, val_loader,
            criterion_valence, criterion_arousal, criterion_dominance,
            task_weights, total_weight, device
        )
        
        # Update scheduler
        if in_swa_phase:
            swa_model.update_parameters(model)
            swa_scheduler.step()
            swa_updates += 1
        elif epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update metrics
        epoch_metrics = {
            **train_metrics,
            **val_metrics,
            'learning_rate': current_lr
        }
        metrics_tracker.update(epoch_metrics)
        
        epoch_time = time.time() - epoch_start
        
        # Print progress with segment & trial balanced accuracy and AUC
        swa_tag = " [SWA]" if in_swa_phase else ""
        de_gate_str = f" | DE-Gate: {torch.tanh(model.de_gate).item():+.3f}" if hasattr(model, 'de_gate') else ""
        print(f"Epoch {epoch+1}/{config.num_epochs} ({epoch_time:.1f}s){swa_tag}")
        print(f"  Train Loss: {train_metrics['train_loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}{de_gate_str}")
        print(f"  Val Seg BA   - V: {val_metrics['val_valence_bal_acc']:.4f}, A: {val_metrics['val_arousal_bal_acc']:.4f}, D: {val_metrics['val_dominance_bal_acc']:.4f} | Mean: {val_metrics['val_mean_bal_acc']:.4f}")
        print(f"  Val Seg AUC  - V: {val_metrics['val_valence_auc']:.4f}, A: {val_metrics['val_arousal_auc']:.4f}, D: {val_metrics['val_dominance_auc']:.4f} | Mean: {val_metrics['val_mean_auc']:.4f}")
        if 'val_trial_valence_bal_acc' in val_metrics:
            print(f"  Val Trial BA - V: {val_metrics['val_trial_valence_bal_acc']:.4f}, A: {val_metrics['val_trial_arousal_bal_acc']:.4f}, D: {val_metrics['val_trial_dominance_bal_acc']:.4f} | Mean: {val_metrics['val_trial_mean_bal_acc']:.4f} (AUC: {val_metrics['val_trial_mean_auc']:.4f})")
        print(f"  LR: {current_lr:.6f}")
        
        # Model Selection based on objective validation metric
        if val_metric_name == 'val_loss':
            current_score = val_metrics['val_loss']
            is_best = current_score < best_metric_val
        elif val_metric_name == 'val_auc':
            current_score = 0.5 * (val_metrics['val_mean_auc'] + val_metrics['val_trial_mean_auc'])
            is_best = current_score > best_metric_val
        elif val_metric_name in ['val_balanced_acc', 'val_bal_acc']:
            current_score = 0.5 * (val_metrics['val_mean_bal_acc'] + val_metrics['val_trial_mean_bal_acc'])
            is_best = current_score > best_metric_val
        else:
            current_score = val_metrics['val_mean_f1']
            is_best = current_score > best_metric_val
            
        if is_best:
            best_metric_val = current_score
            best_val_acc = val_metrics['val_mean_acc']
            best_val_bal_acc = val_metrics['val_mean_bal_acc']
            best_val_f1 = val_metrics['val_mean_f1']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric_val': best_metric_val,
                'best_metric_name': val_metric_name,
                'best_val_acc': best_val_acc,
                'best_val_bal_acc': best_val_bal_acc,
                'best_val_f1': best_val_f1,
                'optimal_thresholds': val_metrics.get('optimal_thresholds', {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5}),
                'optimal_thresholds_trial': val_metrics.get('optimal_thresholds_trial', val_metrics.get('optimal_thresholds', {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5})),
                'val_metrics': val_metrics,
                'config': config.__dict__
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"  ★ New best model! ({metric_label}: {best_metric_val:.4f}, Seg BA: {best_val_bal_acc:.4f}, Trial BA: {val_metrics['val_trial_mean_bal_acc']:.4f}, Trial AUC: {val_metrics['val_trial_mean_auc']:.4f})")
        
        # Save latest checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric_val': best_metric_val,
            'best_val_acc': best_val_acc,
            'best_val_bal_acc': best_val_bal_acc,
            'best_val_f1': best_val_f1,
            'optimal_thresholds': val_metrics.get('optimal_thresholds', {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5}),
            'optimal_thresholds_trial': val_metrics.get('optimal_thresholds_trial', val_metrics.get('optimal_thresholds', {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5})),
            'val_metrics': val_metrics,
            'config': config.__dict__
        }, os.path.join(checkpoint_dir, 'latest_model.pth'))
        
        # Early stopping based on chosen validation metric
        if not in_swa_phase and early_stopping(current_score):
            print(f"\nEarly stopping triggered at epoch {epoch+1} ({metric_label}: {current_score:.4f})")
            break
            
        print()
    
    # ---------- SWA: Update batch normalization ----------
    if config.use_swa and swa_model is not None and swa_updates > 0:
        print("Updating SWA batch normalization statistics...")
        update_bn_multimodal(train_loader, swa_model, device=device)
        swa_val = validate(
            swa_model.module, val_loader,
            criterion_valence, criterion_arousal, criterion_dominance,
            task_weights, total_weight, device
        )

        # Save SWA model
        torch.save({
            'epoch': epoch,
            'model_state_dict': swa_model.module.state_dict(),
            'optimal_thresholds': swa_val.get('optimal_thresholds', {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5}),
            'optimal_thresholds_trial': swa_val.get('optimal_thresholds_trial', swa_val.get('optimal_thresholds', {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5})),
            'val_metrics': swa_val,
            'config': config.__dict__
        }, os.path.join(checkpoint_dir, 'swa_model.pth'))
        print("SWA model saved.")
    elif config.use_swa and swa_model is not None:
        print("SWA skipped: early stopping occurred before swa_start_epoch.")
    
    # Save training history
    metrics_tracker.save(os.path.join(checkpoint_dir, 'training_history.json'))
    
    print("-" * 70)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    
    return model, metrics_tracker


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader,
    device: torch.device,
    optimal_thresholds: dict = None,
    optimal_thresholds_trial: dict = None
) -> dict:
    """
    Evaluate model on test set with detailed segment-level, trial-level, and per-subject metrics,
    plus post-hoc validation-calibrated threshold evaluation when available.
    """
    model.eval()
    
    task_preds = {'valence': [], 'arousal': [], 'dominance': []}
    task_probs = {'valence': [], 'arousal': [], 'dominance': []}
    task_targets = {'valence': [], 'arousal': [], 'dominance': []}
    task_sub_ids = {'valence': [], 'arousal': [], 'dominance': []}
    task_tri_ids = {'valence': [], 'arousal': [], 'dominance': []}
    
    for batch in tqdm(test_loader, desc='Evaluating', leave=False):
        eeg = batch['eeg'].to(device)
        eog = batch['eog'].to(device)
        emg = batch['emg'].to(device)
        de = batch.get('de', None)
        if de is not None:
            de = de.to(device)
        labels = batch['labels']
        valid_mask = batch.get('valid_mask', None)
        
        sub_ids = None
        if 'subject_id' in batch:
            sub_ids = batch['subject_id'].cpu().numpy()
        elif 'participant_id' in batch:
            sub_ids = batch['participant_id'].cpu().numpy() + 1
            
        tri_ids = None
        if 'trial_id' in batch:
            tri_ids = batch['trial_id'].cpu().numpy()
            
        outputs = model(eeg, eog, emg, de=de)
        
        for task in ['valence', 'arousal', 'dominance']:
            if isinstance(valid_mask, dict) and task in valid_mask:
                mask = valid_mask[task].cpu().numpy().astype(bool)
            elif isinstance(labels, dict):
                mask = (labels[task] != -1).cpu().numpy().astype(bool)
            else:
                mask = np.ones(eeg.size(0), dtype=bool)
                
            if np.any(mask):
                probs = F.softmax(outputs[task], dim=1).cpu().numpy()[mask]
                preds = outputs[task].argmax(dim=1).cpu().numpy()[mask]
                if isinstance(labels, dict):
                    lbl = labels[task].cpu().numpy()[mask]
                else:
                    lbl = labels.cpu().numpy()[mask]
                task_preds[task].append(preds)
                task_probs[task].append(probs)
                task_targets[task].append(lbl)
                if sub_ids is not None:
                    task_sub_ids[task].append(sub_ids[mask])
                if tri_ids is not None:
                    task_tri_ids[task].append(tri_ids[mask])
        
    # Segment-level & Trial-level evaluation (Standard threshold = 0.50)
    segment_metrics = {}
    trial_metrics = {}
    segment_cal_metrics = {}
    trial_cal_metrics = {}
    per_subject_trial_summary = {}
    has_any_trial = False
    has_calibrated = isinstance(optimal_thresholds, dict) and len(optimal_thresholds) > 0
    if optimal_thresholds_trial is None:
        optimal_thresholds_trial = optimal_thresholds
    
    for task in ['valence', 'arousal', 'dominance']:
        if len(task_preds[task]) > 0:
            preds = np.concatenate(task_preds[task])
            probs = np.concatenate(task_probs[task])
            targets = np.concatenate(task_targets[task])
            segment_metrics[task] = compute_classification_metrics(targets, preds, probs)
            
            tau_seg = float(optimal_thresholds.get(task, 0.5)) if has_calibrated else 0.5
            tau_tri = float(optimal_thresholds_trial.get(task, tau_seg)) if (optimal_thresholds_trial is not None) else tau_seg
            if has_calibrated:
                preds_cal = (probs[:, 1] >= tau_seg).astype(np.int64)
                segment_cal_metrics[task] = compute_classification_metrics(targets, preds_cal, probs)
            
            if len(task_sub_ids[task]) > 0 and len(task_tri_ids[task]) > 0:
                subjs = np.concatenate(task_sub_ids[task])
                tris = np.concatenate(task_tri_ids[task])
                t_m, _ = aggregate_trial_predictions(subjs, tris, probs, targets, threshold=0.5)
                trial_metrics[task] = t_m
                if has_calibrated:
                    t_m_cal, _ = aggregate_trial_predictions(subjs, tris, probs, targets, threshold=tau_tri)
                    trial_cal_metrics[task] = t_m_cal
                has_any_trial = True

                # Per-subject trial diagnostics
                per_subject_trial_summary[task] = {}
                for s_id in sorted(np.unique(subjs).tolist()):
                    s_mask = (subjs == s_id)
                    s_tm, _ = aggregate_trial_predictions(subjs[s_mask], tris[s_mask], probs[s_mask], targets[s_mask], threshold=tau_tri)
                    per_subject_trial_summary[task][int(s_id)] = {
                        'n_trials': s_tm['n_samples'],
                        'low': s_tm['class_distribution']['class_0'],
                        'high': s_tm['class_distribution']['class_1'],
                        'bal_acc': s_tm['balanced_accuracy'],
                        'auc': s_tm['auc_roc']
                    }
        else:
            empty_m = {'accuracy': 0.0, 'balanced_accuracy': 0.0, 'macro_f1': 0.0, 'auc_roc': 0.5}
            segment_metrics[task] = empty_m
            if has_calibrated:
                segment_cal_metrics[task] = empty_m
        
    print("\n" + format_metrics_table(segment_metrics, title="TEST SET: SEGMENT-LEVEL RESULTS (Threshold = 0.50)"))
    
    # Trial-level evaluation (if metadata available)
    if has_any_trial and len(trial_metrics) == 3:
        print("\n" + format_metrics_table(trial_metrics, title="TEST SET: TRIAL-LEVEL AGGREGATED RESULTS (Threshold = 0.50)"))
        
    # Post-hoc validation-calibrated threshold evaluation
    if has_calibrated and len(segment_cal_metrics) == 3:
        seg_thresh_str = ", ".join([f"{t[0].upper()}={optimal_thresholds.get(t, 0.5):.2f}" for t in ['valence', 'arousal', 'dominance']])
        print("\n" + format_metrics_table(
            segment_cal_metrics,
            title=f"TEST SET: SEGMENT-LEVEL RESULTS (Val-Calibrated: {seg_thresh_str})"
        ))
        if has_any_trial and len(trial_cal_metrics) == 3:
            tri_thresh_str = ", ".join([f"{t[0].upper()}={optimal_thresholds_trial.get(t, 0.5):.2f}" for t in ['valence', 'arousal', 'dominance']])
            print("\n" + format_metrics_table(
                trial_cal_metrics,
                title=f"TEST SET: TRIAL-LEVEL AGGREGATED RESULTS (Val-Calibrated: {tri_thresh_str})"
            ))

    # Print concise per-subject trial breakdown to expose single-subject label imbalance (e.g. Subject 3)
    if per_subject_trial_summary:
        print("\n  Per-Subject Trial Diagnostics (Class counts Low/High | Bal-Acc | AUC):")
        all_test_s = sorted({s for td in per_subject_trial_summary.values() for s in td.keys()})
        for s_id in all_test_s:
            parts = []
            for task in ['valence', 'arousal', 'dominance']:
                info = per_subject_trial_summary.get(task, {}).get(s_id)
                if info:
                    auc_str = f"{info['auc']:.3f}" if (info['low'] > 0 and info['high'] > 0) else " N/A "
                    parts.append(f"{task[0].upper()}[{info['low']:2d}/{info['high']:2d}] BA={info['bal_acc']:.3f} AUC={auc_str}")
            print(f"    Subject {s_id:02d}: " + "  |  ".join(parts))
        
    metrics = {
        'segment': segment_metrics,
        'trial': trial_metrics if has_any_trial else None,
        'optimal_thresholds': optimal_thresholds,
        'optimal_thresholds_trial': optimal_thresholds_trial,
        'segment_calibrated': segment_cal_metrics if has_calibrated else None,
        'trial_calibrated': trial_cal_metrics if (has_calibrated and has_any_trial) else None,
        'per_subject_trial': per_subject_trial_summary if per_subject_trial_summary else None,
        'mean_accuracy': float(np.mean([segment_metrics[t]['accuracy'] for t in ['valence', 'arousal', 'dominance']])),
        'mean_balanced_accuracy': float(np.mean([segment_metrics[t]['balanced_accuracy'] for t in ['valence', 'arousal', 'dominance']])),
        'mean_f1': float(np.mean([segment_metrics[t]['macro_f1'] for t in ['valence', 'arousal', 'dominance']])),
        'mean_auc': float(np.mean([segment_metrics[t]['auc_roc'] for t in ['valence', 'arousal', 'dominance']]))
    }
    
    # Backwards compatibility: keep task keys on root level
    for task in ['valence', 'arousal', 'dominance']:
        metrics[task] = segment_metrics[task]
        
    return metrics


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='BiMamba-Transformer Training')
    
    # Actions
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model')
    
    # Data paths
    parser.add_argument('--data_path', type=str, default='data/DEAP/deap_preprocessed.npz',
                        help='Path to preprocessed data')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    
    # Training hyperparameters (overrides config defaults)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--val_metric', type=str, default='val_balanced_acc',
                        choices=['val_balanced_acc', 'val_auc', 'val_macro_f1', 'val_loss'],
                        help='Validation metric for model selection & early stopping')
    parser.add_argument('--val_subjects', type=int, nargs='+', default=None,
                        help='List of validation subject IDs')
    parser.add_argument('--val_size', type=float, default=0.20,
                        help='Fraction of training subjects for validation')
    parser.add_argument('--use_focal_loss', action='store_true', help='Enable Focal Loss (for ablation)')
    parser.add_argument('--no_focal_loss', action='store_true', help='Disable Focal Loss')
    parser.add_argument('--use_dann', action='store_true', help='Enable DANN domain adversarial alignment (for ablation)')
    parser.add_argument('--use_mixup', action='store_true', help='Enable Mixup augmentation')
    parser.add_argument('--no_mixup', action='store_true', help='Disable Mixup augmentation (for ablation)')
    parser.add_argument('--no_de', action='store_true', help='Disable DE feature branch (for ablation)')
    parser.add_argument('--norm_level', type=str, default=None, choices=['trial', 'subject'],
                        help='Waveform normalization mode (default: trial)')
    parser.add_argument('--no_swa', action='store_true', help='Disable SWA')
    
    # Evaluation mode
    parser.add_argument('--subject_independent', action='store_true', default=True)
    parser.add_argument('--test_subjects', type=int, nargs='+', default=[1, 2, 3])
    
    # Other
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create config from defaults, override with args
    config = DEAPConfig(
        preprocessed_path=args.data_path,
        subject_independent=args.subject_independent,
        test_subjects=args.test_subjects,
        val_subjects=args.val_subjects,
        val_size=args.val_size,
        val_metric=args.val_metric,
        device=args.device,
        num_workers=args.num_workers,
        random_state=args.seed
    )
    
    # Apply CLI overrides
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.use_focal_loss:
        config.use_focal_loss = True
    if args.no_focal_loss:
        config.use_focal_loss = False
    if args.use_dann:
        config.use_dann = True
    if args.use_mixup:
        config.use_mixup = True
    if args.no_mixup:
        config.use_mixup = False
    if args.no_de:
        config.use_de = False
    if args.norm_level is not None:
        config.norm_level = args.norm_level
    if args.no_swa:
        config.use_swa = False
    
    # Training
    if args.train:
        print("=" * 70)
        print("BiMamba-Transformer Training (Optimized Protocol)")
        print("=" * 70)
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_deap_dataloaders(
            data_path=config.preprocessed_path,
            batch_size=config.batch_size,
            val_size=config.val_size,
            val_subjects=config.val_subjects,
            subject_independent=config.subject_independent,
            test_subjects=config.test_subjects,
            random_state=config.random_state,
            num_workers=config.num_workers,
            filter_ambiguous=True,
            waveform_norm=config.norm_level,
            use_de=config.use_de
        )
        
        # Train
        model, metrics = train(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=args.checkpoint_dir,
            resume_from=args.checkpoint
        )
        
        # Final evaluation on test set
        print("\n" + "=" * 70)
        print("Final Evaluation on Test Set")
        print("=" * 70)
        
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Load best model
        best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
        opt_thresholds = None
        opt_thresholds_trial = None
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            opt_thresholds = checkpoint.get('optimal_thresholds', None)
            opt_thresholds_trial = checkpoint.get('optimal_thresholds_trial', opt_thresholds)
        
        test_metrics = evaluate(
            model, test_loader, device,
            optimal_thresholds=opt_thresholds,
            optimal_thresholds_trial=opt_thresholds_trial
        )
        
        # Also evaluate SWA model if available
        swa_path = os.path.join(args.checkpoint_dir, 'swa_model.pth')
        if os.path.exists(swa_path):
            print(f"\n--- SWA Model Evaluation ---")
            swa_checkpoint = torch.load(swa_path, map_location=device, weights_only=False)
            model.load_state_dict(swa_checkpoint['model_state_dict'], strict=False)
            swa_thresholds = swa_checkpoint.get('optimal_thresholds', opt_thresholds)
            swa_thresholds_trial = swa_checkpoint.get('optimal_thresholds_trial', swa_thresholds)
            swa_metrics = evaluate(
                model, test_loader, device,
                optimal_thresholds=swa_thresholds,
                optimal_thresholds_trial=swa_thresholds_trial
            )
            
            # Save comparison
            test_metrics['swa'] = swa_metrics
        
        # Save test results
        with open(os.path.join(args.checkpoint_dir, 'test_results.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    # Evaluation only
    if args.evaluate and not args.train:
        print("=" * 70)
        print("Model Evaluation")
        print("=" * 70)
        
        if args.checkpoint is None:
            print("Error: Please specify --checkpoint for evaluation")
            return
        
        device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_config = checkpoint.get('config', {})
        if saved_config:
            # Backward compatibility for legacy checkpoints saved before fusion_layers/num_experts existed
            if saved_config.get('d_model') == 96 and 'fusion_layers' not in saved_config:
                config.fusion_layers = 2
                config.num_experts = 4
            for key in [
                'd_model', 'hidden_dim', 'dropout', 'eeg_layers',
                'peripheral_layers', 'fusion_layers', 'num_experts',
                'n_classes_emotion', 'n_tokens', 'norm_level', 'use_de'
            ]:
                if key in saved_config:
                    setattr(config, key, saved_config[key])
        if args.norm_level is not None:
            config.norm_level = args.norm_level
        if args.no_de:
            config.use_de = False
        
        model = BiMambaTransformerNetwork(
            d_model=config.d_model,
            n_classes_emotion=config.n_classes_emotion,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            eeg_layers=config.eeg_layers,
            peripheral_layers=config.peripheral_layers,
            n_tokens=config.n_tokens,
            fusion_layers=getattr(config, 'fusion_layers', 1),
            num_experts=getattr(config, 'num_experts', 2)
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        _, _, test_loader = create_deap_dataloaders(
            data_path=config.preprocessed_path,
            batch_size=config.batch_size,
            val_size=config.val_size,
            val_subjects=config.val_subjects,
            subject_independent=config.subject_independent,
            test_subjects=config.test_subjects,
            random_state=config.random_state,
            num_workers=config.num_workers,
            filter_ambiguous=True,
            waveform_norm=getattr(config, 'norm_level', 'trial'),
            use_de=getattr(config, 'use_de', True)
        )
        
        opt_t = checkpoint.get('optimal_thresholds', None)
        opt_t_trial = checkpoint.get('optimal_thresholds_trial', opt_t)
        test_metrics = evaluate(
            model, test_loader, device,
            optimal_thresholds=opt_t,
            optimal_thresholds_trial=opt_t_trial
        )
        
        # Save results
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        out_eval_path = os.path.join(args.checkpoint_dir, 'test_results.json')
        with open(out_eval_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        print(f"\nResults saved to: {out_eval_path}")


if __name__ == '__main__':
    main()
