"""
Subject-Independent Cross-Validation (K-Fold / LOSO) for BiMamba-Transformer on DEAP dataset.

Features:
- Configurable K-Fold Cross-Validation (default 8 folds, 4 subjects per fold) or 32-fold LOSO
- Automatic partition of 32 subjects into Train / Validation / Test sets
- Representative validation set (at least 4 validation subjects per fold)
- Model selection and early stopping based on objective validation metric (val_balanced_acc, val_auc, etc.)
- Comprehensive evaluation at both Segment-level and Trial-level (with standard 0.5 and validation-calibrated thresholds)
- SWA (Stochastic Weight Averaging) model evaluation
- Checkpoint resume capability (skips already-trained folds automatically)
- Robust JSON serialization avoiding numpy int64/float32 serialization errors
- Aggregation of results across folds with Mean ± Standard Deviation
- Outputs detailed summary tables, text reports, and JSON metrics

Usage:
    # 8-fold Cross-Validation (standard 4-subjects-per-fold, 80 epochs per fold)
    python evaluate/cv.py --k_folds 8 --epochs 80

    # Resume interrupted CV run (default behavior: resumes completed folds automatically)
    python evaluate/cv.py --k_folds 8

    # Force re-training all folds from scratch
    python evaluate/cv.py --k_folds 8 --no_resume

    # Run only a specific fold (e.g. Fold 0)
    python evaluate/cv.py --k_folds 8 --fold 0

    # Full 32-fold Leave-One-Subject-Out (LOSO)
    python evaluate/cv.py --loso --epochs 80
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Prefer local package "code" over stdlib
existing_code_module = sys.modules.get('code')
if existing_code_module is not None and not hasattr(existing_code_module, '__path__'):
    del sys.modules['code']

from code.model import BiMambaTransformerNetwork
from code.data import create_deap_dataloaders
from code.data.config import DEAPConfig
from code.utils.metrics import (
    compute_classification_metrics,
    aggregate_trial_predictions,
    format_metrics_table
)
from train import train, evaluate


# ============================================================================
# Robust JSON Serialization Utilities
# ============================================================================

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts numpy scalars and arrays to native Python types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
        return super().default(obj)


def make_json_serializable(data: Any) -> Any:
    """Recursively convert numpy types to native Python types for clean JSON dumps."""
    if isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        return float(data)
    elif isinstance(data, (np.bool_, bool)):
        return bool(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {str(k): make_json_serializable(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple, set)):
        return [make_json_serializable(x) for x in data]
    return data


# ============================================================================
# Subject Split Generation
# ============================================================================

def generate_subject_splits(
    all_subjects: List[int],
    k_folds: int = 8,
    n_val_subjects: int = 4,
    random_state: int = 42
) -> List[Dict[str, Any]]:
    """
    Generate subject-independent folds with pure native Python integer IDs.

    Args:
        all_subjects: List of participant IDs (e.g. 1..32)
        k_folds: Number of folds (e.g. 8 for 4-subjects-per-fold, 32 for LOSO)
        n_val_subjects: Number of validation subjects per fold (min 4 recommended)
        random_state: Random seed for validation subject shuffling

    Returns:
        List of fold dicts with 'fold', 'test_subjects', 'val_subjects', 'train_subjects'
    """
    all_subjects = [int(s) for s in sorted(list(all_subjects))]
    n_total = len(all_subjects)

    if k_folds <= 0 or k_folds > n_total:
        raise ValueError(f"k_folds must be between 1 and {n_total}")

    # Partition subjects into k test chunks
    # For 32 subjects & 8 folds: 4 subjects per fold
    chunks = np.array_split(all_subjects, k_folds)
    test_chunks = [[int(s) for s in sorted(chunk.tolist())] for chunk in chunks]

    folds = []
    rng = np.random.RandomState(random_state)

    for fold_idx in range(k_folds):
        test_subjs = [int(s) for s in test_chunks[fold_idx]]
        remaining = [int(s) for s in all_subjects if s not in test_subjs]

        # Select validation subjects from remaining
        actual_val_n = min(n_val_subjects, len(remaining) - 1)
        shuffled = list(remaining)
        rng.shuffle(shuffled)
        val_subjs = [int(s) for s in sorted(shuffled[:actual_val_n])]
        train_subjs = [int(s) for s in sorted([s for s in remaining if s not in val_subjs])]

        folds.append({
            'fold': int(fold_idx),
            'test_subjects': test_subjs,
            'val_subjects': val_subjs,
            'train_subjects': train_subjs
        })

    return folds


# ============================================================================
# Summary Formatting
# ============================================================================

def format_cv_summary_table(
    fold_metrics_list: List[Dict],
    title: str = "CROSS-VALIDATION RESULTS SUMMARY"
) -> str:
    """
    Format fold-by-fold results and overall Mean ± Std into a clean table.
    """
    lines = []
    lines.append("=" * 96)
    lines.append(f"  {title}")
    lines.append("=" * 96)
    header = (
        f"  {'Fold':<6s} {'Test Subjects':<18s} "
        f"{'Val-Acc':>8s} {'Bal-Acc':>8s} {'Maj-Base':>9s} "
        f"{'Macro-F1':>9s} {'AUC':>8s} {'MCC':>8s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 92)

    accs = []
    bal_accs = []
    maj_bases = []
    f1s = []
    aucs = []
    mccs = []

    for item in fold_metrics_list:
        fold = item['fold'] + 1  # 1-indexed for display
        subjs_str = str(item['test_subjects'])
        m = item.get('mean', {})
        acc = m.get('accuracy', 0.0)
        bal = m.get('balanced_accuracy', 0.0)
        maj = m.get('majority_baseline', 0.0)
        f1 = m.get('macro_f1', 0.0)
        auc = m.get('auc_roc', 0.0)
        mcc = m.get('mcc', 0.0)

        accs.append(acc)
        bal_accs.append(bal)
        maj_bases.append(maj)
        f1s.append(f1)
        aucs.append(auc)
        mccs.append(mcc)

        lines.append(
            f"  {fold:<6d} {subjs_str:<18s} "
            f"{acc:>8.4f} {bal:>8.4f} {maj:>9.4f} "
            f"{f1:>9.4f} {auc:>8.4f} {mcc:>8.4f}"
        )

    lines.append("  " + "-" * 92)

    # Compute Mean and Std
    if accs:
        mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
        mean_bal, std_bal = float(np.mean(bal_accs)), float(np.std(bal_accs))
        mean_maj, std_maj = float(np.mean(maj_bases)), float(np.std(maj_bases))
        mean_f1, std_f1 = float(np.mean(f1s)), float(np.std(f1s))
        mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
        mean_mcc, std_mcc = float(np.mean(mccs)), float(np.std(mccs))

        lines.append(
            f"  {'MEAN':<6s} {'All Folds':<18s} "
            f"{mean_acc:>8.4f} {mean_bal:>8.4f} {mean_maj:>9.4f} "
            f"{mean_f1:>9.4f} {mean_auc:>8.4f} {mean_mcc:>8.4f}"
        )
        lines.append(
            f"  {'STD':<6s} {'—':<18s} "
            f"{std_acc:>8.4f} {std_bal:>8.4f} {std_maj:>9.4f} "
            f"{std_f1:>9.4f} {std_auc:>8.4f} {std_mcc:>8.4f}"
        )
    lines.append("=" * 96)
    return "\n".join(lines)


# ============================================================================
# Cross-Validation Runner
# ============================================================================

def run_cross_validation(
    config: DEAPConfig,
    data_path: str,
    k_folds: int = 8,
    specific_fold: Optional[int] = None,
    output_dir: str = 'cv_results',
    val_subjects_count: int = 4,
    resume: bool = True
) -> dict:
    """
    Run full K-fold Subject-Independent Cross-Validation with resume support and robust JSON serialization.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    # Load data file to get available participants
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load master dataset into RAM once (avoids duplicating 2.5GB and peak memory spikes)
    print(f"Loading master DEAP dataset from {data_path}...")
    npz_data = np.load(data_path, allow_pickle=True)
    shared_data_dict = {key: npz_data[key] for key in npz_data.files}
    del npz_data

    available_subjects = [int(s) for s in sorted(list(np.unique(shared_data_dict['participant_id'])))]
    print(f"Available subjects ({len(available_subjects)}): {available_subjects}")

    # Generate splits
    splits = generate_subject_splits(
        available_subjects,
        k_folds=k_folds,
        n_val_subjects=val_subjects_count,
        random_state=config.random_state
    )

    if specific_fold is not None:
        if specific_fold < 0 or specific_fold >= len(splits):
            raise ValueError(f"Specific fold {specific_fold} out of range [0, {len(splits)-1}]")
        splits_to_run = [splits[specific_fold]]
        print(f"\nRunning ONLY Fold {specific_fold + 1} of {k_folds}...")
    else:
        splits_to_run = splits
        print(f"\nRunning full {k_folds}-Fold Cross-Validation...")

    all_segment_results = []
    all_trial_results = []
    all_trial_cal_results = []
    all_swa_trial_results = []

    for fold_info in splits_to_run:
        fold = int(fold_info['fold'])
        test_subjs = [int(s) for s in fold_info['test_subjects']]
        val_subjs = [int(s) for s in fold_info['val_subjects']]
        train_subjs = [int(s) for s in fold_info['train_subjects']]

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        best_path = os.path.join(fold_dir, 'best_model.pth')
        history_path = os.path.join(fold_dir, 'training_history.json')
        swa_path = os.path.join(fold_dir, 'swa_model.pth')

        # Check if this fold has already completed training
        is_already_trained = resume and os.path.exists(best_path) and os.path.exists(history_path)

        print("\n" + "#" * 80)
        if is_already_trained:
            print(f"  [RESUME] Fold {fold + 1}/{len(splits)} already trained! Found checkpoint in {best_path}")
            print(f"  Test  Subjects: {test_subjs}")
            print(f"  Val   Subjects: {val_subjs}")
            print(f"  Skipping training phase and proceeding straight to test evaluation...")
        else:
            print(f"  Starting Fold {fold + 1}/{len(splits)}")
            print(f"  Test  Subjects: {test_subjs}")
            print(f"  Val   Subjects: {val_subjs}")
            print(f"  Train Subjects: {train_subjs}")
        print("#" * 80)

        # Set seed per fold for reproducibility
        torch.manual_seed(config.random_state + fold)
        np.random.seed(config.random_state + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.random_state + fold)

        # Create Dataloaders for this fold (zero-copy from shared_data_dict)
        train_loader, val_loader, test_loader = create_deap_dataloaders(
            data_path=data_path,
            batch_size=config.batch_size,
            val_subjects=val_subjs,
            subject_independent=True,
            test_subjects=test_subjs,
            random_state=config.random_state + fold,
            num_workers=config.num_workers,
            filter_ambiguous=True,
            waveform_norm=getattr(config, 'norm_level', 'trial'),
            use_de=getattr(config, 'use_de', True),
            data_dict=shared_data_dict
        )

        if is_already_trained:
            # Instantiate model architecture and load checkpoint weights directly
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
        else:
            # Train model on this fold
            model, metrics_tracker = train(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=fold_dir
            )

        # Load best model from this fold
        opt_thresholds = None
        opt_thresholds_trial = None
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            opt_thresholds = checkpoint.get('optimal_thresholds', None)
            opt_thresholds_trial = checkpoint.get('optimal_thresholds_trial', opt_thresholds)

        # Evaluate on test set
        test_eval = evaluate(
            model, test_loader, device,
            optimal_thresholds=opt_thresholds,
            optimal_thresholds_trial=opt_thresholds_trial
        )

        # Collect segment results
        seg_record = {
            'fold': int(fold),
            'test_subjects': [int(s) for s in test_subjs],
            'val_subjects': [int(s) for s in val_subjs],
            'mean': {
                'accuracy': float(test_eval['mean_accuracy']),
                'balanced_accuracy': float(test_eval['mean_balanced_accuracy']),
                'majority_baseline': float(np.mean([test_eval[t]['majority_baseline'] for t in ['valence', 'arousal', 'dominance']])),
                'macro_f1': float(test_eval['mean_f1']),
                'auc_roc': float(test_eval['mean_auc']),
                'mcc': float(np.mean([test_eval[t]['mcc'] for t in ['valence', 'arousal', 'dominance']]))
            },
            'tasks': {t: test_eval[t] for t in ['valence', 'arousal', 'dominance']}
        }
        all_segment_results.append(seg_record)

        # Collect trial results
        trial_record = None
        if test_eval.get('trial'):
            trial_eval = test_eval['trial']
            trial_record = {
                'fold': int(fold),
                'test_subjects': [int(s) for s in test_subjs],
                'val_subjects': [int(s) for s in val_subjs],
                'mean': {
                    'accuracy': float(np.mean([trial_eval[t]['accuracy'] for t in ['valence', 'arousal', 'dominance']])),
                    'balanced_accuracy': float(np.mean([trial_eval[t]['balanced_accuracy'] for t in ['valence', 'arousal', 'dominance']])),
                    'majority_baseline': float(np.mean([trial_eval[t]['majority_baseline'] for t in ['valence', 'arousal', 'dominance']])),
                    'macro_f1': float(np.mean([trial_eval[t]['macro_f1'] for t in ['valence', 'arousal', 'dominance']])),
                    'auc_roc': float(np.mean([trial_eval[t]['auc_roc'] for t in ['valence', 'arousal', 'dominance']])),
                    'mcc': float(np.mean([trial_eval[t]['mcc'] for t in ['valence', 'arousal', 'dominance']]))
                },
                'tasks': {t: trial_eval[t] for t in ['valence', 'arousal', 'dominance']}
            }
            all_trial_results.append(trial_record)

        # Collect trial-calibrated results
        trial_cal_record = None
        if test_eval.get('trial_calibrated'):
            trial_cal = test_eval['trial_calibrated']
            trial_cal_record = {
                'fold': int(fold),
                'test_subjects': [int(s) for s in test_subjs],
                'val_subjects': [int(s) for s in val_subjs],
                'optimal_thresholds': test_eval.get('optimal_thresholds'),
                'optimal_thresholds_trial': test_eval.get('optimal_thresholds_trial'),
                'mean': {
                    'accuracy': float(np.mean([trial_cal[t]['accuracy'] for t in ['valence', 'arousal', 'dominance']])),
                    'balanced_accuracy': float(np.mean([trial_cal[t]['balanced_accuracy'] for t in ['valence', 'arousal', 'dominance']])),
                    'majority_baseline': float(np.mean([trial_cal[t]['majority_baseline'] for t in ['valence', 'arousal', 'dominance']])),
                    'macro_f1': float(np.mean([trial_cal[t]['macro_f1'] for t in ['valence', 'arousal', 'dominance']])),
                    'auc_roc': float(np.mean([trial_cal[t]['auc_roc'] for t in ['valence', 'arousal', 'dominance']])),
                    'mcc': float(np.mean([trial_cal[t]['mcc'] for t in ['valence', 'arousal', 'dominance']]))
                },
                'tasks': {t: trial_cal[t] for t in ['valence', 'arousal', 'dominance']}
            }
            all_trial_cal_results.append(trial_cal_record)

        # Evaluate SWA model (averaged weights over the SWA phase) when available
        swa_trial_record = None
        if os.path.exists(swa_path):
            print(f"\n--- Fold {fold + 1} SWA Model Evaluation ---")
            swa_ckpt = torch.load(swa_path, map_location=device, weights_only=False)
            model.load_state_dict(swa_ckpt['model_state_dict'], strict=False)
            swa_eval = evaluate(model, test_loader, device)
            if swa_eval.get('trial'):
                swa_trial = swa_eval['trial']
                swa_trial_record = {
                    'fold': int(fold),
                    'test_subjects': [int(s) for s in test_subjs],
                    'val_subjects': [int(s) for s in val_subjs],
                    'mean': {
                        'accuracy': float(np.mean([swa_trial[t]['accuracy'] for t in ['valence', 'arousal', 'dominance']])),
                        'balanced_accuracy': float(np.mean([swa_trial[t]['balanced_accuracy'] for t in ['valence', 'arousal', 'dominance']])),
                        'majority_baseline': float(np.mean([swa_trial[t]['majority_baseline'] for t in ['valence', 'arousal', 'dominance']])),
                        'macro_f1': float(np.mean([swa_trial[t]['macro_f1'] for t in ['valence', 'arousal', 'dominance']])),
                        'auc_roc': float(np.mean([swa_trial[t]['auc_roc'] for t in ['valence', 'arousal', 'dominance']])),
                        'mcc': float(np.mean([swa_trial[t]['mcc'] for t in ['valence', 'arousal', 'dominance']]))
                    },
                    'tasks': {t: swa_trial[t] for t in ['valence', 'arousal', 'dominance']}
                }
                all_swa_trial_results.append(swa_trial_record)

        # Save individual fold metrics JSON safely
        fold_record = {
            'fold': int(fold),
            'test_subjects': [int(s) for s in test_subjs],
            'val_subjects': [int(s) for s in val_subjs],
            'train_subjects': [int(s) for s in train_subjs],
            'segment': seg_record,
            'trial': trial_record,
            'trial_calibrated': trial_cal_record,
            'trial_swa': swa_trial_record
        }
        with open(os.path.join(fold_dir, 'fold_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable(fold_record), f, indent=2, cls=NumpyJSONEncoder)

        # Save partial progress JSON (robust to numpy scalars)
        progress_path = os.path.join(output_dir, 'cv_progress.json')
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(make_json_serializable({
                'k_folds': int(k_folds),
                'completed_folds': len(all_segment_results),
                'segment': all_segment_results,
                'trial': all_trial_results,
                'trial_calibrated': all_trial_cal_results,
                'trial_swa': all_swa_trial_results
            }), f, indent=2, cls=NumpyJSONEncoder)
        print(f"  [SAVED] Progress updated: {len(all_segment_results)}/{len(splits_to_run)} folds recorded in {progress_path}")

        # Explicitly release dataloaders, model, and intermediate tensors to prevent memory accumulation across folds
        del train_loader, val_loader, test_loader, model
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary tables
    seg_summary_text = format_cv_summary_table(all_segment_results, title=f"{k_folds}-FOLD SEGMENT-LEVEL RESULTS (Threshold = 0.50)")
    print("\n" + seg_summary_text)

    trial_summary_text = ""
    if all_trial_results:
        trial_summary_text = format_cv_summary_table(all_trial_results, title=f"{k_folds}-FOLD TRIAL-LEVEL RESULTS (Threshold = 0.50)")
        print("\n" + trial_summary_text)

    trial_cal_summary_text = ""
    if all_trial_cal_results:
        trial_cal_summary_text = format_cv_summary_table(all_trial_cal_results, title=f"{k_folds}-FOLD TRIAL-LEVEL RESULTS (Val-Calibrated)")
        print("\n" + trial_cal_summary_text)

    trial_swa_summary_text = ""
    if all_swa_trial_results:
        trial_swa_summary_text = format_cv_summary_table(all_swa_trial_results, title=f"{k_folds}-FOLD TRIAL-LEVEL RESULTS (SWA Model, Threshold = 0.50)")
        print("\n" + trial_swa_summary_text)

    # Save summary report
    summary_file = os.path.join(output_dir, 'cv_summary_report.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Subject-Independent Cross-Validation Report ({k_folds} Folds)\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(seg_summary_text + "\n\n")
        if trial_summary_text:
            f.write(trial_summary_text + "\n\n")
        if trial_cal_summary_text:
            f.write(trial_cal_summary_text + "\n\n")
        if trial_swa_summary_text:
            f.write(trial_swa_summary_text + "\n\n")
    print(f"\nSummary report saved to: {summary_file}")

    # Save final JSON
    results_json_file = os.path.join(output_dir, 'cv_results.json')
    with open(results_json_file, 'w', encoding='utf-8') as f:
        json.dump(make_json_serializable({
            'k_folds': int(k_folds),
            'segment_results': all_segment_results,
            'trial_results': all_trial_results,
            'trial_calibrated_results': all_trial_cal_results,
            'trial_swa_results': all_swa_trial_results
        }), f, indent=2, cls=NumpyJSONEncoder)
    print(f"Full results JSON saved to: {results_json_file}")

    return {
        'segment': all_segment_results,
        'trial': all_trial_results
    }


def main():
    parser = argparse.ArgumentParser(description='BiMamba-Transformer Subject-Independent Cross-Validation')
    parser.add_argument('--data_path', type=str, default='data/DEAP/deap_preprocessed.npz',
                        help='Path to preprocessed DEAP .npz')
    parser.add_argument('--k_folds', type=int, default=8,
                        help='Number of cross-validation folds (default: 8)')
    parser.add_argument('--loso', action='store_true',
                        help='Perform full 32-fold Leave-One-Subject-Out evaluation')
    parser.add_argument('--fold', type=int, default=None,
                        help='Run only a specific fold (0-indexed)')
    parser.add_argument('--output_dir', type=str, default='cv_results',
                        help='Output directory for checkpoints and reports')
    parser.add_argument('--val_metric', type=str, default='val_balanced_acc',
                        choices=['val_balanced_acc', 'val_auc', 'val_macro_f1', 'val_loss'],
                        help='Validation metric for model selection & early stopping')
    parser.add_argument('--val_subjects', type=int, default=4,
                        help='Number of validation subjects per fold (default: 4)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=80,
                        help='Number of training epochs per fold (default: 80)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--no_de', action='store_true', help='Disable DE feature branch (for ablation)')
    parser.add_argument('--no_resume', action='store_true',
                        help='Force retraining of all folds even if checkpoints exist')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of dataloader workers (default: 2 for low RAM overhead under WSL)')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    k_folds = 32 if args.loso else args.k_folds

    config = DEAPConfig(
        preprocessed_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_metric=args.val_metric,
        use_de=not args.no_de,
        device=args.device,
        num_workers=args.num_workers,
        random_state=args.seed
    )

    run_cross_validation(
        config=config,
        data_path=args.data_path,
        k_folds=k_folds,
        specific_fold=args.fold,
        output_dir=args.output_dir,
        val_subjects_count=args.val_subjects,
        resume=not args.no_resume
    )


if __name__ == '__main__':
    main()
