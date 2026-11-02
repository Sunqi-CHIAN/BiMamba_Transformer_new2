"""
Evaluation metrics and trial aggregation utilities for BiMamba-Transformer.

This module provides standardized, unbiased metric calculations for EEG emotion recognition:
- Balanced Accuracy (macro recall) to eliminate majority class bias
- Macro-F1 score (unweighted mean of class-0 and class-1 F1)
- ROC-AUC and PR-AUC
- Majority Class Baseline
- Matthews Correlation Coefficient (MCC)
- Trial-level aggregation (averaging segment probabilities per trial)
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None
) -> Dict[str, Union[float, int, dict]]:
    """
    Compute comprehensive and unbiased classification metrics for binary classification.

    Args:
        y_true: True labels (0 or 1), shape (N,)
        y_pred: Predicted class labels (0 or 1), shape (N,)
        y_prob: Predicted probability for positive class (class 1) or both classes (N, 2), shape (N,) or (N, 2)

    Returns:
        Dictionary of metrics including balanced accuracy, macro-F1, AUC, and majority baseline.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    if y_prob is not None:
        if y_prob.ndim == 2:
            prob_pos = y_prob[:, 1]
        else:
            prob_pos = y_prob
    else:
        prob_pos = None

    n_samples = len(y_true)
    if n_samples == 0:
        return {}

    # Basic accuracy & balanced accuracy
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    # Majority class baseline
    count_0 = int(np.sum(y_true == 0))
    count_1 = int(np.sum(y_true == 1))
    maj_baseline = float(max(count_0, count_1) / n_samples)

    # Macro Precision, Recall, F1
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    # Per-class metrics
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # recall class 1
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # recall class 0

    # Matthews Correlation Coefficient (computed from the fixed 2x2 confusion matrix so
    # single-class subsets, e.g. a test subject with only Low labels, do not trigger sklearn warnings)
    mcc_den = float(np.sqrt(float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)))
    mcc = float((float(tp) * float(tn) - float(fp) * float(fn)) / mcc_den) if mcc_den > 0 else 0.0

    # ROC-AUC & PR-AUC
    auc_roc = 0.0
    auc_pr = 0.0
    if prob_pos is not None:
        try:
            # Need both classes present to compute AUC
            if len(np.unique(y_true)) > 1:
                auc_roc = float(roc_auc_score(y_true, prob_pos))
                auc_pr = float(average_precision_score(y_true, prob_pos))
            else:
                auc_roc = 0.5
                auc_pr = maj_baseline
        except Exception:
            auc_roc = 0.0
            auc_pr = 0.0

    return {
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'majority_baseline': maj_baseline,
        'macro_f1': float(f1_macro),
        'f1_score': float(f1_macro),  # Alias for compatibility with standard reporting
        'f1_class_0': float(per_class_f1[0]),
        'f1_class_1': float(per_class_f1[1]),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'mcc': mcc,
        'confusion_matrix': cm.tolist(),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'class_accuracy': {
            'class_0': specificity,
            'class_1': sensitivity
        },
        'class_distribution': {
            'class_0': count_0,
            'class_1': count_1
        },
        'n_samples': n_samples
    }


def find_optimal_threshold(
    y_true: Union[np.ndarray, torch.Tensor],
    y_prob: Union[np.ndarray, torch.Tensor],
    low: float = 0.35,
    high: float = 0.65,
    step: float = 0.01,
    center_penalty: float = 0.02
) -> float:
    """
    Find the probability threshold in [low, high] that maximizes Balanced Accuracy
    on validation predictions, with a gentle proximity penalty toward 0.50 so flat
    validation plateaus do not drift to extreme thresholds.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_true = np.asarray(y_true, dtype=np.int64)
    prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else np.asarray(y_prob, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        return 0.50

    pos_mask = (y_true == 1)
    neg_mask = (y_true == 0)
    n_pos = max(int(pos_mask.sum()), 1)
    n_neg = max(int(neg_mask.sum()), 1)

    best_tau = 0.50
    best_obj = -1.0

    for tau in np.arange(low, high + 1e-6, step):
        preds = (prob_pos >= tau).astype(np.int64)
        sens = float((preds[pos_mask] == 1).sum()) / n_pos
        spec = float((preds[neg_mask] == 0).sum()) / n_neg
        bal = 0.5 * (sens + spec)
        obj = bal - center_penalty * abs(tau - 0.50)
        if obj > best_obj + 1e-6 or (
            abs(obj - best_obj) <= 1e-6 and abs(tau - 0.50) < abs(best_tau - 0.50)
        ):
            best_obj = obj
            best_tau = float(round(tau, 2))

    return best_tau


def aggregate_trial_predictions(
    subject_ids: Union[np.ndarray, List[int]],
    trial_ids: Union[np.ndarray, List[int]],
    probs: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor],
    threshold: float = 0.5
) -> Tuple[Dict[str, Union[float, int, dict]], Dict[str, np.ndarray]]:
    """
    Aggregate segment-level predictions to trial-level predictions by averaging
    probabilities across segments belonging to the same trial.

    Args:
        subject_ids: Subject/Participant IDs for each segment, shape (N,)
        trial_ids: Trial IDs for each segment, shape (N,)
        probs: Predicted probability distributions, shape (N, 2) or (N,)
        labels: Ground truth labels for each segment, shape (N,)
        threshold: Decision threshold on positive-class probability (default: 0.5)

    Returns:
        trial_metrics: Metrics computed on trial-level aggregated predictions
        trial_data: Dict with trial-level 'preds', 'probs', 'labels', 'trial_keys'
    """
    if isinstance(subject_ids, torch.Tensor):
        subject_ids = subject_ids.detach().cpu().numpy()
    else:
        subject_ids = np.asarray(subject_ids)

    if isinstance(trial_ids, torch.Tensor):
        trial_ids = trial_ids.detach().cpu().numpy()
    else:
        trial_ids = np.asarray(trial_ids)

    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=-1)

    # Group segments by unique trial key: (subject_id, trial_id)
    unique_trials = list(dict.fromkeys(zip(subject_ids.tolist(), trial_ids.tolist())))

    trial_keys = []
    trial_probs_list = []
    trial_labels_list = []

    for subj, trial in unique_trials:
        mask = (subject_ids == subj) & (trial_ids == trial)
        trial_prob = probs[mask].mean(axis=0)  # Average segment probabilities
        # Ground truth should be constant for the trial
        trial_label = labels[mask][0]

        trial_keys.append(f"s{subj:02d}_t{trial:02d}")
        trial_probs_list.append(trial_prob)
        trial_labels_list.append(trial_label)

    trial_probs = np.array(trial_probs_list)
    trial_labels = np.array(trial_labels_list, dtype=np.int64)
    trial_preds = (trial_probs[:, 1] >= threshold).astype(np.int64)

    trial_metrics = compute_classification_metrics(trial_labels, trial_preds, trial_probs)

    trial_data = {
        'trial_keys': trial_keys,
        'trial_preds': trial_preds,
        'trial_probs': trial_probs,
        'trial_labels': trial_labels,
        'preds': trial_preds,
        'probs': trial_probs,
        'labels': trial_labels
    }

    return trial_metrics, trial_data


def OrderedDict_unique(seq: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Return unique elements preserving original order."""
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def format_metrics_table(
    metrics_per_task: Dict[str, dict],
    title: str = "MULTI-TASK EVALUATION SUMMARY"
) -> str:
    """
    Format evaluation metrics into a clean, comprehensive text table.
    """
    lines = []
    lines.append("=" * 86)
    lines.append(f"  {title}")
    lines.append("=" * 86)
    header = (
        f"  {'Task':<12s} {'Acc':>7s} {'Bal-Acc':>8s} {'Maj-Base':>9s} "
        f"{'Macro-F1':>9s} {'AUC':>7s} {'MCC':>7s} {'Spec(C0)':>9s} {'Sens(C1)':>9s}"
    )
    lines.append(header)
    lines.append("  " + "-" * 82)

    task_keys = [t for t in ['valence', 'arousal', 'dominance'] if t in metrics_per_task]

    for task in task_keys:
        m = metrics_per_task[task]
        lines.append(
            f"  {task.capitalize():<12s} "
            f"{m.get('accuracy', 0.0):>7.4f} "
            f"{m.get('balanced_accuracy', 0.0):>8.4f} "
            f"{m.get('majority_baseline', 0.0):>9.4f} "
            f"{m.get('macro_f1', 0.0):>9.4f} "
            f"{m.get('auc_roc', 0.0):>7.4f} "
            f"{m.get('mcc', 0.0):>7.4f} "
            f"{m.get('specificity', 0.0):>9.4f} "
            f"{m.get('sensitivity', 0.0):>9.4f}"
        )

    lines.append("  " + "-" * 82)

    # Mean row
    if task_keys:
        mean_acc = np.mean([metrics_per_task[t].get('accuracy', 0.0) for t in task_keys])
        mean_bal = np.mean([metrics_per_task[t].get('balanced_accuracy', 0.0) for t in task_keys])
        mean_maj = np.mean([metrics_per_task[t].get('majority_baseline', 0.0) for t in task_keys])
        mean_f1 = np.mean([metrics_per_task[t].get('macro_f1', 0.0) for t in task_keys])
        mean_auc = np.mean([metrics_per_task[t].get('auc_roc', 0.0) for t in task_keys])
        mean_mcc = np.mean([metrics_per_task[t].get('mcc', 0.0) for t in task_keys])
        mean_spec = np.mean([metrics_per_task[t].get('specificity', 0.0) for t in task_keys])
        mean_sens = np.mean([metrics_per_task[t].get('sensitivity', 0.0) for t in task_keys])

        lines.append(
            f"  {'Mean':<12s} "
            f"{mean_acc:>7.4f} "
            f"{mean_bal:>8.4f} "
            f"{mean_maj:>9.4f} "
            f"{mean_f1:>9.4f} "
            f"{mean_auc:>7.4f} "
            f"{mean_mcc:>7.4f} "
            f"{mean_spec:>9.4f} "
            f"{mean_sens:>9.4f}"
        )
    lines.append("=" * 86)
    return "\n".join(lines)
