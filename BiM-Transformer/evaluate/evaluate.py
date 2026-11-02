"""
BiMamba-Transformer 综合评估与推理脚本

功能:
1. 模型参数统计与架构分析 (ModelProfiler)
2. 推理速度基准测试 (InferenceBenchmark)
3. 多任务 (Valence/Arousal/Dominance) 分类评估 (MultiTaskEvaluator)
4. 全面可视化: ROC曲线、混淆矩阵、PR曲线、训练历史、雷达图 (Visualizer)

Usage:
    python evaluate/evaluate.py --data_path data/DEAP/deap_preprocessed.npz --checkpoint checkpoints/best_model.pth
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, classification_report
)
from tqdm import tqdm

# Add project root to path and prefer local package "code" over stdlib module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


# ============================================================================
# 1. 模型参数统计器
# ============================================================================

class ModelProfiler:
    """模型架构与参数统计分析"""

    def __init__(self, model: nn.Module):
        self.model = model

    def count_parameters(self) -> dict:
        """统计模型总参数及各模块参数"""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable = total - trainable

        # 按模块分组统计
        module_stats = OrderedDict()
        module_map = {
            'EEG Encoder': 'eeg_encoder',
            'EOG Encoder': 'eog_encoder',
            'EMG Encoder': 'emg_encoder',
            'Cross-Modal Fusion': 'fusion',
            'Task-Aware Layer': 'task_layer',
            'Valence Head': 'valence_head',
            'Arousal Head': 'arousal_head',
            'Dominance Head': 'dominance_head',
            'Mental Health Head': 'mental_health_head',
        }

        for display_name, attr_name in module_map.items():
            if hasattr(self.model, attr_name):
                module = getattr(self.model, attr_name)
                n_params = sum(p.numel() for p in module.parameters())
                n_train = sum(p.numel() for p in module.parameters() if p.requires_grad)
                module_stats[display_name] = {
                    'total': n_params,
                    'trainable': n_train,
                    'percentage': n_params / total * 100 if total > 0 else 0
                }

        return {
            'total': total,
            'trainable': trainable,
            'non_trainable': non_trainable,
            'modules': module_stats
        }

    def estimate_memory(self) -> dict:
        """估算模型内存占用"""
        total_params = sum(p.numel() for p in self.model.parameters())
        fp32_mb = total_params * 4 / (1024 ** 2)
        fp16_mb = total_params * 2 / (1024 ** 2)
        # 训练时需要梯度 + 优化器状态 (Adam ~ 3x)
        training_fp32_mb = fp32_mb * 4  # params + grads + momentum + variance
        return {
            'fp32_inference_mb': fp32_mb,
            'fp16_inference_mb': fp16_mb,
            'fp32_training_est_mb': training_fp32_mb,
        }

    def print_model_summary(self):
        """打印完整模型摘要"""
        stats = self.count_parameters()
        memory = self.estimate_memory()

        print("\n" + "=" * 78)
        print("  📋 MODEL ARCHITECTURE SUMMARY")
        print("=" * 78)

        print(f"\n  Model: {self.model.__class__.__name__}")
        print(f"  Total Parameters:       {stats['total']:>12,}")
        print(f"  Trainable Parameters:   {stats['trainable']:>12,}")
        print(f"  Non-trainable:          {stats['non_trainable']:>12,}")

        print("\n  " + "-" * 74)
        print(f"  {'Module':<30s} {'Params':>12s} {'Trainable':>12s} {'Share':>8s}")
        print("  " + "-" * 74)
        for name, info in stats['modules'].items():
            print(f"  {name:<30s} {info['total']:>12,} {info['trainable']:>12,} {info['percentage']:>7.2f}%")
        print("  " + "-" * 74)

        print("\n  💾 Memory Estimation:")
        print(f"    FP32 Inference:  {memory['fp32_inference_mb']:>8.2f} MB")
        print(f"    FP16 Inference:  {memory['fp16_inference_mb']:>8.2f} MB")
        print(f"    FP32 Training:   {memory['fp32_training_est_mb']:>8.2f} MB (est.)")

        # 打印各层详细结构
        print("\n  📐 Layer Details:")
        print("  " + "-" * 74)
        for name, param in self.model.named_parameters():
            print(f"    {name:<55s} {str(list(param.shape)):>18s}  ({param.numel():>8,})")
        print("=" * 78)

        return stats, memory


# ============================================================================
# 2. 推理速度基准测试
# ============================================================================

class InferenceBenchmark:
    """推理速度与性能基准测试"""

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def _using_cuda(self) -> bool:
        return str(self.device).startswith('cuda') and torch.cuda.is_available()

    def _create_dummy_input(self, batch_size: int, seq_len: int = 512):
        """创建模拟输入"""
        eeg = torch.randn(batch_size, 1, 32, seq_len).to(self.device)
        eog = torch.randn(batch_size, 2, seq_len).to(self.device)
        emg = torch.randn(batch_size, 2, seq_len).to(self.device)
        return eeg, eog, emg

    @torch.no_grad()
    def measure_latency(
        self,
        batch_size: int = 1,
        seq_len: int = 512,
        n_warmup: int = 10,
        n_runs: int = 50
    ) -> dict:
        """测量推理延迟"""
        eeg, eog, emg = self._create_dummy_input(batch_size, seq_len)

        # Warmup
        for _ in range(n_warmup):
            _ = self.model(eeg, eog, emg)

        # Sync before timing
        if self._using_cuda():
            torch.cuda.synchronize()

        latencies = []
        for _ in range(n_runs):
            if self._using_cuda():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = self.model(eeg, eog, emg)
            if self._using_cuda():
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)
        return {
            'batch_size': batch_size,
            'seq_len': seq_len,
            'mean_ms': float(np.mean(latencies)),
            'std_ms': float(np.std(latencies)),
            'min_ms': float(np.min(latencies)),
            'max_ms': float(np.max(latencies)),
            'median_ms': float(np.median(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'throughput_samples_per_sec': batch_size / (np.mean(latencies) / 1000)
        }

    def benchmark_batch_sizes(
        self,
        batch_sizes: list = None,
        seq_len: int = 512,
        n_warmup: int = 5,
        n_runs: int = 30
    ) -> list:
        """测试不同 batch size 的推理速度"""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32, 64]

        results = []
        for bs in batch_sizes:
            try:
                result = self.measure_latency(bs, seq_len, n_warmup, n_runs)
                results.append(result)
            except RuntimeError as e:
                # OOM for large batch sizes
                print(f"    ⚠ Batch size {bs}: {e}")
                break
        return results

    def measure_gpu_memory(self, batch_size: int = 1, seq_len: int = 512) -> dict:
        """测量 GPU 显存使用"""
        if not self._using_cuda():
            return {'available': False}

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        eeg, eog, emg = self._create_dummy_input(batch_size, seq_len)
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)

        with torch.no_grad():
            _ = self.model(eeg, eog, emg)

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated() / (1024 ** 2)
        mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)

        return {
            'available': True,
            'before_mb': mem_before,
            'after_mb': mem_after,
            'peak_mb': mem_peak,
            'model_inference_mb': mem_after - mem_before,
            'batch_size': batch_size
        }

    def print_benchmark_report(self, seq_len: int = 512):
        """打印完整基准测试报告"""
        print("\n" + "=" * 78)
        print("  ⚡ INFERENCE SPEED BENCHMARK")
        print("=" * 78)
        print(f"  Device: {self.device}")
        if self._using_cuda():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"  GPU Memory: {total_mem:.1f} GB")
        print(f"  Sequence Length: {seq_len}")

        # Single sample latency
        print("\n  🔬 Single-Sample Latency:")
        single = self.measure_latency(batch_size=1, seq_len=seq_len)
        print(f"    Mean:   {single['mean_ms']:>8.2f} ms")
        print(f"    Std:    {single['std_ms']:>8.2f} ms")
        print(f"    Min:    {single['min_ms']:>8.2f} ms")
        print(f"    Max:    {single['max_ms']:>8.2f} ms")
        print(f"    P95:    {single['p95_ms']:>8.2f} ms")
        print(f"    Throughput: {single['throughput_samples_per_sec']:>.1f} samples/sec")

        # Batch size sweep
        print("\n  📊 Batch Size Sweep:")
        print("  " + "-" * 74)
        print(f"  {'Batch':>6s} {'Mean(ms)':>10s} {'Std(ms)':>10s} {'P95(ms)':>10s} {'Throughput':>16s}")
        print("  " + "-" * 74)
        batch_results = self.benchmark_batch_sizes(seq_len=seq_len)
        for r in batch_results:
            print(f"  {r['batch_size']:>6d} {r['mean_ms']:>10.2f} {r['std_ms']:>10.2f} "
                  f"{r['p95_ms']:>10.2f} {r['throughput_samples_per_sec']:>12.1f} s/s")
        print("  " + "-" * 74)

        # GPU memory
        if self._using_cuda():
            print("\n  🖥️ GPU Memory Usage:")
            for bs in [1, 8, 32]:
                try:
                    mem = self.measure_gpu_memory(batch_size=bs, seq_len=seq_len)
                    if mem['available']:
                        print(f"    Batch {bs:>3d}: Peak {mem['peak_mb']:>8.1f} MB")
                except RuntimeError:
                    break

        print("=" * 78)
        return batch_results


# ============================================================================
# 3. 多任务评估器
# ============================================================================

class MultiTaskEvaluator:
    """多任务 (Valence/Arousal/Dominance) 分类评估器"""

    TASK_NAMES = ['valence', 'arousal', 'dominance']
    CLASS_NAMES = ['Low', 'High']

    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def evaluate(self, dataloader, verbose: bool = True) -> dict:
        """
        在完整数据集上评估模型，同时收集 subject_id 与 trial_id 用于 Trial 级聚合。

        Returns:
            results: dict with keys per task containing predictions, probabilities, labels, and trial metadata
        """
        results = {
            task: {'preds': [], 'probs': [], 'labels': [], 'subject_ids': [], 'trial_ids': []}
            for task in self.TASK_NAMES
        }

        iterator = tqdm(dataloader, desc='Evaluating', leave=False) if verbose else dataloader
        for batch in iterator:
            eeg = batch['eeg'].to(self.device)
            eog = batch['eog'].to(self.device)
            emg = batch['emg'].to(self.device)
            de = batch.get('de', None)
            if de is not None:
                de = de.to(self.device)
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

            outputs = self.model(eeg, eog, emg, de=de)

            for task in self.TASK_NAMES:
                logits = outputs[task]
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                if valid_mask is not None and task in valid_mask:
                    mask = valid_mask[task].cpu().numpy().astype(bool)
                elif isinstance(labels, dict) and task in labels:
                    mask = (labels[task] != -1).cpu().numpy().astype(bool)
                else:
                    mask = np.ones(len(logits), dtype=bool)

                if np.any(mask):
                    results[task]['preds'].append(preds.cpu().numpy()[mask])
                    results[task]['probs'].append(probs.cpu().numpy()[mask])

                    if isinstance(labels, dict):
                        lbl = labels[task].cpu().numpy()[mask]
                    else:
                        lbl = labels.cpu().numpy()[mask]
                    results[task]['labels'].append(lbl)

                    if sub_ids is not None:
                        results[task]['subject_ids'].append(sub_ids[mask])
                    if tri_ids is not None:
                        results[task]['trial_ids'].append(tri_ids[mask])

        # Concatenate per task
        for task in self.TASK_NAMES:
            results[task]['preds'] = np.concatenate(results[task]['preds']) if len(results[task]['preds']) > 0 else np.array([])
            results[task]['probs'] = np.concatenate(results[task]['probs']) if len(results[task]['probs']) > 0 else np.empty((0, 2))
            results[task]['labels'] = np.concatenate(results[task]['labels']) if len(results[task]['labels']) > 0 else np.array([])
            results[task]['subject_ids'] = np.concatenate(results[task]['subject_ids']) if len(results[task]['subject_ids']) > 0 else None
            results[task]['trial_ids'] = np.concatenate(results[task]['trial_ids']) if len(results[task]['trial_ids']) > 0 else None

        return results

    def compute_metrics(self, results: dict, optimal_thresholds: dict = None) -> dict:
        """计算所有任务的详细指标 (包含片段级、Trial 级聚合指标以及验证集校准阈值评估)"""
        segment_metrics = {}
        segment_cal_metrics = {}
        has_calibrated = isinstance(optimal_thresholds, dict) and len(optimal_thresholds) > 0

        for task in self.TASK_NAMES:
            preds = results[task]['preds']
            probs = results[task]['probs']
            labels = results[task]['labels']
            if len(preds) > 0:
                segment_metrics[task] = compute_classification_metrics(labels, preds, probs)
                segment_metrics[task]['confusion_matrix'] = np.array(segment_metrics[task]['confusion_matrix'])
                if has_calibrated:
                    tau = float(optimal_thresholds.get(task, 0.5))
                    preds_cal = (probs[:, 1] >= tau).astype(np.int64)
                    segment_cal_metrics[task] = compute_classification_metrics(labels, preds_cal, probs)
                    segment_cal_metrics[task]['confusion_matrix'] = np.array(segment_cal_metrics[task]['confusion_matrix'])
            else:
                empty_m = {'accuracy': 0.0, 'balanced_accuracy': 0.0, 'macro_f1': 0.0, 'auc_roc': 0.5, 'confusion_matrix': np.zeros((2, 2))}
                segment_metrics[task] = empty_m
                if has_calibrated:
                    segment_cal_metrics[task] = empty_m

        segment_metrics['mean'] = {
            'accuracy': float(np.mean([segment_metrics[t]['accuracy'] for t in self.TASK_NAMES])),
            'balanced_accuracy': float(np.mean([segment_metrics[t]['balanced_accuracy'] for t in self.TASK_NAMES])),
            'majority_baseline': float(np.mean([segment_metrics[t]['majority_baseline'] for t in self.TASK_NAMES])),
            'f1_score': float(np.mean([segment_metrics[t]['macro_f1'] for t in self.TASK_NAMES])),
            'macro_f1': float(np.mean([segment_metrics[t]['macro_f1'] for t in self.TASK_NAMES])),
            'auc_roc': float(np.mean([segment_metrics[t]['auc_roc'] for t in self.TASK_NAMES])),
            'mcc': float(np.mean([segment_metrics[t]['mcc'] for t in self.TASK_NAMES])),
        }

        # Trial-level aggregation
        trial_metrics = {}
        trial_cal_metrics = {}
        has_any_trial = False
        for task in self.TASK_NAMES:
            subjs = results[task].get('subject_ids')
            tris = results[task].get('trial_ids')
            if subjs is not None and tris is not None and len(subjs) > 0:
                has_any_trial = True
                probs = results[task]['probs']
                labels = results[task]['labels']
                t_m, _ = aggregate_trial_predictions(subjs, tris, probs, labels, threshold=0.5)
                t_m['confusion_matrix'] = np.array(t_m['confusion_matrix'])
                trial_metrics[task] = t_m
                if has_calibrated:
                    tau = float(optimal_thresholds.get(task, 0.5))
                    t_m_cal, _ = aggregate_trial_predictions(subjs, tris, probs, labels, threshold=tau)
                    t_m_cal['confusion_matrix'] = np.array(t_m_cal['confusion_matrix'])
                    trial_cal_metrics[task] = t_m_cal

        if has_any_trial and len(trial_metrics) == len(self.TASK_NAMES):
            trial_metrics['mean'] = {
                'accuracy': float(np.mean([trial_metrics[t]['accuracy'] for t in self.TASK_NAMES])),
                'balanced_accuracy': float(np.mean([trial_metrics[t]['balanced_accuracy'] for t in self.TASK_NAMES])),
                'majority_baseline': float(np.mean([trial_metrics[t]['majority_baseline'] for t in self.TASK_NAMES])),
                'f1_score': float(np.mean([trial_metrics[t]['macro_f1'] for t in self.TASK_NAMES])),
                'macro_f1': float(np.mean([trial_metrics[t]['macro_f1'] for t in self.TASK_NAMES])),
                'auc_roc': float(np.mean([trial_metrics[t]['auc_roc'] for t in self.TASK_NAMES])),
                'mcc': float(np.mean([trial_metrics[t]['mcc'] for t in self.TASK_NAMES])),
            }

        all_metrics = dict(segment_metrics)
        all_metrics['segment'] = segment_metrics
        all_metrics['trial'] = trial_metrics if trial_metrics else None
        all_metrics['optimal_thresholds'] = optimal_thresholds
        all_metrics['segment_calibrated'] = segment_cal_metrics if has_calibrated else None
        all_metrics['trial_calibrated'] = trial_cal_metrics if (has_calibrated and trial_cal_metrics) else None

        return all_metrics

    def print_evaluation_report(self, metrics: dict):
        """打印详细评估报告"""
        print("\n" + format_metrics_table(metrics.get('segment', metrics), title="SEGMENT-LEVEL EVALUATION REPORT (Threshold = 0.50)"))

        if metrics.get('trial'):
            print("\n" + format_metrics_table(metrics['trial'], title="TRIAL-LEVEL AGGREGATED EVALUATION REPORT (Threshold = 0.50)"))

        opt_thresh = metrics.get('optimal_thresholds')
        if opt_thresh and metrics.get('segment_calibrated'):
            thresh_str = ", ".join([f"{t[0].upper()}={opt_thresh.get(t, 0.5):.2f}" for t in self.TASK_NAMES])
            print("\n" + format_metrics_table(
                metrics['segment_calibrated'],
                title=f"SEGMENT-LEVEL EVALUATION REPORT (Val-Calibrated: {thresh_str})"
            ))
            if metrics.get('trial_calibrated'):
                print("\n" + format_metrics_table(
                    metrics['trial_calibrated'],
                    title=f"TRIAL-LEVEL AGGREGATED EVALUATION REPORT (Val-Calibrated: {thresh_str})"
                ))

        # 各任务详细混淆矩阵
        print("\n  🎯 Confusion Matrix & Class Distribution (Segment-level):")
        for task in self.TASK_NAMES:
            m = metrics[task]
            cm = m['confusion_matrix']
            print(f"    [{task.upper()}] (n={m['n_samples']}, Maj-Base: {m['majority_baseline']:.4f})")
            print(f"      Spec(Low): {m['specificity']:.4f} | Sens(High): {m['sensitivity']:.4f}")
            print(f"      Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]} | FN={cm[1, 0]}, TP={cm[1, 1]}")
        print("=" * 86)

    @torch.no_grad()
    def predict_single(self, eeg, eog, emg, de=None) -> dict:
        """单样本推理"""
        if isinstance(eeg, np.ndarray):
            eeg = torch.from_numpy(eeg).float()
        if isinstance(eog, np.ndarray):
            eog = torch.from_numpy(eog).float()
        if isinstance(emg, np.ndarray):
            emg = torch.from_numpy(emg).float()
        if de is not None and isinstance(de, np.ndarray):
            de = torch.from_numpy(de).float()

        # Add batch dimension if needed
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(0)
        elif eeg.dim() == 3 and eeg.shape[0] != 1:
            eeg = eeg.unsqueeze(0)
        if eog.dim() == 1:
            eog = eog.unsqueeze(0)
        if emg.dim() == 1:
            emg = emg.unsqueeze(0)
        if eog.dim() == 2:
            eog = eog.unsqueeze(0)
        if emg.dim() == 2:
            emg = emg.unsqueeze(0)
        if de is not None:
            if de.dim() == 1:
                de = de.unsqueeze(0)
            de = de.to(self.device)

        eeg = eeg.to(self.device)
        eog = eog.to(self.device)
        emg = emg.to(self.device)

        outputs = self.model(eeg, eog, emg, de=de)

        result = {}
        for task in self.TASK_NAMES:
            probs = torch.softmax(outputs[task], dim=1).squeeze().cpu().numpy()
            pred = int(probs.argmax())
            result[task] = {
                'prediction': pred,
                'label': self.CLASS_NAMES[pred],
                'probabilities': probs.tolist(),
                'confidence': float(probs.max())
            }
        return result


# ============================================================================
# 4. 可视化模块
# ============================================================================

class Visualizer:
    """评估结果可视化"""

    TASK_NAMES = ['valence', 'arousal', 'dominance']
    TASK_COLORS = {'valence': '#FF6B6B', 'arousal': '#4ECDC4', 'dominance': '#45B7D1'}
    TASK_LABELS = {'valence': 'Valence', 'arousal': 'Arousal', 'dominance': 'Dominance'}

    def __init__(self, output_dir: str = 'evaluation_results'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 设置全局绘图风格
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })
        sns.set_style("whitegrid")

    def plot_roc_curves(self, results: dict, metrics: dict):
        """绘制三任务 ROC 曲线对比图"""
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        # 各任务单独 ROC
        for idx, task in enumerate(self.TASK_NAMES):
            labels = results[task]['labels']
            probs = results[task]['probs'][:, 1]
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)

            ax = axes[idx]
            ax.plot(fpr, tpr, color=self.TASK_COLORS[task], lw=2.5,
                    label=f'{self.TASK_LABELS[task]} (AUC={roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random')
            ax.fill_between(fpr, tpr, alpha=0.15, color=self.TASK_COLORS[task])
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{self.TASK_LABELS[task]} ROC Curve')
            ax.legend(loc='lower right', fontsize=10)
            ax.grid(True, alpha=0.3)

        # 合并 ROC 对比图
        ax = axes[3]
        for task in self.TASK_NAMES:
            labels = results[task]['labels']
            probs = results[task]['probs'][:, 1]
            fpr, tpr, _ = roc_curve(labels, probs)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=self.TASK_COLORS[task], lw=2.5,
                    label=f'{self.TASK_LABELS[task]} (AUC={roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Combined ROC Curves')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  ✅ ROC curves saved to {save_path}")

    def plot_confusion_matrices(self, metrics: dict):
        """绘制三任务混淆矩阵"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        class_names = ['Low', 'High']

        for idx, task in enumerate(self.TASK_NAMES):
            cm = metrics[task]['confusion_matrix']
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            ax = axes[idx]
            sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'label': 'Count', 'shrink': 0.8})

            # 添加数值和百分比
            for i in range(2):
                for j in range(2):
                    ax.text(j + 0.5, i + 0.35, f'{cm[i, j]}',
                            ha='center', va='center', fontsize=16, fontweight='bold',
                            color='white' if cm_norm[i, j] > 0.5 else 'black')
                    ax.text(j + 0.5, i + 0.65, f'({cm_norm[i, j]:.1%})',
                            ha='center', va='center', fontsize=11,
                            color='white' if cm_norm[i, j] > 0.5 else 'gray')

            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title(f'{self.TASK_LABELS[task]} Confusion Matrix\n'
                         f'Acc={metrics[task]["accuracy"]:.4f}',
                         fontsize=13)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'confusion_matrices.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  ✅ Confusion matrices saved to {save_path}")

    def plot_pr_curves(self, results: dict):
        """绘制 Precision-Recall 曲线"""
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))

        for idx, task in enumerate(self.TASK_NAMES):
            labels = results[task]['labels']
            probs = results[task]['probs'][:, 1]
            precision, recall, _ = precision_recall_curve(labels, probs)
            ap = average_precision_score(labels, probs)

            ax = axes[idx]
            ax.plot(recall, precision, color=self.TASK_COLORS[task], lw=2.5,
                    label=f'{self.TASK_LABELS[task]} (AP={ap:.4f})')
            ax.fill_between(recall, precision, alpha=0.15, color=self.TASK_COLORS[task])
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{self.TASK_LABELS[task]} PR Curve')
            ax.legend(loc='lower left', fontsize=10)
            ax.grid(True, alpha=0.3)

        # Combined
        ax = axes[3]
        for task in self.TASK_NAMES:
            labels = results[task]['labels']
            probs = results[task]['probs'][:, 1]
            precision, recall, _ = precision_recall_curve(labels, probs)
            ap = average_precision_score(labels, probs)
            ax.plot(recall, precision, color=self.TASK_COLORS[task], lw=2.5,
                    label=f'{self.TASK_LABELS[task]} (AP={ap:.4f})')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Combined PR Curves')
        ax.legend(loc='lower left', fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'pr_curves.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  ✅ PR curves saved to {save_path}")

    def plot_metrics_radar(self, metrics: dict):
        """绘制各任务指标雷达图"""
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC', 'MCC+0.5']
        metric_keys = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'mcc']

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]  # 闭合

        for task in self.TASK_NAMES:
            values = []
            for key in metric_keys:
                v = metrics[task][key]
                # MCC 范围 [-1, 1], 映射到 [0, 1]
                if key == 'mcc':
                    v = (v + 1) / 2
                values.append(v)
            values += values[:1]

            ax.plot(angles, values, 'o-', lw=2, color=self.TASK_COLORS[task],
                    label=self.TASK_LABELS[task], markersize=6)
            ax.fill(angles, values, alpha=0.1, color=self.TASK_COLORS[task])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.set_title('Multi-Task Metrics Overview', fontsize=14, pad=20)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'metrics_radar.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  ✅ Metrics radar saved to {save_path}")

    def plot_per_class_accuracy(self, metrics: dict):
        """绘制每类别准确率对比柱状图"""
        fig, ax = plt.subplots(figsize=(10, 6))

        x = np.arange(len(self.TASK_NAMES))
        width = 0.3

        low_accs = [metrics[t]['class_accuracy']['class_0'] for t in self.TASK_NAMES]
        high_accs = [metrics[t]['class_accuracy']['class_1'] for t in self.TASK_NAMES]
        overall_accs = [metrics[t]['accuracy'] for t in self.TASK_NAMES]

        bars1 = ax.bar(x - width, low_accs, width, label='Low Class',
                       color='#FF6B6B', alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x, high_accs, width, label='High Class',
                       color='#4ECDC4', alpha=0.85, edgecolor='white')
        bars3 = ax.bar(x + width, overall_accs, width, label='Overall',
                       color='#45B7D1', alpha=0.85, edgecolor='white')

        # 添加数据标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=10)

        ax.set_xticks(x)
        ax.set_xticklabels([self.TASK_LABELS[t] for t in self.TASK_NAMES], fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Class Accuracy Comparison', fontsize=14)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.15)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'per_class_accuracy.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  ✅ Per-class accuracy saved to {save_path}")

    def plot_training_history(self, history_path: str):
        """从 training_history.json 绘制训练历史曲线"""
        if not os.path.exists(history_path):
            print(f"  ⚠ Training history not found: {history_path}")
            return

        with open(history_path, 'r') as f:
            history = json.load(f)

        epochs = range(1, len(history['train_loss']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Loss Curves
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], 'b-', lw=2, label='Train Loss', marker='o', markersize=3)
        ax.plot(epochs, history['val_loss'], 'r-', lw=2, label='Val Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Validation Accuracy per Task
        ax = axes[0, 1]
        task_keys = {
            'val_valence_acc': ('Valence', '#FF6B6B'),
            'val_arousal_acc': ('Arousal', '#4ECDC4'),
            'val_dominance_acc': ('Dominance', '#45B7D1'),
        }
        for key, (label, color) in task_keys.items():
            if key in history:
                ax.plot(epochs, history[key], lw=2, label=label, color=color,
                        marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy per Task')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Mean Accuracy
        ax = axes[1, 0]
        if all(k in history for k in ['val_valence_acc', 'val_arousal_acc', 'val_dominance_acc']):
            mean_acc = [(v + a + d) / 3 for v, a, d in
                        zip(history['val_valence_acc'],
                            history['val_arousal_acc'],
                            history['val_dominance_acc'])]
            ax.plot(epochs, mean_acc, 'g-', lw=2.5, label='Mean Val Accuracy',
                    marker='D', markersize=4)
            best_epoch = np.argmax(mean_acc) + 1
            best_acc = max(mean_acc)
            ax.axvline(x=best_epoch, color='orange', linestyle='--', alpha=0.7,
                       label=f'Best Epoch {best_epoch} ({best_acc:.4f})')
            ax.scatter([best_epoch], [best_acc], color='red', s=100, zorder=5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Mean Validation Accuracy')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Learning Rate
        ax = axes[1, 1]
        if 'learning_rate' in history:
            ax.plot(epochs, history['learning_rate'], 'purple', lw=2, marker='o', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-4, -4))

        plt.suptitle('Training History', fontsize=16, y=1.01)
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'training_history.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  ✅ Training history saved to {save_path}")

    def plot_prediction_distribution(self, results: dict):
        """绘制各任务预测概率分布"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, task in enumerate(self.TASK_NAMES):
            ax = axes[idx]
            probs = results[task]['probs'][:, 1]
            labels = results[task]['labels']

            ax.hist(probs[labels == 0], bins=50, alpha=0.6, color='#FF6B6B',
                    label='True: Low', density=True, edgecolor='white')
            ax.hist(probs[labels == 1], bins=50, alpha=0.6, color='#4ECDC4',
                    label='True: High', density=True, edgecolor='white')
            ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5, alpha=0.7,
                       label='Decision Boundary')
            ax.set_xlabel('Predicted Probability (High)')
            ax.set_ylabel('Density')
            ax.set_title(f'{self.TASK_LABELS[task]} Prediction Distribution')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, 'prediction_distribution.png')
        plt.savefig(save_path)
        plt.close()
        print(f"  ✅ Prediction distribution saved to {save_path}")

    def generate_all(self, results: dict, metrics: dict, history_path: str = None):
        """生成所有可视化"""
        print("\n  📈 Generating visualizations...")
        self.plot_roc_curves(results, metrics)
        self.plot_confusion_matrices(metrics)
        self.plot_pr_curves(results)
        self.plot_metrics_radar(metrics)
        self.plot_per_class_accuracy(metrics)
        self.plot_prediction_distribution(results)
        if history_path:
            self.plot_training_history(history_path)


# ============================================================================
# 5. 报告保存
# ============================================================================

def save_evaluation_report(metrics: dict, output_dir: str, args=None):
    """保存评估结果到文件 (包含片段级与 Trial 级)"""
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    json_path = os.path.join(output_dir, 'evaluation_metrics.json')

    lines = []
    lines.append("=" * 86)
    lines.append("BIMAMBA-TRANSFORMER COMPREHENSIVE EVALUATION REPORT")
    lines.append("=" * 86)
    lines.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if args:
        lines.append(f"Checkpoint: {args.checkpoint}")
        lines.append(f"Data Path: {args.data_path}")
    lines.append("")

    seg_metrics = metrics.get('segment', metrics)
    lines.append(format_metrics_table(seg_metrics, title="SEGMENT-LEVEL RESULTS"))
    lines.append("")

    for task in MultiTaskEvaluator.TASK_NAMES:
        m = seg_metrics[task]
        lines.append(f"--- {task.upper()} (Segment-Level, n={m['n_samples']}) ---")
        lines.append(f"  Accuracy:          {m['accuracy']:.4f}")
        lines.append(f"  Balanced Accuracy: {m['balanced_accuracy']:.4f}")
        lines.append(f"  Majority Baseline: {m['majority_baseline']:.4f}")
        lines.append(f"  Macro F1:          {m['macro_f1']:.4f}")
        lines.append(f"  AUC-ROC:           {m['auc_roc']:.4f}")
        lines.append(f"  AUC-PR:            {m['auc_pr']:.4f}")
        lines.append(f"  MCC:               {m['mcc']:.4f}")
        lines.append(f"  Specificity (Low): {m['specificity']:.4f}")
        lines.append(f"  Sensitivity (High):{m['sensitivity']:.4f}")
        lines.append("")

    trial_metrics = metrics.get('trial')
    if trial_metrics:
        lines.append(format_metrics_table(trial_metrics, title="TRIAL-LEVEL AGGREGATED RESULTS"))
        lines.append("")
        for task in MultiTaskEvaluator.TASK_NAMES:
            m = trial_metrics[task]
            lines.append(f"--- {task.upper()} (Trial-Level, n={m['n_samples']}) ---")
            lines.append(f"  Accuracy:          {m['accuracy']:.4f}")
            lines.append(f"  Balanced Accuracy: {m['balanced_accuracy']:.4f}")
            lines.append(f"  Majority Baseline: {m['majority_baseline']:.4f}")
            lines.append(f"  Macro F1:          {m['macro_f1']:.4f}")
            lines.append(f"  AUC-ROC:           {m['auc_roc']:.4f}")
            lines.append(f"  MCC:               {m['mcc']:.4f}")
            lines.append(f"  Specificity (Low): {m['specificity']:.4f}")
            lines.append(f"  Sensitivity (High):{m['sensitivity']:.4f}")
            lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  ✅ Text report saved to {report_path}")

    # JSON 报告
    def serialize_metrics(m_dict):
        serializable = {}
        for k, v in m_dict.items():
            if isinstance(v, dict):
                serializable[k] = serialize_metrics(v)
            elif isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, (np.floating, float)):
                serializable[k] = float(v)
            elif isinstance(v, (np.integer, int)):
                serializable[k] = int(v)
            else:
                serializable[k] = v
        return serializable

    json_metrics = serialize_metrics(metrics)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_metrics, f, indent=2, ensure_ascii=False)
    print(f"  ✅ JSON metrics saved to {json_path}")


def load_model_weights(model: nn.Module, checkpoint: dict):
    """Load state dict from different checkpoint formats and handle DataParallel prefix."""
    state_dict = checkpoint.get('model_state_dict')
    if state_dict is None:
        state_dict = checkpoint.get('state_dict')

    if state_dict is None:
        if isinstance(checkpoint, dict) and checkpoint and all(
            torch.is_tensor(v) for v in checkpoint.values()
        ):
            state_dict = checkpoint
        else:
            raise ValueError("Checkpoint does not contain a valid state_dict.")

    try:
        model.load_state_dict(state_dict, strict=False)
    except RuntimeError:
        stripped_state = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                stripped_state[key[7:]] = value
            else:
                stripped_state[key] = value
        model.load_state_dict(stripped_state, strict=False)


# ============================================================================
# 6. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='BiMamba-Transformer Comprehensive Evaluation & Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full evaluation
  python evaluate/evaluate.py --data_path data/DEAP/deap_preprocessed.npz --checkpoint checkpoints/best_model.pth

  # Speed benchmark only (no data needed)
  python evaluate/evaluate.py --benchmark_only --checkpoint checkpoints/best_model.pth

  # Model profiling only
  python evaluate/evaluate.py --profile_only
        """
    )

    # Data & Model
    parser.add_argument('--data_path', type=str, default='data/DEAP/deap_preprocessed.npz',
                        help='Path to preprocessed DEAP data (.npz)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--history_path', type=str, default='checkpoints/training_history.json',
                        help='Path to training history JSON')

    # Model config (override defaults)
    parser.add_argument('--d_model', type=int, default=None, help='Model dimension (auto from config)')
    parser.add_argument('--hidden_dim', type=int, default=None, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=None, help='Dropout rate')
    parser.add_argument('--eeg_layers', type=int, default=None, help='EEG encoder layers')
    parser.add_argument('--peripheral_layers', type=int, default=None, help='Peripheral encoder layers')
    parser.add_argument('--n_tokens', type=int, default=None, help='Number of temporal tokens')

    # Evaluation config
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--subject_independent', dest='subject_independent', action='store_true', default=True,
                        help='Use subject-independent split')
    parser.add_argument('--subject_dependent', dest='subject_independent', action='store_false',
                        help='Use random split across all subjects')
    parser.add_argument('--test_subjects', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    # Output
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save all outputs')

    # Mode flags
    parser.add_argument('--profile_only', action='store_true',
                        help='Only show model profile (no data needed)')
    parser.add_argument('--benchmark_only', action='store_true',
                        help='Only run speed benchmark (no data needed)')
    parser.add_argument('--skip_benchmark', action='store_true',
                        help='Skip speed benchmark')
    parser.add_argument('--skip_visualize', action='store_true',
                        help='Skip visualization generation')
    parser.add_argument('--visualize_samples', type=int, default=5,
                        help='Number of single sample predictions to show')

    args = parser.parse_args()

    # Setup
    requested_cuda = args.device.startswith('cuda')
    if requested_cuda and not torch.cuda.is_available():
        device = 'cpu'
        print("⚠ CUDA not available, falling back to CPU")
    else:
        device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # Load config from checkpoint if available, else use defaults
    config = DEAPConfig()

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_config = checkpoint.get('config', {})
        if saved_config:
            print(f"  📂 Loading config from checkpoint...")
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
    else:
        checkpoint = None

    # Override with command-line args
    if args.d_model is not None:
        config.d_model = args.d_model
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.dropout is not None:
        config.dropout = args.dropout
    if args.eeg_layers is not None:
        config.eeg_layers = args.eeg_layers
    if args.peripheral_layers is not None:
        config.peripheral_layers = args.peripheral_layers
    if args.n_tokens is not None:
        config.n_tokens = args.n_tokens

    # ========== Create Model ==========
    print("\n" + "=" * 78)
    print("  🧠 BiMamba-Transformer Comprehensive Evaluation")
    print("=" * 78)
    print(f"  Device: {device}")
    print(f"  Config: d_model={config.d_model}, hidden_dim={config.hidden_dim}, "
          f"dropout={config.dropout}")
    print(f"  EEG Layers: {config.eeg_layers}, Peripheral Layers: {config.peripheral_layers}, "
          f"Fusion Layers: {getattr(config, 'fusion_layers', 1)}")

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

    # Load weights
    if checkpoint is not None:
        load_model_weights(model, checkpoint)
        epoch = checkpoint.get('epoch', '?')
        best_acc = checkpoint.get('best_val_acc', '?')
        print(f"  ✅ Loaded checkpoint from epoch {epoch} (best val acc: {best_acc})")
    else:
        print("  ⚠ No checkpoint loaded — using random weights")

    model.eval()

    # ========== 1. Model Profiling ==========
    profiler = ModelProfiler(model)
    profiler.print_model_summary()

    if args.profile_only:
        print("\n  Done (profile_only mode).")
        return

    # ========== 2. Inference Benchmark ==========
    if not args.skip_benchmark:
        benchmark = InferenceBenchmark(model, device)
        benchmark.print_benchmark_report(seq_len=512)

    if args.benchmark_only:
        print("\n  Done (benchmark_only mode).")
        return

    # ========== 3. Dataset Evaluation ==========
    if not os.path.exists(args.data_path):
        print(f"\n  ❌ Data file not found: {args.data_path}")
        print("  Skipping dataset evaluation. Run preprocessing first.")
        return

    print("\n  Loading dataset...")
    _, _, test_loader = create_deap_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        subject_independent=args.subject_independent,
        test_subjects=args.test_subjects,
        random_state=42,
        num_workers=args.num_workers,
        filter_ambiguous=True,
        waveform_norm=getattr(config, 'norm_level', 'trial'),
        use_de=getattr(config, 'use_de', True)
    )

    evaluator = MultiTaskEvaluator(model, device)

    print("\n  Running evaluation on test set...")
    results = evaluator.evaluate(test_loader, verbose=True)
    opt_thresholds = checkpoint.get('optimal_thresholds', None) if checkpoint is not None else None
    metrics = evaluator.compute_metrics(results, optimal_thresholds=opt_thresholds)
    evaluator.print_evaluation_report(metrics)

    # ========== 4. Detailed Classification Reports ==========
    print("\n  📋 Detailed Classification Reports:")
    for task in MultiTaskEvaluator.TASK_NAMES:
        print(f"\n  --- {task.upper()} ---")
        report = classification_report(
            results[task]['labels'], results[task]['preds'],
            target_names=['Low', 'High'], digits=4, zero_division=0
        )
        for line in report.split('\n'):
            print(f"    {line}")

    # ========== 5. Single Sample Inference Demo ==========
    if args.visualize_samples > 0:
        print(f"\n  🔍 Single Sample Inference Demo ({args.visualize_samples} samples):")
        print("  " + "-" * 74)
        sample_iter = iter(test_loader)
        try:
            batch = next(sample_iter)
        except StopIteration:
            batch = None

        if batch is None:
            print("    No test samples available after filtering.")
            print("  " + "-" * 74)
        else:
            for idx in range(min(args.visualize_samples, batch['eeg'].shape[0])):
                eeg_sample = batch['eeg'][idx]
                eog_sample = batch['eog'][idx]
                emg_sample = batch['emg'][idx]
                de_sample = batch['de'][idx] if 'de' in batch and batch['de'] is not None else None

                if isinstance(batch['labels'], dict):
                    true_labels = {t: batch['labels'][t][idx].item()
                                   for t in MultiTaskEvaluator.TASK_NAMES}
                else:
                    true_labels = {t: batch['labels'][idx].item()
                                   for t in MultiTaskEvaluator.TASK_NAMES}

                pred = evaluator.predict_single(eeg_sample, eog_sample, emg_sample, de=de_sample)

                line_parts = []
                for task in MultiTaskEvaluator.TASK_NAMES:
                    p = pred[task]
                    true_val = true_labels[task]
                    correct = '✓' if p['prediction'] == true_val else '✗'
                    line_parts.append(
                        f"{task[0].upper()}: {p['label']}({p['confidence']:.2f})"
                        f"[true={true_val}]{correct}"
                    )
                print(f"    Sample {idx + 1}: {'  |  '.join(line_parts)}")
            print("  " + "-" * 74)

    # ========== 6. Visualizations ==========
    if not args.skip_visualize:
        visualizer = Visualizer(args.output_dir)
        history_path = args.history_path if os.path.exists(args.history_path) else None
        visualizer.generate_all(results, metrics, history_path)

    # ========== 7. Save Reports ==========
    save_evaluation_report(metrics, args.output_dir, args)

    print(f"\n  🎉 Evaluation complete! All results saved to: {args.output_dir}/")
    print("=" * 78)


if __name__ == '__main__':
    main()
