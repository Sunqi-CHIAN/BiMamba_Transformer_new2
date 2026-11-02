"""
Check class balance for Valence, Arousal, and Dominance labels in DEAP dataset.

Usage:
    python code/utils/check_class_balance.py --data_path data/deap_preprocessed.npz
"""
import argparse
import numpy as np
from collections import Counter


def analyze_class_balance(data_path: str):
    """Analyze class balance for all emotion dimensions."""
    print("=" * 60)
    print("Class Balance Analysis for DEAP Dataset")
    print("=" * 60)
    
    # Load preprocessed data
    print(f"\nLoading data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    
    # Get labels
    valence = data['valence']
    arousal = data['arousal']
    dominance = data['dominance']
    
    print(f"Total samples: {len(valence)}")
    print()
    
    # Analyze each dimension
    for name, labels in [('Valence', valence), ('Arousal', arousal), ('Dominance', dominance)]:
        print("-" * 40)
        print(f"{name} Distribution:")
        print("-" * 40)
        
        counter = Counter(labels)
        total = len(labels)
        
        for class_id in sorted(counter.keys()):
            count = counter[class_id]
            percentage = count / total * 100
            print(f"  Class {class_id}: {count:5d} samples ({percentage:5.2f}%)")
        
        # Calculate imbalance ratio
        counts = list(counter.values())
        imbalance_ratio = max(counts) / min(counts)
        
        print(f"\n  Imbalance Ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 1.5:
            print(f"  [!] WARNING: Significant class imbalance detected!")
        elif imbalance_ratio > 1.2:
            print(f"  [~] Moderate imbalance - consider class weighting")
        else:
            print(f"  [OK] Relatively balanced")
        
        print()
    
    # Cross-correlation analysis
    print("-" * 40)
    print("Cross-Dimension Correlation:")
    print("-" * 40)
    
    print(f"  Valence-Arousal agreement: {(valence == arousal).mean()*100:.2f}%")
    print(f"  Valence-Dominance agreement: {(valence == dominance).mean()*100:.2f}%")
    print(f"  Arousal-Dominance agreement: {(arousal == dominance).mean()*100:.2f}%")
    
    # Analyze per-subject distribution if subject IDs are available
    if 'subject_ids' in data:
        subject_ids = data['subject_ids']
        unique_subjects = np.unique(subject_ids)
        
        print("\n" + "-" * 40)
        print("Per-Subject Arousal Distribution:")
        print("-" * 40)
        
        for subj in unique_subjects[:5]:  # Show first 5 subjects
            mask = subject_ids == subj
            subj_arousal = arousal[mask]
            counter = Counter(subj_arousal)
            total = len(subj_arousal)
            class_0 = counter.get(0, 0)
            class_1 = counter.get(1, 0)
            print(f"  Subject {subj:2d}: Class 0: {class_0:4d} ({class_0/total*100:5.1f}%), "
                  f"Class 1: {class_1:4d} ({class_1/total*100:5.1f}%)")
        
        if len(unique_subjects) > 5:
            print(f"  ... ({len(unique_subjects) - 5} more subjects)")
    
    print("\n" + "=" * 60)
    print("Recommendation:")
    print("=" * 60)
    
    arousal_counter = Counter(arousal)
    arousal_imbalance = max(arousal_counter.values()) / min(arousal_counter.values())
    
    if arousal_imbalance > 1.5:
        majority_class = max(arousal_counter, key=arousal_counter.get)
        minority_class = min(arousal_counter, key=arousal_counter.get)
        print(f"Arousal shows significant imbalance (Class {majority_class} dominant)")
        print("Suggested actions:")
        print("  1. Use class-weighted CrossEntropyLoss")
        print("  2. Apply oversampling for minority class")
        print("  3. Use Focal Loss to focus on hard examples")
        
        # Calculate suggested weights
        weights = []
        total = len(arousal)
        for c in sorted(arousal_counter.keys()):
            w = total / (len(arousal_counter) * arousal_counter[c])
            weights.append(w)
        print(f"\n  Suggested class weights: {weights}")
    else:
        print("Arousal class distribution is relatively balanced.")
        print("Low accuracy may be due to:")
        print("  1. Arousal being inherently harder to predict from EEG")
        print("  2. Model architecture not capturing arousal-related features")
        print("  3. Need for task-specific attention mechanism")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check class balance in DEAP dataset')
    parser.add_argument('--data_path', type=str, default='data/deap_preprocessed.npz',
                        help='Path to preprocessed data')
    
    args = parser.parse_args()
    analyze_class_balance(args.data_path)
