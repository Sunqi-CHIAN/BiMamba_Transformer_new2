"""
DEAP Dataset Preprocessing and Loading Module.

This module provides utilities for preprocessing and loading the DEAP dataset
(Database for Emotion Analysis using Physiological Signals) for use with
the BiMamba-Transformer network.

DEAP Dataset Structure:
- 32 participants
- 40 trials per participant (40 music videos)
- 32 EEG channels + 8 peripheral channels (40 total)
- Labels: Valence, Arousal, Dominance, Liking (1-9 scale)

Reference:
Koelstra, S., et al. (2012). DEAP: A Database for Emotion Analysis 
using Physiological Signals. IEEE Transactions on Affective Computing.
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List, Union, Any
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings


# DEAP Dataset Constants
DEAP_SAMPLING_RATE = 128  # Hz
DEAP_TRIAL_DURATION = 63  # seconds (3s baseline + 60s trial)
DEAP_NUM_PARTICIPANTS = 32
DEAP_NUM_TRIALS = 40
DEAP_NUM_EEG_CHANNELS = 32
DEAP_NUM_PERIPHERAL_CHANNELS = 8

# Channel indices in DEAP data
# EEG: 0-31, hEOG: 32, vEOG: 33, zEMG: 34, tEMG: 35, 
# GSR: 36, Respiration: 37, Plethysmograph: 38, Temperature: 39
EEG_CHANNEL_IDX = slice(0, 32)
EOG_CHANNEL_IDX = [32, 33]  # hEOG (horizontal) and vEOG (vertical)
EMG_CHANNEL_IDX = [34, 35]  # zEMG (zygomatic) and tEMG (trapezius)

# EEG Channel Names (10-20 system)
EEG_CHANNEL_NAMES = [
    'Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7',
    'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz',
    'Fp2', 'AF4', 'Fz', 'F4', 'F8', 'FC6', 'FC2', 'Cz',
    'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]

# DE frequency bands
DE_FREQ_BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta': (14, 31),
    'gamma': (31, 45)
}


class DEAPPreprocessor:
    """
    Preprocessor for DEAP dataset.
    
    Key improvements over prior version:
    - EEG signals are NOT re-filtered (already 4-45Hz in data_preprocessed_python)
    - EOG/EMG get modality-specific bandpass filtering
    - Trial-level z-score normalization (preserves amplitude info across segments)
    - Ambiguous labels (between threshold_low and threshold_high) are marked for exclusion
    
    Args:
        data_dir: Path to DEAP data directory containing s01.dat to s32.dat
        baseline_duration: Duration of baseline period in seconds (default: 3)
        eog_lowcut/eog_highcut: EOG bandpass filter range
        emg_lowcut/emg_highcut: EMG bandpass filter range
        segment_length: Length of each segment in seconds (default: 4)
        segment_overlap: Overlap between segments in seconds (default: 0)
        binary_labels: If True, convert labels to binary (low/high)
        label_threshold_low: Labels below this are "low" class (default: 4.0)
        label_threshold_high: Labels >= this are "high" class (default: 6.0)
    """
    def __init__(
        self,
        data_dir: str,
        baseline_duration: float = 3.0,
        eog_lowcut: float = 1.0,
        eog_highcut: float = 45.0,
        emg_lowcut: float = 10.0,
        emg_highcut: float = 45.0,
        segment_length: float = 4.0,
        segment_overlap: float = 0.0,
        binary_labels: bool = True,
        label_threshold_low: float = 4.0,
        label_threshold_high: float = 6.0,
        norm_level: str = 'trial',
        extract_de: bool = True
    ):
        self.data_dir = data_dir
        self.baseline_duration = baseline_duration
        self.eog_lowcut = eog_lowcut
        self.eog_highcut = eog_highcut
        self.emg_lowcut = emg_lowcut
        self.emg_highcut = emg_highcut
        self.segment_length = segment_length
        self.segment_overlap = segment_overlap
        self.binary_labels = binary_labels
        self.label_threshold_low = label_threshold_low
        self.label_threshold_high = label_threshold_high
        self.norm_level = norm_level
        self.extract_de = extract_de
        self.fs = DEAP_SAMPLING_RATE
        
        # Calculate sample indices
        self.baseline_samples = int(baseline_duration * self.fs)
        self.segment_samples = int(segment_length * self.fs)
        self.overlap_samples = int(segment_overlap * self.fs)
        
    def load_participant(self, participant_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load data for a single participant.
        
        Args:
            participant_id: Participant number (1-32)
            
        Returns:
            Tuple of (data, labels)
            - data: shape (40, 40, 8064) - 40 trials, 40 channels, 8064 samples
            - labels: shape (40, 4) - Valence, Arousal, Dominance, Liking
        """
        filename = f"s{participant_id:02d}.dat"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"DEAP data file not found: {filepath}")
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        return data['data'], data['labels']
    
    def _bandpass_filter(self, data: np.ndarray, lowcut: float, highcut: float) -> np.ndarray:
        """Apply bandpass filter to the data with specified cutoff frequencies."""
        nyquist = self.fs / 2
        low = max(lowcut / nyquist, 0.001)
        high = min(highcut / nyquist, 0.999)
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=-1)
        return filtered
    
    def bandpass_filter_eog(self, data: np.ndarray) -> np.ndarray:
        """Apply EOG-specific bandpass filter (1-45Hz)."""
        return self._bandpass_filter(data, self.eog_lowcut, self.eog_highcut)
    
    def bandpass_filter_emg(self, data: np.ndarray) -> np.ndarray:
        """Apply EMG-specific bandpass filter (10-45Hz)."""
        return self._bandpass_filter(data, self.emg_lowcut, self.emg_highcut)
        
    def compute_de_from_signal(self, eeg: np.ndarray) -> np.ndarray:
        """Compute DE for 5 frequency bands from raw EEG array (32, samples) -> (32, 5)."""
        n_ch = eeg.shape[0]
        de_out = np.zeros((n_ch, 5), dtype=np.float32)
        nyq = self.fs / 2
        for i, (b_name, (low, high)) in enumerate(DE_FREQ_BANDS.items()):
            low_norm = max(low / nyq, 0.001)
            high_norm = min(high / nyq, 0.999)
            try:
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                filt = signal.filtfilt(b, a, eeg, axis=-1)
                var = np.var(filt, axis=-1)
                de_out[:, i] = 0.5 * np.log(2 * np.pi * np.e * var + 1e-12)
            except Exception:
                de_out[:, i] = 0.0
        return de_out

    def filter_trial_into_bands(self, eeg: np.ndarray) -> np.ndarray:
        """Filter EEG (32, samples) into 5 frequency bands -> shape (5, 32, samples)."""
        nyq = self.fs / 2
        bands_data = []
        for b_name, (low, high) in DE_FREQ_BANDS.items():
            low_norm = max(low / nyq, 0.001)
            high_norm = min(high / nyq, 0.999)
            try:
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                filt = signal.filtfilt(b, a, eeg, axis=-1)
            except Exception:
                filt = np.zeros_like(eeg)
            bands_data.append(filt)
        return np.stack(bands_data, axis=0)
    
    def remove_baseline_period(self, data: np.ndarray) -> np.ndarray:
        """
        Remove baseline period and subtract baseline mean from trial data.
        
        Args:
            data: Input data with shape (..., samples)
            
        Returns:
            Data with baseline removed and baseline mean subtracted
        """
        baseline = data[..., :self.baseline_samples]
        trial = data[..., self.baseline_samples:]
        baseline_mean = baseline.mean(axis=-1, keepdims=True)
        trial_corrected = trial - baseline_mean
        return trial_corrected
    
    def segment_data(self, data: np.ndarray) -> np.ndarray:
        """
        Segment continuous data into fixed-length windows.
        
        Args:
            data: Input data with shape (channels, samples)
            
        Returns:
            Segmented data with shape (num_segments, channels, segment_samples)
        """
        n_channels, n_samples = data.shape
        step = self.segment_samples - self.overlap_samples
        
        segments = []
        start = 0
        while start + self.segment_samples <= n_samples:
            segment = data[:, start:start + self.segment_samples]
            segments.append(segment)
            start += step
            
        if len(segments) == 0:
            padded = np.zeros((n_channels, self.segment_samples))
            padded[:, :n_samples] = data
            segments.append(padded)
            
        return np.array(segments)

    def _count_segments(self, n_samples: int) -> int:
        """Count how many segments will be produced for a sequence length."""
        step = self.segment_samples - self.overlap_samples
        if step <= 0:
            raise ValueError("segment_overlap must be smaller than segment_length")

        if n_samples < self.segment_samples:
            return 1

        return 1 + (n_samples - self.segment_samples) // step
    
    def normalize_trial(self, data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """
        Normalize data at the trial level (before segmentation).
        This preserves amplitude information across segments within the same trial.
        
        Args:
            data: Input data of shape (channels, samples)
            method: Normalization method ('zscore', 'minmax')
            
        Returns:
            Normalized data of same shape
        """
        if method == 'zscore':
            mean = data.mean(axis=-1, keepdims=True)
            std = data.std(axis=-1, keepdims=True) + 1e-8
            return (data - mean) / std
        elif method == 'minmax':
            min_val = data.min(axis=-1, keepdims=True)
            max_val = data.max(axis=-1, keepdims=True)
            return (data - min_val) / (max_val - min_val + 1e-8)
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def extract_signals(
        self, 
        data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract EEG, EOG, and EMG signals from DEAP data.
        
        Args:
            data: Raw DEAP data with shape (40, samples)
            
        Returns:
            Tuple of (eeg, eog, emg)
        """
        eeg = data[EEG_CHANNEL_IDX]   # (32, samples)
        eog = data[EOG_CHANNEL_IDX]   # (2, samples)
        emg = data[EMG_CHANNEL_IDX]   # (2, samples)
        return eeg, eog, emg
    
    def preprocess_trial(
        self, 
        trial_data: np.ndarray, 
        labels: np.ndarray
    ) -> Optional[Dict]:
        """
        Preprocess a single trial.
        
        Key improvements:
        - EEG: NO re-filtering (DEAP data_preprocessed_python is already 4-45Hz filtered)
        - EOG/EMG: modality-specific bandpass filtering
        - Trial-level normalization BEFORE segmentation (preserves amplitude info)
        - Ambiguous labels are flagged for exclusion
        
        Args:
            trial_data: Raw trial data with shape (40, 8064)
            labels: Trial labels [valence, arousal, dominance, liking]
            
        Returns:
            Dictionary containing preprocessed segments and labels, or None if all labels ambiguous
        """
        # Extract modalities
        eeg, eog, emg = self.extract_signals(trial_data)
        
        # EEG: NO additional filtering (already 4-45Hz in data_preprocessed_python)
        # Only remove baseline period
        eeg_processed = self.remove_baseline_period(eeg)
        
        # EOG/EMG: Apply modality-specific bandpass filters, then remove baseline
        eog_filtered = self.bandpass_filter_eog(eog)
        emg_filtered = self.bandpass_filter_emg(emg)
        eog_processed = self.remove_baseline_period(eog_filtered)
        emg_processed = self.remove_baseline_period(emg_filtered)
        
        # Trial-level normalization BEFORE segmentation
        eeg_normalized = self.normalize_trial(eeg_processed, method='zscore')
        eog_normalized = self.normalize_trial(eog_processed, method='zscore')
        emg_normalized = self.normalize_trial(emg_processed, method='zscore')
        
        # Segment data
        eeg_segments = self.segment_data(eeg_normalized).astype(np.float32, copy=False)   # (n_seg, 32, seg_len)
        eog_segments = self.segment_data(eog_normalized).astype(np.float32, copy=False)   # (n_seg, 2, seg_len)
        emg_segments = self.segment_data(emg_normalized).astype(np.float32, copy=False)   # (n_seg, 2, seg_len)
        
        # Process labels with ambiguity handling
        if self.binary_labels:
            processed_labels = np.full(4, -1, dtype=np.int64)  # -1 = ambiguous
            valid_mask = np.zeros(4, dtype=bool)
            
            for i in range(4):  # V, A, D, L
                if labels[i] < self.label_threshold_low:
                    processed_labels[i] = 0  # Low
                    valid_mask[i] = True
                elif labels[i] >= self.label_threshold_high:
                    processed_labels[i] = 1  # High
                    valid_mask[i] = True
                # else: stays -1 (ambiguous), valid_mask stays False
        else:
            processed_labels = labels.astype(np.float32)
            valid_mask = np.ones(4, dtype=bool)
            
        return {
            'eeg': eeg_segments,
            'eog': eog_segments,
            'emg': emg_segments,
            'valence': processed_labels[0],
            'arousal': processed_labels[1],
            'dominance': processed_labels[2],
            'liking': processed_labels[3],
            'valid_valence': valid_mask[0],
            'valid_arousal': valid_mask[1],
            'valid_dominance': valid_mask[2],
            'valid_liking': valid_mask[3],
            'n_segments': len(eeg_segments)
        }
    
    def preprocess_all(
        self, 
        participant_ids: Optional[List[int]] = None,
        save_path: Optional[str] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Preprocess data for all specified participants.
        
        Args:
            participant_ids: List of participant IDs to process, None for all
            save_path: Path to save preprocessed data
            verbose: Whether to print progress
            
        Returns:
            Dictionary containing all preprocessed data
        """
        if participant_ids is None:
            participant_ids = list(range(1, DEAP_NUM_PARTICIPANTS + 1))

        available_participants = []
        for pid in participant_ids:
            filename = f"s{pid:02d}.dat"
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                available_participants.append(pid)
            elif verbose:
                print(f"Warning: DEAP data file not found: {filepath}")

        if len(available_participants) == 0:
            raise FileNotFoundError(
                f"No DEAP participant files found in: {self.data_dir}"
            )

        # Pre-allocate arrays to avoid Python list overhead that can trigger OOM.
        trial_samples = int(DEAP_TRIAL_DURATION * self.fs) - self.baseline_samples
        segments_per_trial = self._count_segments(trial_samples)
        total_samples = len(available_participants) * DEAP_NUM_TRIALS * segments_per_trial

        label_dtype = np.int8 if self.binary_labels else np.float32

        all_data = {
            'eeg': np.empty((total_samples, DEAP_NUM_EEG_CHANNELS, self.segment_samples), dtype=np.float32),
            'eog': np.empty((total_samples, len(EOG_CHANNEL_IDX), self.segment_samples), dtype=np.float32),
            'emg': np.empty((total_samples, len(EMG_CHANNEL_IDX), self.segment_samples), dtype=np.float32),
            'valence': np.empty((total_samples,), dtype=label_dtype),
            'arousal': np.empty((total_samples,), dtype=label_dtype),
            'dominance': np.empty((total_samples,), dtype=label_dtype),
            'liking': np.empty((total_samples,), dtype=label_dtype),
            'valid_valence': np.empty((total_samples,), dtype=bool),
            'valid_arousal': np.empty((total_samples,), dtype=bool),
            'valid_dominance': np.empty((total_samples,), dtype=bool),
            'valid_liking': np.empty((total_samples,), dtype=bool),
            'participant_id': np.empty((total_samples,), dtype=np.int16),
            'trial_id': np.empty((total_samples,), dtype=np.int16)
        }
        if self.extract_de:
            all_data['de'] = np.empty((total_samples, DEAP_NUM_EEG_CHANNELS, 5), dtype=np.float32)

        if verbose:
            bytes_per_sample = (DEAP_NUM_EEG_CHANNELS + len(EOG_CHANNEL_IDX) + len(EMG_CHANNEL_IDX)) * self.segment_samples * 4
            est_gb = total_samples * bytes_per_sample / (1024 ** 3)
            print(f"Planned samples: {total_samples} ({segments_per_trial} segments/trial)")
            print(f"Estimated signal memory footprint: ~{est_gb:.2f} GB")
            print(f"Normalization level: {self.norm_level}")
            print(f"Extract baseline-corrected DE: {self.extract_de}")

        write_idx = 0
        for p_idx, pid in enumerate(available_participants, start=1):
            if verbose:
                print(f"Processing participant {pid} ({p_idx}/{len(available_participants)})...")

            data, labels = self.load_participant(pid)

            # Separate modalities across all 40 trials for participant
            eeg_all = data[:, EEG_CHANNEL_IDX]      # (40, 32, 8064)
            eog_all = data[:, EOG_CHANNEL_IDX]      # (40, 2, 8064)
            emg_all = data[:, EMG_CHANNEL_IDX]      # (40, 2, 8064)

            # 1) Extract baseline-corrected DE features FIRST from continuous raw EEG
            subj_de_all = None
            if self.extract_de:
                bands_full = self.filter_trial_into_bands(eeg_all)  # (5, 40, 32, 8064)
                bands_base = bands_full[:, :, :, :self.baseline_samples]
                bands_stim = bands_full[:, :, :, self.baseline_samples:]

                n_base_sec = max(1, self.baseline_samples // self.fs)
                base_trim = n_base_sec * self.fs
                base_win = bands_base[:, :, :, :base_trim].reshape(
                    5, DEAP_NUM_TRIALS, DEAP_NUM_EEG_CHANNELS, n_base_sec, self.fs
                )
                base_var = np.var(base_win, axis=-1)
                de_base_all = (0.5 * np.log(2 * np.pi * np.e * base_var + 1e-12)).mean(axis=-1)
                de_base_all = np.transpose(de_base_all, (1, 2, 0))  # (40, 32, 5)

                step = self.segment_samples - self.overlap_samples
                n_stim_samples = bands_stim.shape[-1]
                n_seg_sec = max(1, self.segment_samples // self.fs)
                use_subwin = (self.segment_samples == n_seg_sec * self.fs)

                de_seg_list = []
                start = 0
                while start + self.segment_samples <= n_stim_samples:
                    seg_chunk = bands_stim[:, :, :, start:start + self.segment_samples]
                    if use_subwin:
                        seg_win = seg_chunk.reshape(
                            5, DEAP_NUM_TRIALS, DEAP_NUM_EEG_CHANNELS, n_seg_sec, self.fs
                        )
                        seg_var = np.var(seg_win, axis=-1)
                        de_seg = (0.5 * np.log(2 * np.pi * np.e * seg_var + 1e-12)).mean(axis=-1)
                    else:
                        seg_var = np.var(seg_chunk, axis=-1)
                        de_seg = 0.5 * np.log(2 * np.pi * np.e * seg_var + 1e-12)
                    de_seg = np.transpose(de_seg, (1, 2, 0))  # (40, 32, 5)
                    de_seg_list.append(de_seg - de_base_all)
                    start += step

                while len(de_seg_list) < segments_per_trial:
                    de_seg_list.append(np.zeros((DEAP_NUM_TRIALS, DEAP_NUM_EEG_CHANNELS, 5), dtype=np.float32))

                subj_de_all = np.stack(de_seg_list[:segments_per_trial], axis=1).astype(np.float32)

                # Always apply winsorized subject-level z-score to DE features across the 40 trials
                de_flat = subj_de_all.reshape(-1, DEAP_NUM_EEG_CHANNELS, 5)
                p1 = np.percentile(de_flat, 1.0, axis=0, keepdims=True)
                p99 = np.percentile(de_flat, 99.0, axis=0, keepdims=True)
                de_clipped = np.clip(de_flat, p1, p99)
                de_mean = de_clipped.mean(axis=0, keepdims=True).reshape(1, 1, DEAP_NUM_EEG_CHANNELS, 5)
                de_std = de_clipped.std(axis=0, keepdims=True).reshape(1, 1, DEAP_NUM_EEG_CHANNELS, 5) + 1e-6
                subj_de_all = np.clip((subj_de_all - de_mean) / de_std, -10.0, 10.0).astype(np.float32)

            # 2) Vectorized bandpass filter for EOG and EMG across continuous 63s trials
            eog_filtered = self.bandpass_filter_eog(eog_all)
            emg_filtered = self.bandpass_filter_emg(emg_all)

            # Baseline and stimulus slices
            eeg_base = eeg_all[:, :, :self.baseline_samples]
            eeg_stim = eeg_all[:, :, self.baseline_samples:]

            eog_base = eog_filtered[:, :, :self.baseline_samples]
            eog_stim = eog_filtered[:, :, self.baseline_samples:]

            emg_base = emg_filtered[:, :, :self.baseline_samples]
            emg_stim = emg_filtered[:, :, self.baseline_samples:]

            # Subtract trial-level baseline mean
            eeg_stim = eeg_stim - eeg_base.mean(axis=-1, keepdims=True)
            eog_stim = eog_stim - eog_base.mean(axis=-1, keepdims=True)
            emg_stim = emg_stim - emg_base.mean(axis=-1, keepdims=True)

            # Subject-level normalization (preserves trial-to-trial variance within subject)
            if self.norm_level == 'subject':
                def _robust_subj_norm(stim: np.ndarray) -> np.ndarray:
                    n_c = stim.shape[1]
                    flat = stim.transpose(1, 0, 2).reshape(n_c, -1)
                    lo = np.percentile(flat, 1.0, axis=-1, keepdims=True)
                    hi = np.percentile(flat, 99.0, axis=-1, keepdims=True)
                    w = np.clip(flat, lo, hi)
                    m = w.mean(axis=-1).reshape(1, n_c, 1)
                    s = w.std(axis=-1).reshape(1, n_c, 1) + 1e-8
                    return np.clip((stim - m) / s, -15.0, 15.0)

                eeg_stim = _robust_subj_norm(eeg_stim)
                eog_stim = _robust_subj_norm(eog_stim)
                emg_stim = _robust_subj_norm(emg_stim)
            else:
                for t in range(DEAP_NUM_TRIALS):
                    eeg_stim[t] = self.normalize_trial(eeg_stim[t], method='zscore')
                    eog_stim[t] = self.normalize_trial(eog_stim[t], method='zscore')
                    emg_stim[t] = self.normalize_trial(emg_stim[t], method='zscore')

            for trial_idx in range(DEAP_NUM_TRIALS):
                trial_labels = labels[trial_idx]

                # Segment time-domain signals
                eeg_segs = self.segment_data(eeg_stim[trial_idx]).astype(np.float32, copy=False)
                eog_segs = self.segment_data(eog_stim[trial_idx]).astype(np.float32, copy=False)
                emg_segs = self.segment_data(emg_stim[trial_idx]).astype(np.float32, copy=False)
                n_segments = len(eeg_segs)

                if self.extract_de:
                    de_segs = subj_de_all[trial_idx, :n_segments]

                # Process labels with ambiguity handling
                if self.binary_labels:
                    processed_labels = np.full(4, -1, dtype=np.int64)
                    valid_mask = np.zeros(4, dtype=bool)
                    for i in range(4):
                        if trial_labels[i] < self.label_threshold_low:
                            processed_labels[i] = 0
                            valid_mask[i] = True
                        elif trial_labels[i] >= self.label_threshold_high:
                            processed_labels[i] = 1
                            valid_mask[i] = True
                else:
                    processed_labels = trial_labels.astype(np.float32)
                    valid_mask = np.ones(4, dtype=bool)

                end_idx = write_idx + n_segments
                if end_idx > total_samples:
                    raise RuntimeError("Generated segments exceed preallocated size.")

                all_data['eeg'][write_idx:end_idx] = eeg_segs
                all_data['eog'][write_idx:end_idx] = eog_segs
                all_data['emg'][write_idx:end_idx] = emg_segs
                if self.extract_de:
                    all_data['de'][write_idx:end_idx] = de_segs

                all_data['valence'][write_idx:end_idx] = processed_labels[0]
                all_data['arousal'][write_idx:end_idx] = processed_labels[1]
                all_data['dominance'][write_idx:end_idx] = processed_labels[2]
                all_data['liking'][write_idx:end_idx] = processed_labels[3]

                all_data['valid_valence'][write_idx:end_idx] = valid_mask[0]
                all_data['valid_arousal'][write_idx:end_idx] = valid_mask[1]
                all_data['valid_dominance'][write_idx:end_idx] = valid_mask[2]
                all_data['valid_liking'][write_idx:end_idx] = valid_mask[3]

                all_data['participant_id'][write_idx:end_idx] = pid
                all_data['trial_id'][write_idx:end_idx] = trial_idx
                write_idx = end_idx

        if write_idx < total_samples:
            all_data = {key: value[:write_idx].copy() for key, value in all_data.items()}
            
        if verbose:
            print(f"\nPreprocessing complete!")
            print(f"Total samples: {len(all_data['eeg'])}")
            print(f"EEG shape: {all_data['eeg'].shape}")
            print(f"EOG shape: {all_data['eog'].shape}")
            print(f"EMG shape: {all_data['emg'].shape}")
            if 'de' in all_data:
                print(f"DE shape:  {all_data['de'].shape}")
            # Report valid label counts
            for dim in ['valence', 'arousal', 'dominance']:
                valid_key = f'valid_{dim}'
                n_valid = all_data[valid_key].sum()
                n_total = len(all_data[valid_key])
                print(f"{dim.capitalize()}: {n_valid}/{n_total} valid labels "
                      f"({n_valid/n_total*100:.1f}%), "
                      f"{n_total - n_valid} ambiguous excluded")
            
        if save_path:
            np.savez_compressed(save_path, **all_data)
            if verbose:
                print(f"Saved to: {save_path}")
                
        return all_data


class DEFeatureExtractor:
    """
    Extract Differential Entropy (DE) features from EEG signals.
    DE is one of the most effective features for EEG emotion recognition.
    
    Reference: Zheng & Lu (2015), Investigating Critical Frequency Bands and
    Channels for EEG-based Emotion Recognition with Deep Neural Networks.
    """
    def __init__(self, fs: int = DEAP_SAMPLING_RATE):
        self.fs = fs
        self.freq_bands = DE_FREQ_BANDS
    
    def extract(self, eeg_segment: np.ndarray) -> np.ndarray:
        """
        Extract DE features for each channel and frequency band.
        
        Args:
            eeg_segment: EEG data of shape (channels, time)
            
        Returns:
            DE features of shape (channels, n_bands=5)
        """
        n_channels = eeg_segment.shape[0]
        n_bands = len(self.freq_bands)
        de_features = np.zeros((n_channels, n_bands))
        
        for i, (band_name, (low, high)) in enumerate(self.freq_bands.items()):
            # Bandpass filter for this frequency band
            nyquist = self.fs / 2
            low_norm = max(low / nyquist, 0.001)
            high_norm = min(high / nyquist, 0.999)
            
            try:
                b, a = signal.butter(4, [low_norm, high_norm], btype='band')
                filtered = signal.filtfilt(b, a, eeg_segment, axis=-1)
                # DE = 0.5 * log(2 * pi * e * variance)
                variance = np.var(filtered, axis=-1)
                de = 0.5 * np.log(2 * np.pi * np.e * variance + 1e-12)
                de_features[:, i] = de
            except Exception:
                de_features[:, i] = 0.0
        
        return de_features


class EEGAugmentation:
    """
    Data augmentation for EEG and physiological signals.
    
    Reduced-intensity augmentation to avoid destroying signal information.
    
    Args:
        noise_std: Standard deviation of Gaussian noise to add
        time_shift_max: Maximum time shift in samples
        channel_dropout_prob: Probability of dropping each channel
        amplitude_scale_range: Range for random amplitude scaling
        time_mask_max: Maximum length of time segments to mask
    """
    def __init__(
        self,
        noise_std: float = 0.05,
        time_shift_max: int = 10,
        channel_dropout_prob: float = 0.1,
        amplitude_scale_range: tuple = (0.9, 1.1),
        time_mask_max: int = 10,
        de_noise_std: float = 0.20,
        de_band_dropout_prob: float = 0.15
    ):
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.channel_dropout_prob = channel_dropout_prob
        self.amplitude_scale_range = amplitude_scale_range
        self.time_mask_max = time_mask_max
        self.de_noise_std = de_noise_std
        self.de_band_dropout_prob = de_band_dropout_prob
    
    def __call__(
        self,
        eeg: torch.Tensor,
        eog: torch.Tensor,
        emg: torch.Tensor,
        de: Optional[torch.Tensor] = None
    ):
        """
        Apply augmentations to EEG, EOG, EMG, and optionally DE features.
        
        Args:
            eeg: (1, 32, time)
            eog: (2, time)
            emg: (2, time)
            de: Optional (32, 5) Differential Entropy tensor
        """
        # Gaussian noise injection
        if self.noise_std > 0:
            eeg = eeg + torch.randn_like(eeg) * self.noise_std
            eog = eog + torch.randn_like(eog) * self.noise_std * 0.5
            emg = emg + torch.randn_like(emg) * self.noise_std
        
        # Time shift (circular shift)
        if self.time_shift_max > 0:
            shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
            if shift != 0:
                eeg = torch.roll(eeg, shifts=shift, dims=-1)
                eog = torch.roll(eog, shifts=shift, dims=-1)
                emg = torch.roll(emg, shifts=shift, dims=-1)
        
        # Amplitude scaling
        if self.amplitude_scale_range[0] != 1.0 or self.amplitude_scale_range[1] != 1.0:
            scale = torch.empty(1).uniform_(
                self.amplitude_scale_range[0], self.amplitude_scale_range[1]
            ).item()
            eeg = eeg * scale
            eog = eog * scale
            emg = emg * scale
        
        # Time masking - randomly zero out a segment (20% chance)
        if self.time_mask_max > 0 and torch.rand(1).item() < 0.2:
            mask_len = torch.randint(1, self.time_mask_max + 1, (1,)).item()
            time_dim = eeg.shape[-1]
            mask_start = torch.randint(0, max(1, time_dim - mask_len), (1,)).item()
            eeg[..., mask_start:mask_start + mask_len] = 0
            eog[..., mask_start:mask_start + mask_len] = 0
            emg[..., mask_start:mask_start + mask_len] = 0
        
        # Synchronized EEG electrode dropout across waveform and DE
        ch_mask_1d = None
        if self.channel_dropout_prob > 0:
            ch_mask_1d = (torch.rand(eeg.shape[1]) > self.channel_dropout_prob).float()
            eeg = eeg * ch_mask_1d.unsqueeze(0).unsqueeze(-1)
            
        # Regularize DE to prevent memorizing static per-trial baseline fingerprints
        if de is not None:
            if self.de_noise_std > 0:
                de = de + torch.randn_like(de) * self.de_noise_std
            if ch_mask_1d is not None and de.shape[0] == ch_mask_1d.shape[0]:
                de = de * ch_mask_1d.unsqueeze(-1)
            if self.de_band_dropout_prob > 0 and de.dim() == 2:
                band_mask = (torch.rand(de.shape[1]) > self.de_band_dropout_prob).float()
                de = de * band_mask.unsqueeze(0)
            return eeg, eog, emg, de
        
        return eeg, eog, emg


class DEAPDataset(Dataset):
    """
    PyTorch Dataset for DEAP data with ambiguous label filtering.
    
    Args:
        data_dict: Dictionary containing preprocessed data
        label_type: Type of label to use ('valence', 'arousal', 'dominance', 'all')
        transform: Optional transform to apply to data
        filter_ambiguous: If True, exclude samples with ambiguous labels (label == -1)
    """
    def __init__(
        self,
        data_dict: Dict[str, np.ndarray],
        indices: Optional[Union[np.ndarray, List[int]]] = None,
        label_type: str = 'all',
        transform: Optional[callable] = None,
        filter_ambiguous: bool = True
    ):
        self.label_type = label_type
        self.transform = transform
        
        # Build base index array (zero copy when slicing subsets)
        if indices is not None:
            base_indices = np.asarray(indices, dtype=np.int64)
            n_total = len(base_indices)
        else:
            n_total = len(data_dict['eeg'])
            base_indices = np.arange(n_total, dtype=np.int64)
        
        # Build valid index mask - filter out ambiguous labels
        if filter_ambiguous and 'valid_valence' in data_dict:
            if label_type == 'all':
                # Masked multi-task training: sample is valid if ANY task has a valid label
                valid_mask = (
                    data_dict['valid_valence'][base_indices].astype(bool) |
                    data_dict['valid_arousal'][base_indices].astype(bool) |
                    data_dict['valid_dominance'][base_indices].astype(bool)
                )
            elif label_type in ('valence', 'arousal', 'dominance'):
                valid_mask = data_dict[f'valid_{label_type}'][base_indices].astype(bool)
            else:
                valid_mask = np.ones(n_total, dtype=bool)
            
            self.valid_indices = base_indices[valid_mask]
        else:
            # Fallback: filter by label value == -1
            if label_type == 'all':
                valid_mask = (
                    (data_dict['valence'][base_indices] != -1) |
                    (data_dict['arousal'][base_indices] != -1) |
                    (data_dict['dominance'][base_indices] != -1)
                )
            elif label_type in ('valence', 'arousal', 'dominance'):
                valid_mask = data_dict[label_type][base_indices] != -1
            else:
                valid_mask = np.ones(n_total, dtype=bool)
            self.valid_indices = base_indices[valid_mask]
        
        # Store references to data arrays without copying
        self.eeg = data_dict['eeg']
        self.eog = data_dict['eog']
        self.emg = data_dict['emg']
        self.valence = data_dict['valence']
        self.arousal = data_dict['arousal']
        self.dominance = data_dict['dominance']
        
        self.valid_valence = data_dict.get('valid_valence', None)
        self.valid_arousal = data_dict.get('valid_arousal', None)
        self.valid_dominance = data_dict.get('valid_dominance', None)
        self.de = data_dict.get('de', None)
        
        self.participant_ids = data_dict.get('participant_id', None)
        self.trial_ids = data_dict.get('trial_id', None)
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample (using valid index mapping).
        """
        real_idx = self.valid_indices[idx]
        
        eeg = self.eeg[real_idx]
        eog = self.eog[real_idx]
        emg = self.emg[real_idx]
        
        # Ensure correct shape: EEG -> (1, 32, time)
        if eeg.ndim == 2:
            eeg = eeg[np.newaxis, :, :]
        else:
            eeg = eeg.reshape(1, -1, eeg.shape[-1])
        
        # EOG: (2, time), EMG: (2, time)
        if emg.ndim == 1:
            emg = emg[np.newaxis, :]
        
        # Convert to tensors
        eeg = torch.from_numpy(eeg).float()
        eog = torch.from_numpy(eog).float()
        emg = torch.from_numpy(emg).float()
        de_tensor = torch.from_numpy(self.de[real_idx]).float() if self.de is not None else None
        
        # Apply transforms (synchronizing EEG and DE channel dropout when DE is present)
        if self.transform:
            if de_tensor is not None:
                eeg, eog, emg, de_tensor = self.transform(eeg, eog, emg, de_tensor)
            else:
                eeg, eog, emg = self.transform(eeg, eog, emg)
            
        # Determine validity per task
        if self.valid_valence is not None:
            mask_v = bool(self.valid_valence[real_idx])
            mask_a = bool(self.valid_arousal[real_idx])
            mask_d = bool(self.valid_dominance[real_idx])
        else:
            mask_v = bool(self.valence[real_idx] != -1)
            mask_a = bool(self.arousal[real_idx] != -1)
            mask_d = bool(self.dominance[real_idx] != -1)

        val_raw = int(self.valence[real_idx])
        aro_raw = int(self.arousal[real_idx])
        dom_raw = int(self.dominance[real_idx])

        # Clamp target to >= 0 so CE doesn't crash on invalid (-1) entries before masking
        val_target = max(0, val_raw)
        aro_target = max(0, aro_raw)
        dom_target = max(0, dom_raw)
            
        # Get labels and valid masks
        if self.label_type == 'valence':
            labels = torch.tensor(val_target, dtype=torch.long)
            valid_mask = torch.tensor(mask_v, dtype=torch.bool)
        elif self.label_type == 'arousal':
            labels = torch.tensor(aro_target, dtype=torch.long)
            valid_mask = torch.tensor(mask_a, dtype=torch.bool)
        elif self.label_type == 'dominance':
            labels = torch.tensor(dom_target, dtype=torch.long)
            valid_mask = torch.tensor(mask_d, dtype=torch.bool)
        elif self.label_type == 'all':
            labels = {
                'valence': torch.tensor(val_target, dtype=torch.long),
                'arousal': torch.tensor(aro_target, dtype=torch.long),
                'dominance': torch.tensor(dom_target, dtype=torch.long)
            }
            valid_mask = {
                'valence': torch.tensor(mask_v, dtype=torch.bool),
                'arousal': torch.tensor(mask_a, dtype=torch.bool),
                'dominance': torch.tensor(mask_d, dtype=torch.bool)
            }
        else:
            raise ValueError(f"Unknown label_type: {self.label_type}")
            
        result = {
            'eeg': eeg,
            'eog': eog,
            'emg': emg,
            'labels': labels,
            'valid_mask': valid_mask
        }
        
        if de_tensor is not None:
            result['de'] = de_tensor
        
        if self.participant_ids is not None:
            # Participant IDs are 1-32, make them 0-31 for nn.CrossEntropyLoss
            result['participant_id'] = torch.tensor(self.participant_ids[real_idx] - 1, dtype=torch.long)
            result['subject_id'] = torch.tensor(self.participant_ids[real_idx], dtype=torch.long)
            
        if self.trial_ids is not None:
            result['trial_id'] = torch.tensor(self.trial_ids[real_idx], dtype=torch.long)
            
        return result


def _normalize_waveforms_per_trial_inplace(data_dict: Dict[str, np.ndarray]) -> None:
    """
    Normalize raw waveform arrays ('eeg', 'eog', 'emg') per trial & per channel in-place.
    Eliminates 60-minute session electrode impedance drift for SpatialConv + BiMamba,
    while leaving 'de' features subject-normalized to preserve inter-trial band power.
    """
    if 'participant_id' not in data_dict or 'trial_id' not in data_dict:
        return

    pids = data_dict['participant_id']
    tids = data_dict['trial_id']
    n_total = len(pids)

    # Fast, low-memory chunked path when trials are stored in standard 29-segment blocks
    uniform_29 = (
        n_total > 0 and n_total % 29 == 0
        and np.all(pids.reshape(-1, 29) == pids.reshape(-1, 29)[:, :1])
        and np.all(tids.reshape(-1, 29) == tids.reshape(-1, 29)[:, :1])
    )

    for mod_key in ('eeg', 'eog', 'emg'):
        if mod_key not in data_dict:
            continue
        arr = data_dict[mod_key]
        if not arr.flags.writeable:
            arr = arr.copy()
            data_dict[mod_key] = arr

        if uniform_29:
            n_trials = n_total // 29
            c_dim, t_dim = arr.shape[1], arr.shape[2]
            chunk_trials = 160  # ~300 MB peak temporary buffer
            for t0 in range(0, n_trials, chunk_trials):
                t1 = min(t0 + chunk_trials, n_trials)
                s0, s1 = t0 * 29, t1 * 29
                view = arr[s0:s1].reshape(t1 - t0, 29, c_dim, t_dim)
                # Even segments 0, 2, ..., 28 cover the exact contiguous 60s stimulus without overlap
                non_overlap = view[:, 0::2, :, :]
                m = non_overlap.mean(axis=(1, 3), keepdims=True)
                s = non_overlap.std(axis=(1, 3), keepdims=True) + 1e-8
                view -= m
                view /= s
        else:
            unique_keys = np.unique(np.stack([pids, tids], axis=1), axis=0)
            for p_val, t_val in unique_keys:
                idx = np.where((pids == p_val) & (tids == t_val))[0]
                sub = arr[idx]
                non_ov = sub[0::2] if len(sub) >= 2 else sub
                m = non_ov.mean(axis=(0, 2), keepdims=True)
                s = non_ov.std(axis=(0, 2), keepdims=True) + 1e-8
                arr[idx] = (sub - m) / s


def create_deap_dataloaders(
    data_path: Optional[str] = None,
    batch_size: int = 32,
    test_size: float = 0.2,
    val_size: float = 0.2,
    val_subjects: Optional[List[int]] = None,
    label_type: str = 'all',
    subject_independent: bool = False,
    test_subjects: Optional[List[int]] = None,
    random_state: int = 42,
    num_workers: int = 4,
    filter_ambiguous: bool = True,
    waveform_norm: str = 'trial',
    use_de: bool = True,
    data_dict: Optional[Dict[str, np.ndarray]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for DEAP dataset.
    Supports in-memory zero-copy subsetting to avoid Linux OOM kills during multi-fold CV.
    """
    # Load preprocessed data if not provided
    if data_dict is None:
        if data_path is None:
            raise ValueError("Either data_path or data_dict must be provided.")
        data = np.load(data_path, allow_pickle=True)
        data_dict = {key: data[key] for key in data.files}

    if not use_de and 'de' in data_dict:
        del data_dict['de']
        print("DE features disabled via configuration (waveform-only mode).")
    elif 'de' in data_dict:
        print(f"Loaded DE features: {data_dict['de'].shape} (Subject-normalized + Baseline-corrected)")
    else:
        print("Note: 'de' features not present in .npz (running in waveform-only mode).")

    if waveform_norm == 'trial' and not data_dict.get('_waveforms_normalized', False):
        _normalize_waveforms_per_trial_inplace(data_dict)
        data_dict['_waveforms_normalized'] = True
        print("Waveform normalization: trial-level z-score (eliminates 60-min session impedance drift)")
    elif waveform_norm == 'trial':
        print("Waveform normalization: trial-level z-score (pre-normalized, zero redundant work)")
    else:
        print(f"Waveform normalization: {waveform_norm}-level (as stored in .npz)")
    
    if subject_independent:
        if test_subjects is None:
            test_subjects = [1, 2, 3]
        test_subjects = [int(s) for s in test_subjects]
            
        participant_ids = data_dict['participant_id']
        all_subjects = [int(s) for s in np.unique(participant_ids)]
        remaining_subjects = [s for s in all_subjects if s not in test_subjects]
        
        if val_subjects is not None:
            val_subjects = [int(s) for s in val_subjects if s in remaining_subjects]
            train_subjects = [s for s in remaining_subjects if s not in val_subjects]
        else:
            # Ensure at least 4 validation subjects to guarantee representative validation
            n_val_subjects = max(4, int(len(remaining_subjects) * val_size))
            n_val_subjects = min(n_val_subjects, max(1, len(remaining_subjects) - 1))
            np.random.seed(random_state)
            shuffled = list(remaining_subjects)
            np.random.shuffle(shuffled)
            val_subjects = shuffled[:n_val_subjects]
            train_subjects = shuffled[n_val_subjects:]
        
        train_mask = np.isin(participant_ids, train_subjects)
        val_mask = np.isin(participant_ids, val_subjects)
        test_mask = np.isin(participant_ids, test_subjects)
        
        train_indices = np.where(train_mask)[0]
        val_indices = np.where(val_mask)[0]
        test_indices = np.where(test_mask)[0]
        
        print(f"Subject-independent split:")
        print(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
        print(f"  Val subjects   ({len(val_subjects)}): {sorted(val_subjects)}")
        print(f"  Test subjects  ({len(test_subjects)}): {sorted(test_subjects)}")
        
    else:
        n_samples = len(data_dict['eeg'])
        indices = np.arange(n_samples)
        train_val_indices, test_indices = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size, random_state=random_state
        )
    
    # Proven anti-overfitting augmentation for 50%-overlapping 4s segments + synchronized DE jitter/dropout
    train_transform = EEGAugmentation(
        noise_std=0.03,
        time_shift_max=25,
        channel_dropout_prob=0.15,
        amplitude_scale_range=(0.85, 1.15),
        time_mask_max=50,
        de_noise_std=0.20,
        de_band_dropout_prob=0.15
    )
    
    # Create zero-copy datasets (indexing directly into master arrays without copying 3GB of RAM)
    train_dataset = DEAPDataset(
        data_dict, indices=train_indices, label_type=label_type, 
        transform=train_transform, filter_ambiguous=filter_ambiguous
    )
    val_dataset = DEAPDataset(
        data_dict, indices=val_indices, label_type=label_type, filter_ambiguous=filter_ambiguous
    )
    test_dataset = DEAPDataset(
        data_dict, indices=test_indices, label_type=label_type, filter_ambiguous=filter_ambiguous
    )
    
    # Create dataloaders (pin_memory=False to conserve host memory under WSL2)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )
    
    print(f"Dataset splits (after filtering ambiguous labels):")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader


def load_preprocessed_deap(
    data_path: str,
    label_type: str = 'all'
) -> DEAPDataset:
    """Load preprocessed DEAP data into a Dataset."""
    data = np.load(data_path, allow_pickle=True)
    data_dict = {key: data[key] for key in data.files}
    return DEAPDataset(data_dict, label_type=label_type)


# Usage example and preprocessing script
if __name__ == "__main__":
    import argparse

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
    
    DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "DEAP", "data_preprocessed_python")
    DEFAULT_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "DEAP", "deap_preprocessed.npz")

    parser = argparse.ArgumentParser(description="Preprocess DEAP dataset (Subject-Norm + Baseline-Corrected DE)")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to raw DEAP data_preprocessed_python (.dat files)")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Output .npz path")
    parser.add_argument("--norm_level", type=str, default="trial", choices=["trial", "subject"])
    parser.add_argument("--no_de", action="store_true", help="Disable DE feature extraction")
    cli_args = parser.parse_args()

    DATA_DIR = cli_args.data_dir
    OUTPUT_PATH = cli_args.output_path
    DEFAULT_SEGMENT_LENGTH = 4.0
    DEFAULT_SEGMENT_OVERLAP = 2.0
    DEFAULT_BINARY_LABELS = True
    
    print("=" * 60)
    print("DEAP Dataset Preprocessing (Optimized)")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Segment length: {DEFAULT_SEGMENT_LENGTH}s")
    print(f"Segment overlap: {DEFAULT_SEGMENT_OVERLAP}s")
    print(f"Binary labels: {DEFAULT_BINARY_LABELS}")
    print(f"Label thresholds: low < 4.0, high >= 6.0 (ambiguous excluded)")
    print(f"EEG: NO re-filtering (already 4-45Hz)")
    print(f"EOG: 1-45Hz bandpass")
    print(f"EMG: 10-45Hz bandpass")
    print(f"Normalization: {cli_args.norm_level}-level winsorized z-score (preserves trial dynamics)")
    print(f"Differential Entropy (DE): {'5 frequency bands with baseline correction' if not cli_args.no_de else 'Disabled'}")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        print(f"\n错误: 数据目录不存在: {DATA_DIR}")
        print("请确保 DEAP 原始 .dat 数据集已放置在正确位置，或通过 --data_dir 指定目录。")
        exit(1)
    
    output_dir = os.path.dirname(OUTPUT_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")
    
    preprocessor = DEAPPreprocessor(
        data_dir=DATA_DIR,
        segment_length=DEFAULT_SEGMENT_LENGTH,
        segment_overlap=DEFAULT_SEGMENT_OVERLAP,
        binary_labels=DEFAULT_BINARY_LABELS,
        label_threshold_low=4.0,
        label_threshold_high=6.0,
        norm_level=cli_args.norm_level,
        extract_de=not cli_args.no_de
    )
    
    data = preprocessor.preprocess_all(save_path=OUTPUT_PATH)
    
    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print(f"EEG shape: {data['eeg'].shape}")
    print(f"EOG shape: {data['eog'].shape}")
    print(f"EMG shape: {data['emg'].shape}")
    if 'de' in data:
        print(f"DE shape:  {data['de'].shape}")
    
    # Report label distribution (only valid labels)
    for dim in ['valence', 'arousal', 'dominance']:
        valid = data[f'valid_{dim}'].astype(bool)
        labels = data[dim][valid]
        if len(labels) > 0:
            counts = np.bincount(labels, minlength=2)
            print(f"{dim.capitalize()} distribution (valid only): {counts}")
    
    print(f"\n预处理数据已保存至: {OUTPUT_PATH}")
