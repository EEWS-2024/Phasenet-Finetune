#!/usr/bin/env python3
"""
DataReader untuk PhaseNet Indonesia dengan Sliding Window 3000 samples
Kompatibel dengan model pretrained 190703-214543 (STEAD dataset)

Strategy: Pecah data Indonesia (30,000+ samples) menjadi multiple windows 3000 samples
"""

import os
import numpy as np
import pandas as pd
from data_reader import DataReader, DataConfig

def normalize_long(data):
    """Normalize data dengan robust method untuk data panjang"""
    if data.shape[0] < 10:
        return data
    
    # Per-channel normalization
    for i in range(data.shape[2]):
        channel_data = data[:, 0, i]
        
        # Robust statistics menggunakan percentiles
        p1, p99 = np.percentile(channel_data, [1, 99])
        
        if p99 > p1:
            # Normalize to [-1, 1] range
            data[:, 0, i] = 2 * (channel_data - p1) / (p99 - p1) - 1
            # Clip extreme values
            data[:, 0, i] = np.clip(data[:, 0, i], -3, 3)
        else:
            # Fallback untuk data yang sangat uniform
            std = np.std(channel_data)
            if std > 0:
                data[:, 0, i] = (channel_data - np.mean(channel_data)) / std
                data[:, 0, i] = np.clip(data[:, 0, i], -3, 3)
    
    return data

class DataConfig_Indonesia_3000(DataConfig):
    """Konfigurasi untuk model pretrained 3000 samples"""
    
    # Menggunakan konfigurasi yang sama dengan model pretrained
    sampling_rate = 100
    dt = 1.0 / sampling_rate
    
    # Window size: 30 detik (3000 samples) - sama dengan pretrained model
    window_length = 3000  # 30 detik pada 100 Hz
    
    # Input/Output shape disesuaikan dengan model pretrained
    X_shape = [window_length, 1, 3]
    Y_shape = [window_length, 1, 3]
    
    n_channel = 3
    n_class = 3
    
    # Label parameters sama dengan pretrained
    label_shape = "gaussian"
    label_width = 30
    
    # Event gap minimum
    min_event_gap = 3 * sampling_rate
    
    dtype = "float32"
    
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

class DataReader_Indonesia_Sliding_Train(DataReader):
    """Data Reader dengan sliding window 3000 samples untuk training"""
    
    def __init__(self, format="numpy", config=DataConfig_Indonesia_3000(), **kwargs):
        super().__init__(format=format, config=config, **kwargs)
        
        self.window_length = config.window_length  # 3000 samples
        self.window_step = 1500  # 50% overlap untuk diversity
        self.label_width = config.label_width
        
        # Prepare sliding windows untuk setiap file
        self._prepare_sliding_windows()
        
        print(f"Indonesia Sliding DataReader initialized:")
        print(f"  Window length: {self.window_length} samples ({self.window_length/100:.1f} seconds)")
        print(f"  Window step: {self.window_step} samples ({self.window_step/100:.1f} seconds)")
        print(f"  Total sliding windows: {len(self.sliding_windows)}")

    def _prepare_sliding_windows(self):
        """Prepare list of sliding windows dari semua NPZ files"""
        self.sliding_windows = []
        
        for i, npz_file in enumerate(self.data_list):
            
            try:
                # Load NPZ data untuk analisis ukuran
                npz_path = os.path.join(self.data_dir, npz_file)
                npz_data = np.load(npz_path, allow_pickle=True)
                
                # Get data shape
                if 'data' in npz_data:
                    data_length = npz_data['data'].shape[0]
                elif 'waveform' in npz_data:
                    data_length = npz_data['waveform'].shape[0]
                else:
                    data_keys = [k for k in npz_data.keys() if not k.endswith('_idx')]
                    if data_keys:
                        data_length = npz_data[data_keys[0]].shape[0]
                    else:
                        print(f"⚠️  No waveform data found in {npz_file}, skipping...")
                        continue
                
                # Extract P and S indices
                p_idx = npz_data['p_idx'][0][0] if len(npz_data['p_idx']) > 0 and len(npz_data['p_idx'][0]) > 0 else None
                s_idx = npz_data['s_idx'][0][0] if len(npz_data['s_idx']) > 0 and len(npz_data['s_idx'][0]) > 0 else None
                
                # Generate sliding windows
                start = 0
                while start + self.window_length <= data_length:
                    window_info = {
                        'file_index': i,
                        'npz_file': npz_file,
                        'window_start': start,
                        'window_end': start + self.window_length,
                        'original_p_idx': p_idx,
                        'original_s_idx': s_idx
                    }
                    self.sliding_windows.append(window_info)
                    start += self.window_step
                
                npz_data.close()
                
            except Exception as e:
                print(f"Error analyzing {npz_file}: {e}")
                continue
        
        print(f"Generated {len(self.sliding_windows)} sliding windows from {len(self.data_list)} files")

    def __len__(self):
        return len(self.sliding_windows)

    def __getitem__(self, i):
        window_info = self.sliding_windows[i]
        npz_file = window_info['npz_file']
        window_start = window_info['window_start']
        window_end = window_info['window_end']
        
        try:
            # Load NPZ data
            npz_data = np.load(os.path.join(self.data_dir, npz_file), allow_pickle=True)
            
            # Extract waveform data
            if 'data' in npz_data:
                sample = npz_data['data']
            elif 'waveform' in npz_data:
                sample = npz_data['waveform']
            else:
                data_keys = [k for k in npz_data.keys() if not k.endswith('_idx')]
                if data_keys:
                    sample = npz_data[data_keys[0]]
                else:
                    raise ValueError(f"No waveform data found in {npz_file}")
            
            # Ensure correct shape
            if isinstance(sample, np.ndarray):
                if len(sample.shape) == 2:  # (time, channels)
                    sample = sample[:, np.newaxis, :]  # Add middle dimension
                elif len(sample.shape) == 1:  # (time,) - single channel
                    sample = sample[:, np.newaxis, np.newaxis]  # Add dimensions
            else:
                raise ValueError(f"Sample is not numpy array: {type(sample)}")
            
            # Extract window
            sample_window = sample[window_start:window_end, :, :]
            
            # Get original P and S indices
            original_p_idx = window_info['original_p_idx']
            original_s_idx = window_info['original_s_idx']
            
            # Calculate relative P and S indices in this window
            p_idx_rel = None
            s_idx_rel = None
            
            if original_p_idx is not None:
                if window_start <= original_p_idx < window_end:
                    p_idx_rel = original_p_idx - window_start
            
            if original_s_idx is not None:
                if window_start <= original_s_idx < window_end:
                    s_idx_rel = original_s_idx - window_start
            
            # Create target labels
            target = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target[:, 0, 0] = 1.0  # Background probability
            
            # P-wave label (Gaussian)
            if p_idx_rel is not None and 0 <= p_idx_rel < self.window_length:
                p_start = max(0, p_idx_rel - self.label_width)
                p_end = min(self.window_length, p_idx_rel + self.label_width)
                for j in range(p_start, p_end):
                    target[j, 0, 1] = np.exp(-((j - p_idx_rel) ** 2) / (2 * (self.label_width / 3) ** 2))
                    target[j, 0, 0] = 1.0 - target[j, 0, 1]
            
            # S-wave label (Gaussian)
            if s_idx_rel is not None and 0 <= s_idx_rel < self.window_length and s_idx_rel != p_idx_rel:
                s_start = max(0, s_idx_rel - self.label_width)
                s_end = min(self.window_length, s_idx_rel + self.label_width)
                for j in range(s_start, s_end):
                    target[j, 0, 2] = np.exp(-((j - s_idx_rel) ** 2) / (2 * (self.label_width / 3) ** 2))
                    target[j, 0, 0] = 1.0 - target[j, 0, 1] - target[j, 0, 2]
            
            # Normalize target probabilities
            target_sum = np.sum(target, axis=2, keepdims=True)
            target_sum = np.where(target_sum == 0, 1, target_sum)
            target = target / target_sum
            
            # Normalize waveform data
            sample_window = normalize_long(sample_window)
            
            # Create unique filename for this window
            base_name = npz_file.replace('.npz', '')
            window_name = f"{base_name}_w{window_start:06d}"
            
            return sample_window.astype(np.float32), target.astype(np.float32), window_name
            
        except Exception as e:
            print(f"Error loading window from {npz_file}: {e}")
            # Return zero arrays as fallback
            sample_window = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target[:, 0, 0] = 1.0  # All background
            return sample_window, target, f"error_window_{i}"

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=True, drop_remainder=True):
        """Create TensorFlow dataset"""
        import tensorflow as tf
        
        output_types = (tf.float32, tf.float32, tf.string)
        output_shapes = ([self.window_length, 1, 3], [self.window_length, 1, 3], [])
        
        from data_reader import dataset_map
        dataset = dataset_map(
            self, 
            output_types=output_types, 
            output_shapes=output_shapes, 
            num_parallel_calls=num_parallel_calls, 
            shuffle=shuffle
        )
        
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(1)
        
        return dataset

class DataReader_Indonesia_Sliding_Test(DataReader):
    """Data Reader untuk testing dengan sliding window 3000 samples"""
    
    def __init__(self, format="numpy", config=DataConfig_Indonesia_3000(), **kwargs):
        super().__init__(format=format, config=config, **kwargs)
        
        self.window_length = config.window_length  # 3000 samples
        self.window_step = 3000  # No overlap untuk testing (discrete windows)
        
        # Prepare sliding windows
        self._prepare_sliding_windows()
        
        print(f"Indonesia Sliding Test DataReader initialized:")
        print(f"  Window length: {self.window_length} samples ({self.window_length/100:.1f} seconds)")
        print(f"  Total test windows: {len(self.sliding_windows)}")

    def _prepare_sliding_windows(self):
        """Prepare list of sliding windows untuk testing"""
        self.sliding_windows = []
        
        for i, npz_file in enumerate(self.data_list):
            
            try:
                npz_path = os.path.join(self.data_dir, npz_file)
                npz_data = np.load(npz_path, allow_pickle=True)
                
                if 'data' in npz_data:
                    data_length = npz_data['data'].shape[0]
                elif 'waveform' in npz_data:
                    data_length = npz_data['waveform'].shape[0]
                else:
                    data_keys = [k for k in npz_data.keys() if not k.endswith('_idx')]
                    if data_keys:
                        data_length = npz_data[data_keys[0]].shape[0]
                    else:
                        continue
                
                p_idx = npz_data['p_idx'][0][0] if len(npz_data['p_idx']) > 0 and len(npz_data['p_idx'][0]) > 0 else None
                s_idx = npz_data['s_idx'][0][0] if len(npz_data['s_idx']) > 0 and len(npz_data['s_idx'][0]) > 0 else None
                
                # Generate non-overlapping windows untuk testing
                start = 0
                while start + self.window_length <= data_length:
                    window_info = {
                        'file_index': i,
                        'npz_file': npz_file,
                        'window_start': start,
                        'window_end': start + self.window_length,
                        'original_p_idx': p_idx,
                        'original_s_idx': s_idx
                    }
                    self.sliding_windows.append(window_info)
                    start += self.window_step
                
                npz_data.close()
                
            except Exception as e:
                print(f"Error analyzing {npz_file}: {e}")
                continue

    def __len__(self):
        return len(self.sliding_windows)

    def __getitem__(self, i):
        window_info = self.sliding_windows[i]
        npz_file = window_info['npz_file']
        window_start = window_info['window_start']
        window_end = window_info['window_end']
        
        try:
            npz_data = np.load(os.path.join(self.data_dir, npz_file), allow_pickle=True)
            
            if 'data' in npz_data:
                sample = npz_data['data']
            elif 'waveform' in npz_data:
                sample = npz_data['waveform']
            else:
                data_keys = [k for k in npz_data.keys() if not k.endswith('_idx')]
                if data_keys:
                    sample = npz_data[data_keys[0]]
                else:
                    raise ValueError(f"No waveform data found in {npz_file}")
            
            if isinstance(sample, np.ndarray):
                if len(sample.shape) == 2:
                    sample = sample[:, np.newaxis, :]
                elif len(sample.shape) == 1:
                    sample = sample[:, np.newaxis, np.newaxis]
            else:
                raise ValueError(f"Sample is not numpy array: {type(sample)}")
            
            sample_window = sample[window_start:window_end, :, :]
            
            original_p_idx = window_info['original_p_idx']
            original_s_idx = window_info['original_s_idx']
            
            p_idx_rel = None
            s_idx_rel = None
            
            if original_p_idx is not None:
                if window_start <= original_p_idx < window_end:
                    p_idx_rel = original_p_idx - window_start
            
            if original_s_idx is not None:
                if window_start <= original_s_idx < window_end:
                    s_idx_rel = original_s_idx - window_start
            
            # Create target (same as training)
            target = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target[:, 0, 0] = 1.0
            
            if p_idx_rel is not None and 0 <= p_idx_rel < self.window_length:
                p_start = max(0, p_idx_rel - 30)
                p_end = min(self.window_length, p_idx_rel + 30)
                for j in range(p_start, p_end):
                    target[j, 0, 1] = np.exp(-((j - p_idx_rel) ** 2) / (2 * 10 ** 2))
                    target[j, 0, 0] = 1.0 - target[j, 0, 1]
            
            if s_idx_rel is not None and 0 <= s_idx_rel < self.window_length and s_idx_rel != p_idx_rel:
                s_start = max(0, s_idx_rel - 30)
                s_end = min(self.window_length, s_idx_rel + 30)
                for j in range(s_start, s_end):
                    target[j, 0, 2] = np.exp(-((j - s_idx_rel) ** 2) / (2 * 10 ** 2))
                    target[j, 0, 0] = 1.0 - target[j, 0, 1] - target[j, 0, 2]
            
            target_sum = np.sum(target, axis=2, keepdims=True)
            target_sum = np.where(target_sum == 0, 1, target_sum)
            target = target / target_sum
            
            sample_window = normalize_long(sample_window)
            
            base_name = npz_file.replace('.npz', '')
            window_name = f"{base_name}_w{window_start:06d}"
            
            # Return format untuk testing: sample, target, filename, p_idx, s_idx
            p_return = np.array([p_idx_rel if p_idx_rel is not None else -1], dtype=np.int32)
            s_return = np.array([s_idx_rel if s_idx_rel is not None else -1], dtype=np.int32)
            
            return (sample_window.astype(np.float32), 
                   target.astype(np.float32), 
                   window_name,
                   p_return,
                   s_return)
            
        except Exception as e:
            print(f"Error loading test window from {npz_file}: {e}")
            sample_window = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target[:, 0, 0] = 1.0
            return (sample_window, target, f"error_window_{i}", 
                   np.array([-1], dtype=np.int32), 
                   np.array([-1], dtype=np.int32))

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=False, drop_remainder=False):
        """Create TensorFlow dataset for testing"""
        import tensorflow as tf
        
        output_types = (tf.float32, tf.float32, tf.string, tf.int32, tf.int32)
        output_shapes = ([self.window_length, 1, 3], [self.window_length, 1, 3], [], [1], [1])
        
        from data_reader import dataset_map
        dataset = dataset_map(
            self, 
            output_types=output_types, 
            output_shapes=output_shapes, 
            num_parallel_calls=num_parallel_calls, 
            shuffle=shuffle
        )
        
        dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
        dataset = dataset.prefetch(1)
        
        return dataset 