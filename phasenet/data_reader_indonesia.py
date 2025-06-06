#!/usr/bin/env python3
"""
Data Reader khusus untuk dataset Indonesi
"""

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np
import pandas as pd
from data_reader import DataReader, DataConfig, normalize_long
import os

class DataConfig_Indonesia(DataConfig):
    """Konfigurasi khusus untuk data Indonesia"""
    
    # Berdasarkan analisis data Indonesia
    sampling_rate = 100
    dt = 1.0 / sampling_rate
    
    # Window size: 170 detik (17,000 samples)
    window_length = 17000  # 170 detik pada 100 Hz
    
    # Input/Output shape disesuaikan
    X_shape = [window_length, 1, 3]
    Y_shape = [window_length, 1, 3]
    
    n_channel = 3
    n_class = 3
    
    # Label parameters
    label_shape = "gaussian"
    label_width = 30
    
    # Event gap minimum
    min_event_gap = 3 * sampling_rate
    
    dtype = "float32"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

class DataReader_Indonesia_Train(DataReader):
    """Data Reader untuk training dengan 99% coverage data Indonesia"""
    
    def __init__(self, format="numpy", config=DataConfig_Indonesia(), **kwargs):
        super().__init__(format=format, config=config, **kwargs)
        
        self.min_event_gap = config.min_event_gap
        self.buffer_channels = {}
        self.window_length = config.window_length
        
        # Shift range disesuaikan untuk window yang sangat besar dengan buffer
        self.shift_range = [-2000 + self.label_width * 2, 1000 - self.label_width * 2]
        
        # Select range untuk window extraction - dengan buffer optimal
        self.select_range = [2000, 2000 + self.window_length]  # Start earlier untuk buffer
        
        print(f"Indonesia DataReader initialized:")
        print(f"  Window length: {self.window_length} samples ({self.window_length/100:.1f} seconds)")
        print(f"  Select range: {self.select_range}")

    def __getitem__(self, i):
        base_name = str(self.data_list.iloc[i]).split("/")[-1]
        
        try:
            # Load NPZ data
            npz_data = np.load(os.path.join(self.data_dir, base_name), allow_pickle=True)
            
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
                    raise ValueError(f"No waveform data found in {base_name}")
            
            # Ensure sample is numpy array with correct shape
            if isinstance(sample, np.ndarray):
                if len(sample.shape) == 2:  # (time, channels)
                    sample = sample[:, np.newaxis, :]  # Add middle dimension
                elif len(sample.shape) == 1:  # (time,) - single channel
                    sample = sample[:, np.newaxis, np.newaxis]  # Add dimensions
            else:
                raise ValueError(f"Sample is not numpy array: {type(sample)}")
            
            # Get P and S indices
            p_idx = npz_data['p_idx'][0][0] if len(npz_data['p_idx']) > 0 and len(npz_data['p_idx'][0]) > 0 else 3000
            s_idx = npz_data['s_idx'][0][0] if len(npz_data['s_idx']) > 0 and len(npz_data['s_idx'][0]) > 0 else 6000
            
            # Ensure indices are within bounds
            p_idx = max(1000, min(p_idx, sample.shape[0] - self.window_length))  # Allow earlier start
            s_idx = max(1000, min(s_idx, sample.shape[0] - self.window_length))
            
            # Enhanced windowing strategy dengan equal buffers sebelum P dan sesudah S
            # Target: buffer_sama sebelum P + P-S interval + buffer_sama sesudah S = 170 detik
            MIN_BUFFER = 1000  # Minimum 10 detik per sisi
            
            ps_interval = s_idx - p_idx
            
            # Hitung buffer yang terdistribusi merata di kedua sisi
            total_buffer_available = self.window_length - ps_interval
            buffer_each_side = max(MIN_BUFFER, total_buffer_available // 2)
            
            # Jika P-S interval sangat panjang, gunakan minimum buffer
            if ps_interval > (self.window_length - 2 * MIN_BUFFER):
                buffer_each_side = MIN_BUFFER
                print(f"Long P-S interval {ps_interval/100:.1f}s detected, using minimum buffer {MIN_BUFFER/100:.1f}s")
            
            # Strategy: Start buffer_each_side sebelum P-wave
            start_idx = p_idx - buffer_each_side
            
            # Check if S-wave + buffer fits within window
            s_wave_end_with_buffer = s_idx + buffer_each_side
            window_end = start_idx + self.window_length
            
            if s_wave_end_with_buffer > window_end:
                # S-wave + buffer doesn't fit, adjust strategy for very long intervals
                if ps_interval <= 15000:  # P-S ≤ 150 seconds, center pada P-S dengan equal buffers
                center = (p_idx + s_idx) // 2
                    start_idx = center - self.window_length // 2
                    
                    # Ensure minimum buffer before P
                    if (p_idx - start_idx) < MIN_BUFFER:
                        start_idx = p_idx - MIN_BUFFER
                else:
                    # Very long P-S interval (>150s), prioritize P-wave dengan minimum buffer
                    start_idx = p_idx - MIN_BUFFER
            
            # Ensure we don't exceed data bounds
            start_idx = max(0, start_idx)
            start_idx = min(start_idx, sample.shape[0] - self.window_length)
            end_idx = start_idx + self.window_length
            
            # Extract window
            sample = sample[start_idx:end_idx, :, :]
            
            # Adjust P and S indices relative to window
            p_idx_rel = p_idx - start_idx
            s_idx_rel = s_idx - start_idx
            
            # Ensure relative indices are within window
            p_idx_rel = max(0, min(p_idx_rel, self.window_length - 1))
            s_idx_rel = max(0, min(s_idx_rel, self.window_length - 1))
            
            # Create target labels
            target = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            
            # Background probability
            target[:, 0, 0] = 1.0
            
            # P-wave label (Gaussian)
            if p_idx_rel > 0:
                p_start = max(0, p_idx_rel - self.label_width)
                p_end = min(self.window_length, p_idx_rel + self.label_width)
                for j in range(p_start, p_end):
                    target[j, 0, 1] = np.exp(-((j - p_idx_rel) ** 2) / (2 * (self.label_width / 3) ** 2))
                    target[j, 0, 0] = 1.0 - target[j, 0, 1]
            
            # S-wave label (Gaussian)
            if s_idx_rel > 0 and s_idx_rel != p_idx_rel:
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
            sample = normalize_long(sample)
            
            # Return 3 values to match dataset definition: sample, target, filename
            return sample.astype(np.float32), target.astype(np.float32), base_name
            
        except Exception as e:
            print(f"Error loading {base_name}: {e}")
            # Return zero arrays as fallback
            sample = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target[:, 0, 0] = 1.0  # All background
            return sample, target, base_name

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=True, drop_remainder=True):
        """Create TensorFlow dataset"""
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

class DataReader_Indonesia_Test(DataReader):
    """Data Reader untuk testing dengan 99% coverage data Indonesia"""
    
    def __init__(self, format="numpy", config=DataConfig_Indonesia(), **kwargs):
        super().__init__(format=format, config=config, **kwargs)
        
        self.window_length = config.window_length
        self.select_range = [2000, 2000 + self.window_length]  # Start earlier untuk buffer
        
        print(f"Indonesia DataReader initialized:")
        print(f"  Window length: {self.window_length} samples ({self.window_length/100:.1f} seconds)")

    def __getitem__(self, i):
        base_name = str(self.data_list.iloc[i]).split("/")[-1]
        
        try:
            # Load NPZ data
            npz_data = np.load(os.path.join(self.data_dir, base_name), allow_pickle=True)
            
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
                    raise ValueError(f"No waveform data found in {base_name}")
            
            # Ensure sample is numpy array with correct shape
            if isinstance(sample, np.ndarray):
                if len(sample.shape) == 2:  # (time, channels)
                    sample = sample[:, np.newaxis, :]  # Add middle dimension
                elif len(sample.shape) == 1:  # (time,) - single channel
                    sample = sample[:, np.newaxis, np.newaxis]  # Add dimensions
            else:
                raise ValueError(f"Sample is not numpy array: {type(sample)}")
            
            p_idx = npz_data['p_idx'][0][0] if len(npz_data['p_idx']) > 0 and len(npz_data['p_idx'][0]) > 0 else 3000
            s_idx = npz_data['s_idx'][0][0] if len(npz_data['s_idx']) > 0 and len(npz_data['s_idx'][0]) > 0 else 6000
            
            # Ensure indices are within bounds
            p_idx = max(1000, min(p_idx, sample.shape[0] - self.window_length))  # Allow earlier start
            s_idx = max(1000, min(s_idx, sample.shape[0] - self.window_length))
            
            # Enhanced windowing strategy dengan equal buffers sebelum P dan sesudah S
            # Target: buffer_sama sebelum P + P-S interval + buffer_sama sesudah S = 170 detik
            MIN_BUFFER = 1000  # Minimum 10 detik per sisi
            
            ps_interval = s_idx - p_idx
            
            # Hitung buffer yang terdistribusi merata di kedua sisi
            total_buffer_available = self.window_length - ps_interval
            buffer_each_side = max(MIN_BUFFER, total_buffer_available // 2)
            
            # Jika P-S interval sangat panjang, gunakan minimum buffer
            if ps_interval > (self.window_length - 2 * MIN_BUFFER):
                buffer_each_side = MIN_BUFFER
                print(f"Long P-S interval {ps_interval/100:.1f}s detected, using minimum buffer {MIN_BUFFER/100:.1f}s")
            
            # Strategy: Start buffer_each_side sebelum P-wave
            start_idx = p_idx - buffer_each_side
            
            # Check if S-wave + buffer fits within window
            s_wave_end_with_buffer = s_idx + buffer_each_side
            window_end = start_idx + self.window_length
            
            if s_wave_end_with_buffer > window_end:
                # S-wave + buffer doesn't fit, adjust strategy for very long intervals
                if ps_interval <= 15000:  # P-S ≤ 150 seconds, center pada P-S dengan equal buffers
                center = (p_idx + s_idx) // 2
                    start_idx = center - self.window_length // 2
                    
                    # Ensure minimum buffer before P
                    if (p_idx - start_idx) < MIN_BUFFER:
                        start_idx = p_idx - MIN_BUFFER
                else:
                    # Very long P-S interval (>150s), prioritize P-wave dengan minimum buffer
                    start_idx = p_idx - MIN_BUFFER
            
            # Ensure we don't exceed data bounds
            start_idx = max(0, start_idx)
            start_idx = min(start_idx, sample.shape[0] - self.window_length)
            end_idx = start_idx + self.window_length
            
            # Extract window
            sample = sample[start_idx:end_idx, :, :]
            
            # Adjust indices relative to window
            p_idx_rel = p_idx - start_idx
            s_idx_rel = s_idx - start_idx
            
            # Ensure relative indices are within window
            p_idx_rel = max(0, min(p_idx_rel, self.window_length - 1))
            s_idx_rel = max(0, min(s_idx_rel, self.window_length - 1))
            
            # Create target (same as training)
            target = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target[:, 0, 0] = 1.0
            
            # P-wave label
            if p_idx_rel > 0:
                p_start = max(0, p_idx_rel - 30)
                p_end = min(self.window_length, p_idx_rel + 30)
                for j in range(p_start, p_end):
                    target[j, 0, 1] = np.exp(-((j - p_idx_rel) ** 2) / (2 * 10 ** 2))
                    target[j, 0, 0] = 1.0 - target[j, 0, 1]
            
            # S-wave label
            if s_idx_rel > 0 and s_idx_rel != p_idx_rel:
                s_start = max(0, s_idx_rel - 30)
                s_end = min(self.window_length, s_idx_rel + 30)
                for j in range(s_start, s_end):
                    target[j, 0, 2] = np.exp(-((j - s_idx_rel) ** 2) / (2 * 10 ** 2))
                    target[j, 0, 0] = 1.0 - target[j, 0, 1] - target[j, 0, 2]
            
            # Normalize target
            target_sum = np.sum(target, axis=2, keepdims=True)
            target_sum = np.where(target_sum == 0, 1, target_sum)
            target = target / target_sum
            
            # Normalize waveform
            sample = normalize_long(sample)
            
            return (sample.astype(np.float32), 
                   target.astype(np.float32), 
                   base_name, 
                   np.array([p_idx_rel], dtype=np.int32), 
                   np.array([s_idx_rel], dtype=np.int32))
            
        except Exception as e:
            print(f"Error loading {base_name}: {e}")
            # Return zero arrays as fallback
            sample = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target = np.zeros([self.window_length, 1, 3], dtype=np.float32)
            target[:, 0, 0] = 1.0
            return (sample, target, base_name, 
                   np.array([3000], dtype=np.int32), 
                   np.array([6000], dtype=np.int32))

    def dataset(self, batch_size, num_parallel_calls=2, shuffle=False, drop_remainder=False):
        """Create TensorFlow dataset for testing"""
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