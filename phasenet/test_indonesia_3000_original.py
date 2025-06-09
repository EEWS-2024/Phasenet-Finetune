#!/usr/bin/env python3
"""
Testing script untuk PhaseNet Original dengan data Indonesia
Model: 190703-214543 (NCEDC dataset - belum pernah melihat data Indonesia)
Purpose: Baseline comparison untuk finetuned models
"""

import argparse
import os
import sys

# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'

import tensorflow as tf
# Enable GPU usage with memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 GPU enabled - found {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"⚠️  GPU setup error: {e}")
else:
    print("🖥️  No GPU found - using CPU")

tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Add phasenet to path
sys.path.append(os.path.dirname(__file__))

from model import ModelConfig, UNet
from data_reader_indonesia_sliding import DataConfig_Indonesia_3000, DataReader_Indonesia_Sliding_Test
import json
import datetime
from tqdm import tqdm

def save_model_info(model_dir, output_dir):
    """Save information about the original model being tested"""
    info_file = os.path.join(output_dir, 'model_info.txt')
    
    with open(info_file, 'w') as f:
        f.write("=== ORIGINAL MODEL TESTING INFO ===\n\n")
        f.write(f"Model path: {model_dir}\n")
        f.write(f"Model type: Original PhaseNet (NCEDC dataset)\n")
        f.write(f"Test date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Purpose: Baseline comparison untuk finetuned models\n")
        f.write(f"Expected: Performance mungkin rendah (domain mismatch)\n\n")
        
        f.write("Model Details:\n")
        f.write("  - Pre-trained on NCEDC (Northern California) dataset\n")
        f.write("  - Window size: 3000 samples (30 seconds)\n")
        f.write("  - Never seen Indonesian seismic data before\n")
        f.write("  - Direct application without fine-tuning\n\n")
        
        # Check if model files exist
        checkpoint_file = tf.train.latest_checkpoint(model_dir)
        if checkpoint_file:
            f.write(f"Checkpoint: {checkpoint_file}\n")
            
            # List checkpoint variables
            try:
                checkpoint_vars = tf.train.list_variables(checkpoint_file)
                f.write(f"Total variables in checkpoint: {len(checkpoint_vars)}\n")
                f.write("\nCheckpoint variables (first 10):\n")
                for i, (name, shape) in enumerate(checkpoint_vars[:10]):
                    f.write(f"  {i+1}. {name} - Shape: {shape}\n")
                if len(checkpoint_vars) > 10:
                    f.write(f"  ... and {len(checkpoint_vars) - 10} more variables\n")
            except Exception as e:
                f.write(f"Could not read checkpoint variables: {e}\n")
        else:
            f.write("No checkpoint found in model directory\n")
    
    print(f"📋 Model info saved to: {info_file}")

def save_results(predictions, sliding_windows, output_dir, min_prob=0.3):
    """Save test results to CSV and create performance plots"""
    
    print(f"📊 Processing {len(predictions):,} predictions...")
    print(f"   First prediction shape: {predictions[0].shape}")
    print(f"   Using threshold: {min_prob} (higher threshold for original model)")
    
    # Filter only windows that contain P or S arrivals
    valid_windows_data = []
    
    for i, (pred, window) in enumerate(zip(predictions, sliding_windows)):
        filename = window['npz_file']
        start_idx = window['window_start']
        end_idx = window['window_end']
        p_arrival = window['original_p_idx']
        s_arrival = window['original_s_idx']
        
        # Check if this window contains P or S arrival
        has_p_in_window = p_arrival is not None and start_idx <= p_arrival < end_idx
        has_s_in_window = s_arrival is not None and start_idx <= s_arrival < end_idx
        
        # Only process windows that contain P or S arrivals
        if not (has_p_in_window or has_s_in_window):
            continue
            
        # Calculate relative positions within window
        p_arrival_relative = p_arrival - start_idx if has_p_in_window else None
        s_arrival_relative = s_arrival - start_idx if has_s_in_window else None
        
        valid_windows_data.append({
            'prediction': pred,
            'window': window,
            'has_p': has_p_in_window,
            'has_s': has_s_in_window,
            'p_arrival_relative': p_arrival_relative,
            's_arrival_relative': s_arrival_relative,
            'filename': filename,
            'start_idx': start_idx,
            'end_idx': end_idx
        })
    
    print(f"📊 Found {len(valid_windows_data)} windows containing P/S arrivals out of {len(predictions)} total windows")
    print(f"   Filtering ratio: {len(valid_windows_data)/len(predictions)*100:.1f}%")
    
    # Process only valid windows
    results = []
    
    def find_local_maxima(prob_array, min_prob, min_distance=50):
        """Find local maxima in probability array with minimum distance constraint"""
        from scipy.signal import find_peaks
        
        # Find peaks above threshold with minimum distance
        peaks, properties = find_peaks(prob_array, 
                                     height=min_prob, 
                                     distance=min_distance)
        return peaks
    
    for i, window_data in enumerate(valid_windows_data):
        pred = window_data['prediction']
        filename = window_data['filename']
        start_idx = window_data['start_idx']
        end_idx = window_data['end_idx']
        has_p = window_data['has_p']
        has_s = window_data['has_s']
        p_arrival_relative = window_data['p_arrival_relative']
        s_arrival_relative = window_data['s_arrival_relative']
        
        # Convert predictions to probabilities
        if isinstance(pred, np.ndarray):
            if len(pred.shape) == 3 and pred.shape[1] == 1:
                pred = pred.squeeze(axis=1)
            elif len(pred.shape) == 2 and pred.shape[1] == 1:
                print(f"Warning: Unexpected prediction shape {pred.shape} for window {i}")
                continue
                
            # Apply softmax manually
            exp_pred = np.exp(pred - np.max(pred, axis=-1, keepdims=True))
            prob_pred = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
        else:
            prob_pred = tf.nn.softmax(pred, axis=-1).numpy()
        
        if prob_pred.shape[-1] != 3:
            print(f"Warning: Expected 3 channels, got {prob_pred.shape[-1]} for window {i}")
            continue
        
        # Extract probabilities
        background_prob = prob_pred[:, 0]  # Background
        p_prob = prob_pred[:, 1]  # P wave
        s_prob = prob_pred[:, 2]  # S wave
        
        # Find local maxima as detections (not all points above threshold)
        try:
            p_detections = find_local_maxima(p_prob, min_prob, min_distance=50)
            s_detections = find_local_maxima(s_prob, min_prob, min_distance=50)
        except:
            # Fallback to simple threshold if scipy not available
            p_above_thresh = np.where(p_prob > min_prob)[0]
            s_above_thresh = np.where(s_prob > min_prob)[0]
            
            # Take only local maxima manually
            p_detections = []
            for idx in p_above_thresh:
                if (idx == 0 or p_prob[idx] > p_prob[idx-1]) and \
                   (idx == len(p_prob)-1 or p_prob[idx] > p_prob[idx+1]):
                    p_detections.append(idx)
            
            s_detections = []
            for idx in s_above_thresh:
                if (idx == 0 or s_prob[idx] > s_prob[idx-1]) and \
                   (idx == len(s_prob)-1 or s_prob[idx] > s_prob[idx+1]):
                    s_detections.append(idx)
            
            p_detections = np.array(p_detections)
            s_detections = np.array(s_detections)
        
        # Get maximum probabilities and their indices (relative to window)
        max_p_prob = np.max(p_prob)
        max_s_prob = np.max(s_prob)
        max_p_idx_relative = np.argmax(p_prob)
        max_s_idx_relative = np.argmax(s_prob)
        
        # Calculate prediction errors if ground truth is available
        p_error_samples = None
        s_error_samples = None
        p_detected_correctly = False
        s_detected_correctly = False
        
        if has_p and p_arrival_relative is not None:
            p_error_samples = abs(max_p_idx_relative - p_arrival_relative)
            # Consider detection correct if error < 50 samples (0.5 seconds)
            p_detected_correctly = p_error_samples < 50 and max_p_prob > min_prob
            
        if has_s and s_arrival_relative is not None:
            s_error_samples = abs(max_s_idx_relative - s_arrival_relative)
            # Consider detection correct if error < 50 samples (0.5 seconds)
            s_detected_correctly = s_error_samples < 50 and max_s_prob > min_prob
        
        # Limit detection indices to first 10 for readability
        p_detection_list = p_detections.tolist()[:10] if len(p_detections) > 0 else []
        s_detection_list = s_detections.tolist()[:10] if len(s_detections) > 0 else []
        
        # Store comprehensive results
        result = {
            'filename': filename,
            'window_start': start_idx,
            'window_end': end_idx,
            'window_length': end_idx - start_idx,
            
            # Ground truth information (relative to window)
            'has_p_arrival': has_p,
            'has_s_arrival': has_s,
            'p_arrival_relative_idx': p_arrival_relative,
            's_arrival_relative_idx': s_arrival_relative,
            
            # Predictions (relative to window)
            'max_p_prob': max_p_prob,
            'max_s_prob': max_s_prob,
            'max_p_idx_relative': max_p_idx_relative,
            'max_s_idx_relative': max_s_idx_relative,
            
            # Detection results (using local maxima)
            'p_detected': max_p_prob > min_prob,
            's_detected': max_s_prob > min_prob,
            'p_detections_count': len(p_detections),
            's_detections_count': len(s_detections),
            
            # Accuracy metrics
            'p_error_samples': p_error_samples,
            's_error_samples': s_error_samples,
            'p_detected_correctly': p_detected_correctly,
            's_detected_correctly': s_detected_correctly,
            
            # Detection info (limited to first 10)
            'p_detection_indices': p_detection_list,
            's_detection_indices': s_detection_list,
            
            # Confidence levels
            'p_confidence_category': 'high' if max_p_prob > 0.8 else ('medium' if max_p_prob > 0.5 else 'low'),
            's_confidence_category': 'high' if max_s_prob > 0.8 else ('medium' if max_s_prob > 0.5 else 'low')
        }
        
        results.append(result)
        
        # Show progress
        if (i + 1) % 500 == 0 or i < 5:  # Show first few and every 500
            print(f"   Window {i+1}: P_prob={max_p_prob:.3f} S_prob={max_s_prob:.3f} P_det={len(p_detections)} S_det={len(s_detections)}")
    
    print(f"📊 Processed {len(results)} valid windows with P/S arrivals")
    
    # Save to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'sliding_window_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"📊 Results saved to: {csv_path}")
    
    # Create performance plots
    create_performance_plots(results_df, output_dir, min_prob)
    
    return results_df

def create_performance_plots(results_df, output_dir, min_prob):
    """Create comprehensive performance plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Original PhaseNet Model - Testing Results (min_prob={min_prob})', fontsize=16, fontweight='bold')
    
    # 1. Probability distributions
    axes[0, 0].hist(results_df['max_p_prob'], bins=50, alpha=0.7, label='P-wave', color='blue')
    axes[0, 0].hist(results_df['max_s_prob'], bins=50, alpha=0.7, label='S-wave', color='red')
    axes[0, 0].axvline(min_prob, color='black', linestyle='--', label=f'Threshold ({min_prob})')
    axes[0, 0].set_xlabel('Maximum Probability')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Probability Distributions')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Detection rates
    p_detected = results_df['p_detected'].sum()
    s_detected = results_df['s_detected'].sum()
    total_windows = len(results_df)
    
    detection_data = [p_detected, s_detected, total_windows - p_detected, total_windows - s_detected]
    detection_labels = ['P Detected', 'S Detected', 'P Not Detected', 'S Not Detected']
    colors = ['blue', 'red', 'lightblue', 'lightcoral']
    
    axes[0, 1].bar(range(4), detection_data, color=colors)
    axes[0, 1].set_xticks(range(4))
    axes[0, 1].set_xticklabels(detection_labels, rotation=45)
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(f'Detection Counts (Total: {total_windows} windows)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, v in enumerate(detection_data):
        percentage = (v / total_windows) * 100
        axes[0, 1].text(i, v + total_windows*0.01, f'{percentage:.1f}%', ha='center', va='bottom')
    
    # 3. Error distribution (only for windows with ground truth)
    p_errors = results_df['p_error_samples'].dropna()
    s_errors = results_df['s_error_samples'].dropna()
    
    if len(p_errors) > 0:
        axes[0, 2].hist(p_errors, bins=30, alpha=0.7, label=f'P-wave (n={len(p_errors)})', color='blue')
    if len(s_errors) > 0:
        axes[0, 2].hist(s_errors, bins=30, alpha=0.7, label=f'S-wave (n={len(s_errors)})', color='red')
    
    axes[0, 2].set_xlabel('Prediction Error (samples)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Prediction Errors (Ground Truth vs Predicted)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Probability vs Error scatter plot
    if len(p_errors) > 0:
        p_probs_with_truth = results_df.loc[results_df['p_error_samples'].notna(), 'max_p_prob']
        axes[1, 0].scatter(p_probs_with_truth, p_errors, alpha=0.6, color='blue', label='P-wave', s=20)
    
    if len(s_errors) > 0:
        s_probs_with_truth = results_df.loc[results_df['s_error_samples'].notna(), 'max_s_prob']
        axes[1, 0].scatter(s_probs_with_truth, s_errors, alpha=0.6, color='red', label='S-wave', s=20)
    
    axes[1, 0].set_xlabel('Max Probability')
    axes[1, 0].set_ylabel('Prediction Error (samples)')
    axes[1, 0].set_title('Confidence vs Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Detection summary by file
    file_summary = results_df.groupby('filename').agg({
        'p_detected': 'sum',
        's_detected': 'sum',
        'max_p_prob': 'mean',
        'max_s_prob': 'mean'
    }).reset_index()
    
    if len(file_summary) <= 20:  # Only show if not too many files
        x_pos = range(len(file_summary))
        axes[1, 1].bar([x - 0.2 for x in x_pos], file_summary['p_detected'], width=0.4, 
                      label='P detections', color='blue', alpha=0.7)
        axes[1, 1].bar([x + 0.2 for x in x_pos], file_summary['s_detected'], width=0.4, 
                      label='S detections', color='red', alpha=0.7)
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels([os.path.basename(f)[:10] + '...' for f in file_summary['filename']], 
                                  rotation=45, fontsize=8)
        axes[1, 1].set_ylabel('Detection Count')
        axes[1, 1].set_title('Detections per File')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, f'Too many files to display\n({len(file_summary)} files)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Detections per File (Too many to show)')
    
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Overall statistics
    stats_text = f"""ORIGINAL MODEL PERFORMANCE
    
Total Windows: {total_windows:,}
P-wave Detection Rate: {(p_detected/total_windows)*100:.1f}%
S-wave Detection Rate: {(s_detected/total_windows)*100:.1f}%

Average Probabilities:
P-wave: {results_df['max_p_prob'].mean():.3f} ± {results_df['max_p_prob'].std():.3f}
S-wave: {results_df['max_s_prob'].mean():.3f} ± {results_df['max_s_prob'].std():.3f}

Prediction Errors (if ground truth available):
P-wave MAE: {p_errors.mean():.1f} ± {p_errors.std():.1f} samples
S-wave MAE: {s_errors.mean():.1f} ± {s_errors.std():.1f} samples

Model: Original PhaseNet (NCEDC)
Expected: Lower performance due to domain mismatch
Use for: Baseline comparison with finetuned models"""
    
    axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'sliding_window_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Performance plots saved to: {plot_path}")

def load_original_model_compatible(sess, model_dir):
    """Load original model with compatibility handling for dtype mismatch"""
    checkpoint = tf.train.latest_checkpoint(model_dir)
    if not checkpoint:
        print(f"❌ No checkpoint found in {model_dir}")
        return False
    
    print(f"Loading checkpoint: {checkpoint}")
    
    try:
        # First, try normal restore
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, checkpoint)
        print(f"✅ Original model loaded successfully (normal restore)")
        return True
        
    except tf.errors.InvalidArgumentError as e:
        if "global_step" in str(e) and "dtype" in str(e):
            print(f"⚠️  Dtype mismatch detected for global_step, trying selective restore...")
            
            # Get all variables except those causing problems
            all_vars = tf.compat.v1.global_variables()
            
            # Filter out problematic variables
            vars_to_restore = []
            skip_patterns = ['global_step', 'Adam', 'adam', 'beta1_power', 'beta2_power']
            
            for var in all_vars:
                var_name = var.name
                should_skip = any(pattern in var_name for pattern in skip_patterns)
                
                if not should_skip:
                    vars_to_restore.append(var)
            
            print(f"   Restoring {len(vars_to_restore)} variables (skipping {len(all_vars) - len(vars_to_restore)} problematic)")
            
            # Create selective saver
            selective_saver = tf.compat.v1.train.Saver(var_list=vars_to_restore)
            selective_saver.restore(sess, checkpoint)
            print(f"✅ Original model loaded successfully (selective restore)")
            return True
            
        else:
            print(f"❌ Failed to load original model: {e}")
            return False
    
    except Exception as e:
        print(f"❌ Unexpected error loading model: {e}")
        return False

def test_fn(args, data_reader_test):
    """Testing function untuk original model"""
    
    print(f"🎯 Testing original model: {args.model_dir}")
    print(f"📊 Test windows: {len(data_reader_test.sliding_windows):,}")
    
    # Display data statistics
    print(f"\n📈 DATA STATISTICS:")
    print(f"   Total NPZ files: {len(data_reader_test.data_list)}")
    print(f"   Total sliding windows: {len(data_reader_test.sliding_windows):,}")
    print(f"   Window size: {data_reader_test.config.X_shape[0]} samples ({data_reader_test.config.X_shape[0]/100:.1f} seconds)")
    print(f"   Overlap: No overlap for testing (discrete windows)")
    
    # Show sample distribution
    files_with_windows = {}
    for window in data_reader_test.sliding_windows:
        filename = window['npz_file']  # Changed from 'filename' to 'npz_file'
        if filename not in files_with_windows:
            files_with_windows[filename] = 0
        files_with_windows[filename] += 1
    
    windows_per_file = list(files_with_windows.values())
    print(f"   Average windows per file: {np.mean(windows_per_file):.1f}")
    print(f"   Min windows per file: {min(windows_per_file)}")
    print(f"   Max windows per file: {max(windows_per_file)}")
    
    # Show first few files as example
    print(f"\n📁 SAMPLE FILES (first 5):")
    for i, (filename, count) in enumerate(list(files_with_windows.items())[:5]):
        print(f"   {i+1}. {filename}: {count} windows")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model info
    save_model_info(args.model_dir, args.output_dir)
    
    # Create model config (same as original)
    config = ModelConfig(
        X_shape=data_reader_test.config.X_shape,  # [3000, 1, 3]
        Y_shape=data_reader_test.config.Y_shape,  # [3000, 1, 3]
        n_channel=data_reader_test.config.n_channel,
        n_class=data_reader_test.config.n_class,
        sampling_rate=data_reader_test.config.sampling_rate,
        dt=data_reader_test.config.dt,
        use_batch_norm=True,
        use_dropout=True,
        drop_rate=0.0,  # No dropout during testing
        class_weights=[1.0, 1.0, 1.0]
    )
    
    # Create dataset
    test_dataset = data_reader_test.dataset(args.batch_size, shuffle=False, drop_remainder=False)
    
    # Create placeholders
    X_placeholder = tf.compat.v1.placeholder(tf.float32, [None] + data_reader_test.config.X_shape, name='X_input')
    Y_placeholder = tf.compat.v1.placeholder(tf.float32, [None] + data_reader_test.config.Y_shape, name='Y_target')
    fname_placeholder = tf.compat.v1.placeholder(tf.string, [None], name='fname_input')
    
    # Create model
    model = UNet(config=config, input_batch=(X_placeholder, Y_placeholder, fname_placeholder), mode='test')
    
    # Configure for GPU/CPU
    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.allow_soft_placement = True
    gpu_config.log_device_placement = False
    
    if gpus:
        gpu_config.gpu_options.allow_growth = True
        gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.6  # Conservative for testing
        print(f"🚀 Using GPU for testing")
    else:
        gpu_config.device_count = {'GPU': 0}
        print(f"🖥️  Using CPU for testing")
    
    with tf.compat.v1.Session(config=gpu_config) as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Load original model with compatibility handling
        if not load_original_model_compatible(sess, args.model_dir):
            print("❌ Failed to load original model. Exiting.")
            return
        
        # Run testing
        print(f"\n🧪 Running inference on {len(data_reader_test.sliding_windows):,} windows...")
        
        predictions = []
        
        # Create dataset iterator
        test_iterator = tf.compat.v1.data.make_initializable_iterator(test_dataset)
        next_test_batch = test_iterator.get_next()
        sess.run(test_iterator.initializer)
        
        # Calculate total steps
        total_steps = int(np.ceil(len(data_reader_test.sliding_windows) / args.batch_size))
        
        # Run inference
        try:
            with tqdm(total=total_steps, desc="Testing Original Model", unit="batch") as pbar:
                for step in range(total_steps):
                    try:
                        # Get batch - data reader returns (X, Y, fname, p_idx, s_idx)
                        batch_data = sess.run(next_test_batch)
                        if len(batch_data) == 5:
                            X_batch, Y_batch, fname_batch, p_idx_batch, s_idx_batch = batch_data
                        else:
                            X_batch, Y_batch, fname_batch = batch_data[:3]
                        
                        # Run inference
                        feed_dict = {
                            X_placeholder: X_batch,
                            Y_placeholder: Y_batch,
                            fname_placeholder: fname_batch,
                            model.drop_rate: 0.0,  # No dropout
                            model.is_training: False
                        }
                        
                        # Get predictions
                        batch_predictions = sess.run(model.preds, feed_dict=feed_dict)
                        predictions.extend(batch_predictions)
                        
                        # Update progress bar
                        pbar.set_description(f"Testing Original Model - Step {step+1}/{total_steps}")
                        pbar.update(1)
                        
                    except tf.errors.OutOfRangeError:
                        break
                        
        except Exception as e:
            print(f"⚠️  Testing error: {e}")
            return
        
        print(f"✅ Inference completed - {len(predictions):,} predictions generated")
        
        # Verify predictions count matches windows count
        if len(predictions) != len(data_reader_test.sliding_windows):
            print(f"⚠️  Warning: Predictions count ({len(predictions)}) != Windows count ({len(data_reader_test.sliding_windows)})")
        
        # Save results and create plots
        if args.plot_results:
            print(f"📊 Saving results and creating plots...")
            results_df = save_results(predictions, data_reader_test.sliding_windows, 
                                    args.output_dir, args.min_prob)
            
            # Print summary statistics
            total_windows = len(results_df)
            p_windows = results_df['has_p_arrival'].sum()
            s_windows = results_df['has_s_arrival'].sum()
            p_detected = results_df['p_detected'].sum()
            s_detected = results_df['s_detected'].sum()
            p_correct = results_df['p_detected_correctly'].sum()
            s_correct = results_df['s_detected_correctly'].sum()
            
            print(f"\n📊 TESTING SUMMARY (Only windows with P/S arrivals):")
            print(f"   Total files tested: {len(data_reader_test.data_list)}")
            print(f"   Total windows with P/S arrivals: {total_windows:,}")
            print(f"   Windows with P arrivals: {p_windows:,}")
            print(f"   Windows with S arrivals: {s_windows:,}")
            print(f"")
            print(f"   P-wave detections: {p_detected:,}/{p_windows:,} ({(p_detected/p_windows)*100:.1f}%)")
            print(f"   S-wave detections: {s_detected:,}/{s_windows:,} ({(s_detected/s_windows)*100:.1f}%)")
            print(f"   P-wave correct detections: {p_correct:,}/{p_windows:,} ({(p_correct/p_windows)*100:.1f}%)")
            print(f"   S-wave correct detections: {s_correct:,}/{s_windows:,} ({(s_correct/s_windows)*100:.1f}%)")
            print(f"")
            print(f"   Average P probability: {results_df['max_p_prob'].mean():.3f}")
            print(f"   Average S probability: {results_df['max_s_prob'].mean():.3f}")
            
            # Error statistics for windows with ground truth
            p_errors = results_df[results_df['has_p_arrival']]['p_error_samples'].dropna()
            s_errors = results_df[results_df['has_s_arrival']]['s_error_samples'].dropna()
            
            if len(p_errors) > 0:
                print(f"   P-wave MAE: {p_errors.mean():.1f} ± {p_errors.std():.1f} samples ({p_errors.mean()/100:.2f}s)")
            if len(s_errors) > 0:
                print(f"   S-wave MAE: {s_errors.mean():.1f} ± {s_errors.std():.1f} samples ({s_errors.mean()/100:.2f}s)")
            
            # Confidence distribution
            p_high_conf = (results_df['p_confidence_category'] == 'high').sum()
            p_med_conf = (results_df['p_confidence_category'] == 'medium').sum()
            p_low_conf = (results_df['p_confidence_category'] == 'low').sum()
            
            s_high_conf = (results_df['s_confidence_category'] == 'high').sum()
            s_med_conf = (results_df['s_confidence_category'] == 'medium').sum()
            s_low_conf = (results_df['s_confidence_category'] == 'low').sum()
            
            print(f"\n📈 CONFIDENCE DISTRIBUTION:")
            print(f"   P-wave: High({p_high_conf}) Medium({p_med_conf}) Low({p_low_conf})")
            print(f"   S-wave: High({s_high_conf}) Medium({s_med_conf}) Low({s_low_conf})")
            
            print(f"\n💡 NOTE: Ini adalah model original (NCEDC) yang belum pernah melihat data Indonesia")
            print(f"    Analysis hanya pada windows yang mengandung P/S arrivals")
            print(f"    Semua index adalah relatif terhadap window (0-2999 samples)")
            print(f"    Deteksi dianggap 'correct' jika error < 50 samples (0.5 detik)")
            print(f"    Performance ini adalah baseline untuk comparison dengan model finetuned!")

def main():
    parser = argparse.ArgumentParser(description='Test Original PhaseNet dengan Data Indonesia')
    
    # Data parameters
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--test_list', type=str, required=True, help='Test data list CSV')
    parser.add_argument('--format', type=str, default='numpy', help='Data format')
    
    # Model parameters
    parser.add_argument('--model_dir', type=str, required=True, help='Original model directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    # Testing parameters
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for testing')
    parser.add_argument('--min_prob', type=float, default=0.3, help='Minimum probability threshold')
    
    # Output options
    parser.add_argument('--plot_results', action='store_true', help='Create performance plots')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.test_dir):
        print(f"❌ Test directory not found: {args.test_dir}")
        return
    
    if not os.path.exists(args.test_list):
        print(f"❌ Test list not found: {args.test_list}")
        return
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        return
    
    print("🧪 Testing Original PhaseNet dengan Data Indonesia")
    print("=" * 60)
    print(f"Model: {args.model_dir} (Original NCEDC)")
    print(f"Test dir: {args.test_dir}")
    print(f"Test list: {args.test_list}")
    print(f"Output dir: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Min prob: {args.min_prob}")
    print(f"Plot results: {args.plot_results}")
    print("=" * 60)
    
    # Create data reader
    data_config = DataConfig_Indonesia_3000()
    
    print("Loading test data reader...")
    data_reader_test = DataReader_Indonesia_Sliding_Test(
        data_dir=args.test_dir,
        data_list=args.test_list,
        config=data_config,
        format=args.format
    )
    
    # Start testing
    test_fn(args, data_reader_test)

if __name__ == '__main__':
    main() 