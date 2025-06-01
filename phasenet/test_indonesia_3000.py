#!/usr/bin/env python3
"""
Testing script untuk PhaseNet Indonesia dengan Sliding Window 3000 samples
Compatible dengan model pretrained dan fine-tuning sliding window strategy.

Window size: 3000 samples (30 detik) dengan sliding window overlap strategy
"""

import argparse
import os
import sys
import time

# Set environment variables BEFORE importing TensorFlow to prevent XLA issues
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0=all, 1=info+, 2=warning+, 3=error+
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent GPU memory allocation issues
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'  # Disable XLA to avoid libdevice issues
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only first GPU

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Disable verbose warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Add phasenet to path
sys.path.append(os.path.dirname(__file__))

from model import ModelConfig, UNet
from data_reader_indonesia_sliding import DataConfig_Indonesia_3000, DataReader_Indonesia_Sliding_Test
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import json

# Simple utility function to replace missing import
def LoadConfig(model_dir):
    """Load model configuration from JSON file"""
    config_file = os.path.join(model_dir, 'config.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return config_dict  # Return dict directly
    else:
        return {}  # Return empty dict if no config file

def find_latest_model_dir(base_model_dir):
    """Find the latest model directory"""
    if not os.path.exists(base_model_dir):
        return None
    
    subdirs = [d for d in os.listdir(base_model_dir) 
               if os.path.isdir(os.path.join(base_model_dir, d))]
    
    if not subdirs:
        return None
    
    # Sort by modification time, latest first
    subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(base_model_dir, x)), reverse=True)
    
    return os.path.join(base_model_dir, subdirs[0])

def test_fn(args, data_reader):
    """Test function for Indonesia sliding window model"""
    
    # Determine model directory to use
    if args.model_path:
        model_dir_to_use = os.path.dirname(args.model_path)
    else:
        # Check if args.model_dir is already a specific model directory
        try:
            if os.path.exists(os.path.join(args.model_dir, 'checkpoint')) or \
               os.path.exists(os.path.join(args.model_dir, 'final_model.ckpt.index')) or \
               any(f.endswith('.ckpt.index') for f in os.listdir(args.model_dir) if os.path.isfile(os.path.join(args.model_dir, f))):
                # args.model_dir is already a specific model directory
                model_dir_to_use = args.model_dir
                print(f"Using provided model directory: {model_dir_to_use}")
            else:
                # args.model_dir is a base directory, find latest subdirectory
                model_dir_to_use = find_latest_model_dir(args.model_dir)
                if not model_dir_to_use:
                    print(f"No model directory found in {args.model_dir}")
                    return None
                print(f"Found latest model directory: {model_dir_to_use}")
        except (OSError, PermissionError) as e:
            print(f"Error accessing model directory {args.model_dir}: {e}")
            return None
    
    print(f"Using model directory: {model_dir_to_use}")
    
    # Load model configuration
    config_dict = LoadConfig(model_dir_to_use)
    print(f"Loaded config: {config_dict}")
    
    # Create model config with sliding window 3000 defaults
    config = ModelConfig(
        X_shape=data_reader.config.X_shape,
        Y_shape=data_reader.config.Y_shape,
        n_channel=data_reader.config.n_channel,
        n_class=data_reader.config.n_class,
        sampling_rate=data_reader.config.sampling_rate,
        dt=data_reader.config.dt,
        use_batch_norm=config_dict.get('use_batch_norm', True) if isinstance(config_dict, dict) else True,
        use_dropout=config_dict.get('use_dropout', True) if isinstance(config_dict, dict) else True,
        drop_rate=config_dict.get('drop_rate', 0.05) if isinstance(config_dict, dict) else 0.05,
        optimizer=config_dict.get('optimizer', 'adam') if isinstance(config_dict, dict) else 'adam',
        learning_rate=config_dict.get('learning_rate', 0.00001) if isinstance(config_dict, dict) else 0.00001,
        decay_step=config_dict.get('decay_step', 10) if isinstance(config_dict, dict) else 10,
        decay_rate=config_dict.get('decay_rate', 0.98) if isinstance(config_dict, dict) else 0.98,
        batch_size=config_dict.get('batch_size', 128) if isinstance(config_dict, dict) else 128,
        epochs=config_dict.get('epochs', 50) if isinstance(config_dict, dict) else 50,
        loss_type=config_dict.get('loss_type', 'cross_entropy') if isinstance(config_dict, dict) else 'cross_entropy',
        weight_decay=config_dict.get('weight_decay', 0.0001) if isinstance(config_dict, dict) else 0.0001,
        summary=config_dict.get('summary', True) if isinstance(config_dict, dict) else True,
        class_weights=config_dict.get('class_weights', [1.0, 1.0, 1.0]) if isinstance(config_dict, dict) else [1.0, 1.0, 1.0]
    )
    
    # Create dataset
    dataset = data_reader.dataset(args.batch_size, shuffle=False, drop_remainder=False)
    batch = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    
    # Create model
    model = UNet(config=config, input_batch=batch, mode='test')
    
    # Session config
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.log_device_placement = False
    sess_config.allow_soft_placement = True
    
    results = []
    
    with tf.compat.v1.Session(config=sess_config) as sess:
        # Load model weights
        saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
        
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = tf.train.latest_checkpoint(model_dir_to_use)
        
        if not model_path:
            print(f"No checkpoint found in {model_dir_to_use}")
            return None
            
        print(f"Loading model from: {model_path}")
        saver.restore(sess, model_path)
        
        # Test loop
        batch_count = 0
        total_samples = 0
        window_results = []
        prob_stats = {'P_max': [], 'S_max': [], 'P_mean': [], 'S_mean': []}
        
        print(f"Starting testing with sliding window 3000 strategy...")
        
        # Define thresholds for detection
        min_threshold = args.min_prob
        med_threshold = min(min_threshold + 0.1, 0.9)
        high_threshold = min(min_threshold + 0.2, 0.9)
        
        print(f"Detection thresholds: {min_threshold:.2f} / {med_threshold:.2f} / {high_threshold:.2f}")
        
        try:
            while True:
                # Prepare feed_dict for testing mode
                feed_dict = {
                    model.is_training: False,  # Testing mode
                    model.drop_rate: 0.0       # No dropout during testing
                }
                
                # Get predictions with proper feed_dict
                sample_batch, target_batch, fname_batch, p_true_batch, s_true_batch, pred_batch = sess.run([
                    batch[0], batch[1], batch[2], batch[3], batch[4], model.preds
                ], feed_dict=feed_dict)
                
                batch_size_actual = sample_batch.shape[0]
                total_samples += batch_size_actual
                
                # Process each sample in the batch
                for i in range(batch_size_actual):
                    fname = fname_batch[i].decode('utf-8') if isinstance(fname_batch[i], bytes) else str(fname_batch[i])
                    p_true = float(p_true_batch[i])
                    s_true = float(s_true_batch[i])
                    
                    # Get predictions (shape: [3000, 3] - Background, P, S)
                    preds = pred_batch[i]  # Shape: [3000, 3]
                    p_prob = preds[:, 1]   # P-wave probabilities
                    s_prob = preds[:, 2]   # S-wave probabilities
                    
                    # Find peaks for P and S waves
                    p_peaks, _ = find_peaks(p_prob, height=min_threshold, distance=10)
                    s_peaks, _ = find_peaks(s_prob, height=min_threshold, distance=10)
                    
                    # Get best picks
                    p_pred_idx = p_peaks[np.argmax(p_prob[p_peaks])] if len(p_peaks) > 0 else -1
                    s_pred_idx = s_peaks[np.argmax(s_prob[s_peaks])] if len(s_peaks) > 0 else -1
                    
                    p_pred_prob = p_prob[p_pred_idx] if p_pred_idx >= 0 else 0.0
                    s_pred_prob = s_prob[s_pred_idx] if s_pred_idx >= 0 else 0.0
                    
                    # Store probability statistics
                    prob_stats['P_max'].append(np.max(p_prob))
                    prob_stats['S_max'].append(np.max(s_prob))
                    prob_stats['P_mean'].append(np.mean(p_prob))
                    prob_stats['S_mean'].append(np.mean(s_prob))
                    
                    # Calculate errors (if ground truth is available)
                    p_error = abs(p_pred_idx - p_true) if p_pred_idx >= 0 and p_true >= 0 else -1
                    s_error = abs(s_pred_idx - s_true) if s_pred_idx >= 0 and s_true >= 0 else -1
                    
                    # Store results
                    window_result = {
                        'fname': fname,
                        'p_true': p_true,
                        's_true': s_true,
                        'p_pred': p_pred_idx,
                        's_pred': s_pred_idx,
                        'p_prob': p_pred_prob,
                        's_prob': s_pred_prob,
                        'p_error': p_error,
                        's_error': s_error,
                        'p_detected': p_pred_idx >= 0 and p_pred_prob >= min_threshold,
                        's_detected': s_pred_idx >= 0 and s_pred_prob >= min_threshold,
                        'window_idx': total_samples - batch_size_actual + i
                    }
                    window_results.append(window_result)
                
                # Show progress every 50 batches
                if (batch_count + 1) % 50 == 0:
                    detected_p = sum(1 for r in window_results if r['p_detected'])
                    detected_s = sum(1 for r in window_results if r['s_detected'])
                    detected_both = sum(1 for r in window_results if r['p_detected'] and r['s_detected'])
                    
                    p_rate = detected_p / len(window_results) * 100
                    s_rate = detected_s / len(window_results) * 100
                    ps_rate = detected_both / len(window_results) * 100
                    
                    # Show probability statistics
                    avg_p_max = np.mean(prob_stats['P_max'][-200:]) if prob_stats['P_max'] else 0
                    avg_s_max = np.mean(prob_stats['S_max'][-200:]) if prob_stats['S_max'] else 0
                    
                    print(f"Batch {batch_count + 1}: {len(window_results)} windows | P: {p_rate:.1f}% | S: {s_rate:.1f}% | PS: {ps_rate:.1f}%")
                    print(f"   Avg Prob - P_max: {avg_p_max:.3f}, S_max: {avg_s_max:.3f}")
                
                batch_count += 1
                
        except tf.errors.OutOfRangeError:
            print(f"Testing completed. Processed {len(window_results)} sliding windows.")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(window_results)
        
        # Calculate overall statistics
        if len(results_df) > 0:
            print(f"\nSLIDING WINDOW TESTING RESULTS:")
            print(f"   Total windows tested: {len(results_df)}")
            
            # Detection rates
            p_detection_rate = (results_df['p_detected'].sum() / len(results_df)) * 100
            s_detection_rate = (results_df['s_detected'].sum() / len(results_df)) * 100
            both_detection_rate = ((results_df['p_detected'] & results_df['s_detected']).sum() / len(results_df)) * 100
            
            print(f"   P-wave detection rate: {p_detection_rate:.1f}%")
            print(f"   S-wave detection rate: {s_detection_rate:.1f}%")
            print(f"   Both P&S detection rate: {both_detection_rate:.1f}%")
            
            # Probability statistics
            print(f"\nPROBABILITY STATISTICS:")
            print(f"   P-wave prob - Mean: {results_df['p_prob'].mean():.3f}, Max: {results_df['p_prob'].max():.3f}")
            print(f"   S-wave prob - Mean: {results_df['s_prob'].mean():.3f}, Max: {results_df['s_prob'].max():.3f}")
            
            # Error statistics (if ground truth available)
            valid_p_errors = results_df[results_df['p_error'] >= 0]['p_error']
            valid_s_errors = results_df[results_df['s_error'] >= 0]['s_error']
            
            if len(valid_p_errors) > 0:
                print(f"\nACCURACY STATISTICS:")
                print(f"   P-wave error - Mean: {valid_p_errors.mean():.1f} samples, Median: {valid_p_errors.median():.1f}")
                print(f"   S-wave error - Mean: {valid_s_errors.mean():.1f} samples, Median: {valid_s_errors.median():.1f}")
        
        return results_df

def create_performance_plots(results_df, output_dir):
    """Create performance plots for sliding window results"""
    if results_df is None or len(results_df) == 0:
        print("No results to plot")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Probability distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # P-wave probabilities
    axes[0, 0].hist(results_df['p_prob'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('P-wave Probability Distribution')
    axes[0, 0].set_xlabel('Probability')
    axes[0, 0].set_ylabel('Count')
    
    # S-wave probabilities  
    axes[0, 1].hist(results_df['s_prob'], bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_title('S-wave Probability Distribution')
    axes[0, 1].set_xlabel('Probability')
    axes[0, 1].set_ylabel('Count')
    
    # Detection rates by threshold
    thresholds = np.arange(0.01, 1.0, 0.02)
    p_rates = []
    s_rates = []
    
    for thresh in thresholds:
        p_rate = (results_df['p_prob'] >= thresh).sum() / len(results_df) * 100
        s_rate = (results_df['s_prob'] >= thresh).sum() / len(results_df) * 100
        p_rates.append(p_rate)
        s_rates.append(s_rate)
    
    axes[1, 0].plot(thresholds, p_rates, 'b-', label='P-wave', linewidth=2)
    axes[1, 0].plot(thresholds, s_rates, 'r-', label='S-wave', linewidth=2)
    axes[1, 0].set_title('Detection Rate vs Threshold')
    axes[1, 0].set_xlabel('Probability Threshold')
    axes[1, 0].set_ylabel('Detection Rate (%)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution (if available)
    valid_errors = results_df[(results_df['p_error'] >= 0) & (results_df['s_error'] >= 0)]
    if len(valid_errors) > 0:
        axes[1, 1].scatter(valid_errors['p_error'], valid_errors['s_error'], alpha=0.6)
        axes[1, 1].set_title('P vs S Prediction Errors')
        axes[1, 1].set_xlabel('P-wave Error (samples)')
        axes[1, 1].set_ylabel('S-wave Error (samples)')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No error data available', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Prediction Errors (No Data)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sliding_window_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plots saved to: {output_dir}/sliding_window_performance.png")

def main():
    parser = argparse.ArgumentParser(description='Test PhaseNet Indonesia dengan Sliding Window 3000')
    parser.add_argument('--test_dir', required=True, help='Directory containing test NPZ files')
    parser.add_argument('--test_list', required=True, help='CSV file listing test files')
    parser.add_argument('--model_dir', required=True, help='Directory containing trained model')
    parser.add_argument('--model_path', help='Specific model checkpoint path (optional)')
    parser.add_argument('--output_dir', help='Output directory for results', default='test_results_sliding3000')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for testing')
    parser.add_argument('--min_prob', type=float, default=0.1, help='Minimum probability threshold')
    parser.add_argument('--plot_results', action='store_true', help='Create performance plots')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("PhaseNet Indonesia Testing - Sliding Window 3000")
    print("=" * 60)
    print(f"Test directory: {args.test_dir}")
    print(f"Test list: {args.test_list}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Min probability: {args.min_prob}")
    print("=" * 60)
    
    # Create data reader
    data_config = DataConfig_Indonesia_3000()
    data_reader = DataReader_Indonesia_Sliding_Test(
        data_dir=args.test_dir,
        data_list=args.test_list,
        config=data_config
    )
    
    print(f"Test dataset loaded: {len(data_reader.data_list)} files")
    print(f"Sliding window config: {data_config.window_length} samples, {data_config.window_step} step")
    
    # Run testing
    start_time = time.time()
    results_df = test_fn(args, data_reader)
    end_time = time.time()
    
    if results_df is not None and len(results_df) > 0:
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        results_csv = os.path.join(args.output_dir, 'sliding_window_results.csv')
        results_df.to_csv(results_csv, index=False)
        print(f"Results saved to: {results_csv}")
        
        # Create plots if requested
        if args.plot_results:
            create_performance_plots(results_df, args.output_dir)
        
        # Summary
        print(f"\nTesting completed in {end_time - start_time:.1f} seconds")
        print(f"Successfully tested {len(results_df)} sliding windows")
        
    else:
        print("Testing failed or no results generated")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 