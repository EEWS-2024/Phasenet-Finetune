#!/usr/bin/env python3
"""
Testing script untuk PhaseNet Indonesia dengan 99% coverage
Window size: 135 detik (13,500 samples) untuk menangkap 99% data Indonesia
"""

import argparse
import os
import sys
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

# Add phasenet to path
sys.path.append(os.path.dirname(__file__))

from model import ModelConfig, UNet
from data_reader_indonesia import DataConfig_Indonesia, DataReader_Indonesia_Test
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
    """Test function for Indonesia 99% coverage model"""
    
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
                    print(f"❌ No model directory found in {args.model_dir}")
                    return None
                print(f"Found latest model directory: {model_dir_to_use}")
        except (OSError, PermissionError) as e:
            print(f"❌ Error accessing model directory {args.model_dir}: {e}")
            return None
    
    print(f"Using model directory: {model_dir_to_use}")
    
    # Load model configuration
    config_dict = LoadConfig(model_dir_to_use)
    print(f"Loaded config: {config_dict}")
    
    # Create model config with proper defaults
    config = ModelConfig(
        X_shape=data_reader.config.X_shape,
        Y_shape=data_reader.config.Y_shape,
        n_channel=data_reader.config.n_channel,
        n_class=data_reader.config.n_class,
        sampling_rate=data_reader.config.sampling_rate,
        dt=data_reader.config.dt,
        use_batch_norm=config_dict.get('use_batch_norm', True) if isinstance(config_dict, dict) else True,
        use_dropout=config_dict.get('use_dropout', True) if isinstance(config_dict, dict) else True,
        drop_rate=config_dict.get('drop_rate', 0.15) if isinstance(config_dict, dict) else 0.15,
        optimizer=config_dict.get('optimizer', 'adam') if isinstance(config_dict, dict) else 'adam',
        learning_rate=config_dict.get('learning_rate', 0.00003) if isinstance(config_dict, dict) else 0.00003,
        decay_step=config_dict.get('decay_step', 8) if isinstance(config_dict, dict) else 8,
        decay_rate=config_dict.get('decay_rate', 0.92) if isinstance(config_dict, dict) else 0.92,
        batch_size=config_dict.get('batch_size', 16) if isinstance(config_dict, dict) else 16,
        epochs=config_dict.get('epochs', 100) if isinstance(config_dict, dict) else 100,
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
            print(f"❌ No checkpoint found in {model_dir_to_use}")
            return None
            
        print(f"Loading model from: {model_path}")
        saver.restore(sess, model_path)
        
        # Test loop
        batch_count = 0
        total_samples = 0
        successful_detections = {'P': 0, 'S': 0, 'PS': 0}
        prob_stats = {'P_max': [], 'S_max': [], 'P_mean': [], 'S_mean': []}
        
        print(f"🚀 Starting testing on validation data...")
        
        # Define thresholds for detection
        min_threshold = args.min_prob
        med_threshold = min(min_threshold + 0.1, 0.9)
        high_threshold = min(min_threshold + 0.2, 0.9)
        
        print(f"🎯 Detection thresholds: {min_threshold:.2f} / {med_threshold:.2f} / {high_threshold:.2f}")
        
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
                
                # Show progress every 50 batches or if verbose every 10 batches
                progress_interval = 10 if hasattr(args, 'verbose') and args.verbose else 50
                if (batch_count + 1) % progress_interval == 0:
                    p_rate = successful_detections['P'] / max(1, total_samples) * 100
                    s_rate = successful_detections['S'] / max(1, total_samples) * 100
                    ps_rate = successful_detections['PS'] / max(1, total_samples) * 100
                    
                    # Show probability statistics
                    avg_p_max = np.mean(prob_stats['P_max'][-50:]) if prob_stats['P_max'] else 0
                    avg_s_max = np.mean(prob_stats['S_max'][-50:]) if prob_stats['S_max'] else 0
                    avg_p_mean = np.mean(prob_stats['P_mean'][-50:]) if prob_stats['P_mean'] else 0
                    avg_s_mean = np.mean(prob_stats['S_mean'][-50:]) if prob_stats['S_mean'] else 0
                    
                    print(f"📊 Batch {batch_count + 1}: {total_samples} samples | P: {p_rate:.1f}% | S: {s_rate:.1f}% | PS: {ps_rate:.1f}%")
                    print(f"   📈 Prob Stats - P_max: {avg_p_max:.3f}, S_max: {avg_s_max:.3f}, P_mean: {avg_p_mean:.3f}, S_mean: {avg_s_mean:.3f}")
                
                # Process each sample in batch
                for i in range(batch_size_actual):
                    fname = fname_batch[i].decode('utf-8')
                    sample = sample_batch[i]
                    target = target_batch[i]
                    pred = pred_batch[i]
                    p_true = p_true_batch[i][0]
                    s_true = s_true_batch[i][0]
                    
                    # Extract predictions
                    p_prob = pred[:, 0, 1]  # P-wave probability
                    s_prob = pred[:, 0, 2]  # S-wave probability
                    
                    # Find peaks using configurable threshold
                    # Use adaptive thresholds: min_prob, min_prob+0.1, min_prob+0.2
                    p_peaks_high, _ = find_peaks(p_prob, height=high_threshold, distance=30)
                    p_peaks_med, _ = find_peaks(p_prob, height=med_threshold, distance=30)  
                    p_peaks_low, _ = find_peaks(p_prob, height=min_threshold, distance=30)
                    
                    s_peaks_high, _ = find_peaks(s_prob, height=high_threshold, distance=30)
                    s_peaks_med, _ = find_peaks(s_prob, height=med_threshold, distance=30)
                    s_peaks_low, _ = find_peaks(s_prob, height=min_threshold, distance=30)
                    
                    # Get best predictions using adaptive threshold
                    p_pred = -1
                    if len(p_peaks_high) > 0:
                        p_pred = p_peaks_high[np.argmax(p_prob[p_peaks_high])]
                    elif len(p_peaks_med) > 0:
                        p_pred = p_peaks_med[np.argmax(p_prob[p_peaks_med])]
                    elif len(p_peaks_low) > 0:
                        p_pred = p_peaks_low[np.argmax(p_prob[p_peaks_low])]
                    
                    s_pred = -1
                    if len(s_peaks_high) > 0:
                        s_pred = s_peaks_high[np.argmax(s_prob[s_peaks_high])]
                    elif len(s_peaks_med) > 0:
                        s_pred = s_peaks_med[np.argmax(s_prob[s_peaks_med])]
                    elif len(s_peaks_low) > 0:
                        s_pred = s_peaks_low[np.argmax(s_prob[s_peaks_low])]
                    
                    # Calculate errors
                    p_error = abs(p_pred - p_true) if p_pred != -1 else -1
                    s_error = abs(s_pred - s_true) if s_pred != -1 else -1
                    
                    # Calculate P-S intervals
                    ps_true = s_true - p_true
                    ps_pred = s_pred - p_pred if p_pred != -1 and s_pred != -1 else -1
                    ps_error = abs(ps_pred - ps_true) if ps_pred != -1 else -1
                    
                    # Count successful detections
                    if p_pred != -1:
                        successful_detections['P'] += 1
                    if s_pred != -1:
                        successful_detections['S'] += 1
                    if ps_pred != -1:
                        successful_detections['PS'] += 1
                    
                    # Collect probability statistics
                    prob_stats['P_max'].append(np.max(p_prob))
                    prob_stats['S_max'].append(np.max(s_prob))
                    prob_stats['P_mean'].append(np.mean(p_prob))
                    prob_stats['S_mean'].append(np.mean(s_prob))
                    
                    result = {
                        'filename': fname,
                        'p_true': p_true,
                        'p_pred': p_pred,
                        'p_error': p_error,
                        's_true': s_true,
                        's_pred': s_pred,
                        's_error': s_error,
                        'ps_true': ps_true,
                        'ps_pred': ps_pred,
                        'ps_error': ps_error,
                        'p_prob_max': np.max(p_prob),
                        's_prob_max': np.max(s_prob)
                    }
                    results.append(result)
                    
                    # Show detailed output only if verbose flag is set
                    if hasattr(args, 'verbose') and args.verbose and i < 3 and (batch_count + 1) % 10 == 0:
                        print(f"  {fname}:")
                        print(f"    P: true={p_true}, pred={p_pred}, error={p_error}")
                        print(f"    S: true={s_true}, pred={s_pred}, error={s_error}")
                        print(f"    P-S: true={ps_true/100:.1f}s, pred={ps_pred/100:.1f}s" if ps_pred != -1 else f"    P-S: true={ps_true/100:.1f}s, pred=FAILED")
                
                batch_count += 1
                
        except tf.errors.OutOfRangeError:
            final_p_rate = successful_detections['P'] / max(1, total_samples) * 100
            final_s_rate = successful_detections['S'] / max(1, total_samples) * 100
            final_ps_rate = successful_detections['PS'] / max(1, total_samples) * 100
            print(f"\n✅ Testing completed!")
            print(f"📈 Final Results: {total_samples} samples | P: {final_p_rate:.1f}% | S: {final_s_rate:.1f}% | PS: {final_ps_rate:.1f}%")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(args.output_dir, 'test_results_indonesia.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    
    # Calculate statistics
    valid_p = results_df[results_df['p_pred'] != -1]
    valid_s = results_df[results_df['s_pred'] != -1]
    valid_ps = results_df[results_df['ps_pred'] != -1]
    
    print("\n=== PERFORMANCE STATISTICS ===")
    print(f"Total samples: {len(results_df)}")
    print(f"P-wave detection rate: {len(valid_p)}/{len(results_df)} ({len(valid_p)/len(results_df)*100:.1f}%)")
    print(f"S-wave detection rate: {len(valid_s)}/{len(results_df)} ({len(valid_s)/len(results_df)*100:.1f}%)")
    print(f"P-S pair detection rate: {len(valid_ps)}/{len(results_df)} ({len(valid_ps)/len(results_df)*100:.1f}%)")
    
    if len(valid_p) > 0:
        print(f"\nP-wave errors (samples):")
        print(f"  Mean: {valid_p['p_error'].mean():.1f}")
        print(f"  Median: {valid_p['p_error'].median():.1f}")
        print(f"  Std: {valid_p['p_error'].std():.1f}")
        print(f"  <10 samples: {(valid_p['p_error'] < 10).sum()}/{len(valid_p)} ({(valid_p['p_error'] < 10).mean()*100:.1f}%)")
    
    if len(valid_s) > 0:
        print(f"\nS-wave errors (samples):")
        print(f"  Mean: {valid_s['s_error'].mean():.1f}")
        print(f"  Median: {valid_s['s_error'].median():.1f}")
        print(f"  Std: {valid_s['s_error'].std():.1f}")
        print(f"  <10 samples: {(valid_s['s_error'] < 10).sum()}/{len(valid_s)} ({(valid_s['s_error'] < 10).mean()*100:.1f}%)")
    
    if len(valid_ps) > 0:
        print(f"\nP-S interval errors (samples):")
        print(f"  Mean: {valid_ps['ps_error'].mean():.1f}")
        print(f"  Median: {valid_ps['ps_error'].median():.1f}")
        print(f"  Std: {valid_ps['ps_error'].std():.1f}")
        print(f"  <50 samples: {(valid_ps['ps_error'] < 50).sum()}/{len(valid_ps)} ({(valid_ps['ps_error'] < 50).mean()*100:.1f}%)")
    
    # Create visualization
    if args.plot_results:
        create_performance_plots(results_df, args.output_dir)
    
    return results_df

def create_performance_plots(results_df, output_dir):
    """Create performance visualization plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PhaseNet Indonesia 99% Coverage - Performance Analysis', fontsize=16)
    
    # P-wave error distribution
    valid_p = results_df[results_df['p_pred'] != -1]
    if len(valid_p) > 0:
        axes[0, 0].hist(valid_p['p_error'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('P-wave Error Distribution')
        axes[0, 0].set_xlabel('Error (samples)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(valid_p['p_error'].median(), color='red', linestyle='--', label=f'Median: {valid_p["p_error"].median():.1f}')
        axes[0, 0].legend()
    
    # S-wave error distribution
    valid_s = results_df[results_df['s_pred'] != -1]
    if len(valid_s) > 0:
        axes[0, 1].hist(valid_s['s_error'], bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('S-wave Error Distribution')
        axes[0, 1].set_xlabel('Error (samples)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(valid_s['s_error'].median(), color='red', linestyle='--', label=f'Median: {valid_s["s_error"].median():.1f}')
        axes[0, 1].legend()
    
    # P-S interval error distribution
    valid_ps = results_df[results_df['ps_pred'] != -1]
    if len(valid_ps) > 0:
        axes[0, 2].hist(valid_ps['ps_error'], bins=50, alpha=0.7, color='orange')
        axes[0, 2].set_title('P-S Interval Error Distribution')
        axes[0, 2].set_xlabel('Error (samples)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(valid_ps['ps_error'].median(), color='red', linestyle='--', label=f'Median: {valid_ps["ps_error"].median():.1f}')
        axes[0, 2].legend()
    
    # Detection rates
    detection_rates = [
        len(valid_p) / len(results_df) * 100,
        len(valid_s) / len(results_df) * 100,
        len(valid_ps) / len(results_df) * 100
    ]
    axes[1, 0].bar(['P-wave', 'S-wave', 'P-S Pair'], detection_rates, color=['blue', 'green', 'orange'])
    axes[1, 0].set_title('Detection Rates')
    axes[1, 0].set_ylabel('Detection Rate (%)')
    axes[1, 0].set_ylim(0, 100)
    for i, rate in enumerate(detection_rates):
        axes[1, 0].text(i, rate + 1, f'{rate:.1f}%', ha='center')
    
    # P-S interval comparison
    if len(valid_ps) > 0:
        axes[1, 1].scatter(valid_ps['ps_true']/100, valid_ps['ps_pred']/100, alpha=0.6)
        max_val = max(valid_ps['ps_true'].max(), valid_ps['ps_pred'].max()) / 100
        axes[1, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect prediction')
        axes[1, 1].set_xlabel('True P-S Interval (seconds)')
        axes[1, 1].set_ylabel('Predicted P-S Interval (seconds)')
        axes[1, 1].set_title('P-S Interval: True vs Predicted')
        axes[1, 1].legend()
    
    # Probability distributions
    axes[1, 2].hist(results_df['p_prob_max'], bins=30, alpha=0.7, label='P-wave', color='blue')
    axes[1, 2].hist(results_df['s_prob_max'], bins=30, alpha=0.7, label='S-wave', color='green')
    axes[1, 2].set_title('Maximum Probability Distributions')
    axes[1, 2].set_xlabel('Maximum Probability')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'performance_analysis_indonesia.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance plots saved to: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description='Test PhaseNet Indonesia 99% Coverage')
    
    # Data parameters
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--test_list', type=str, required=True, help='Test data list CSV')
    parser.add_argument('--format', type=str, default='numpy', help='Data format')
    
    # Model parameters
    parser.add_argument('--model_dir', type=str, required=True, help='Model directory')
    parser.add_argument('--model_path', type=str, help='Specific model path (optional)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='test_results_indonesia', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for testing')
    parser.add_argument('--plot_results', action='store_true', help='Create performance plots')
    parser.add_argument('--verbose', action='store_true', help='Show detailed output for each sample')
    parser.add_argument('--min_prob', type=float, default=0.1, help='Minimum probability threshold for peak detection (default: 0.1)')
    
    # Window parameters
    parser.add_argument('--window_length', type=int, default=13500, help='Window length: 13500 samples (135s)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=== KONFIGURASI TESTING INDONESIA 99% COVERAGE ===")
    print(f"Window length: {args.window_length} samples ({args.window_length/100:.1f} detik)")
    print(f"Test directory: {args.test_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Create data config
    data_config = DataConfig_Indonesia(
        window_length=args.window_length,
        X_shape=[args.window_length, 1, 3],
        Y_shape=[args.window_length, 1, 3]
    )
    
    # Create data reader
    data_reader = DataReader_Indonesia_Test(
        format=args.format,
        config=data_config,
        data_dir=args.test_dir,
        data_list=args.test_list
    )
    
    # Start testing
    results = test_fn(args, data_reader)
    
    print(f"\nTesting completed successfully!")
    print(f"Results saved in: {args.output_dir}")

if __name__ == '__main__':
    main() 