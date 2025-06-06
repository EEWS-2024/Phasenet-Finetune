#!/usr/bin/env python3
"""
Testing script untuk PhaseNet Indonesia Decoder-Only Model
Compatible dengan hasil training decoder-only fine-tuning
"""

import argparse
import os
import sys

# Set environment variables BEFORE importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Enable GPU memory growth
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
from scipy.signal import find_peaks

# Add phasenet to path
sys.path.append(os.path.dirname(__file__))

from model import ModelConfig, UNet
from data_reader_indonesia_sliding import DataConfig_Indonesia_3000, DataReader_Indonesia_Sliding_Test
import json
from tqdm import tqdm

def load_decoder_only_model(sess, model_dir, model_name="decoder_model_final.ckpt"):
    """Load decoder-only fine-tuned model dengan handling khusus"""
    
    model_path = os.path.join(model_dir, model_name)
    
    if not os.path.exists(model_path + ".index"):
        print(f"❌ Model file tidak ditemukan: {model_path}")
        return False
    
    print(f"🔄 Loading decoder-only model from: {model_path}")
    
    try:
        # Get all variables in current graph
        all_vars = tf.compat.v1.global_variables()
        
        # Get checkpoint variables
        checkpoint_vars = tf.train.list_variables(model_path)
        checkpoint_var_names = {name for name, shape in checkpoint_vars}
        
        # Filter variables to load (exclude problematic optimizer variables)
        vars_to_load = []
        excluded_patterns = [
            'Adam', 'adam', 'beta1_power', 'beta2_power', 
            'decoder_optimizer', 'global_step'
        ]
        
        for var in all_vars:
            var_name = var.name.split(':')[0]
            
            # Skip optimizer variables
            if any(pattern in var_name for pattern in excluded_patterns):
                continue
                
            if var_name in checkpoint_var_names:
                vars_to_load.append(var)
        
        print(f"📋 Variables to load: {len(vars_to_load)}/{len(all_vars)}")
        
        # Create saver with filtered variables
        saver = tf.compat.v1.train.Saver(var_list=vars_to_load)
        
        # Restore model
        saver.restore(sess, model_path)
        
        print(f"✅ Successfully loaded decoder-only model")
        print(f"   Loaded {len(vars_to_load)} variables")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading decoder-only model: {e}")
        return False

def test_fn(args, data_reader):
    """Testing function untuk decoder-only model dengan ground truth comparison"""
    
    # Load config
    config_file = os.path.join(args.model_dir, 'config.json')
    if not os.path.exists(config_file):
        print(f"❌ Config file tidak ditemukan: {config_file}")
        return None
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    
    print(f"📊 Loaded config from: {config_file}")
    
    # Create model config
    config = ModelConfig(
        X_shape=config_dict['X_shape'],
        Y_shape=config_dict['Y_shape'],
        n_channel=config_dict['n_channel'],
        n_class=config_dict['n_class'],
        sampling_rate=config_dict['sampling_rate'],
        dt=config_dict['dt'],
        use_batch_norm=config_dict.get('use_batch_norm', True),
        use_dropout=config_dict.get('use_dropout', True),
        drop_rate=config_dict.get('drop_rate', 0.05),
    )
    
    # Create dataset
    test_dataset = data_reader.dataset(args.batch_size, shuffle=False, drop_remainder=False)
    
    # Create placeholders
    X_placeholder = tf.compat.v1.placeholder(tf.float32, [None] + config.X_shape, name='X_input')
    Y_placeholder = tf.compat.v1.placeholder(tf.float32, [None] + config.Y_shape, name='Y_target')
    fname_placeholder = tf.compat.v1.placeholder(tf.string, [None], name='fname_input')
    
    # Create model in inference mode
    model = UNet(config=config, input_batch=(X_placeholder, Y_placeholder, fname_placeholder), mode='test')
    
    # Configure for GPU testing with optimizations
    gpu_config = tf.compat.v1.ConfigProto()
    gpu_config.allow_soft_placement = True
    gpu_config.log_device_placement = False
    
    # GPU memory settings
    if gpus:
        gpu_config.gpu_options.allow_growth = True
        gpu_config.gpu_options.per_process_gpu_memory_fraction = 0.6  # Use 60% of GPU memory for testing
        print(f"🚀 Using GPU for testing with memory growth enabled")
    else:
        gpu_config.device_count = {'GPU': 0}
        print(f"🖥️  Using CPU for testing")
    
    results = []
    
    with tf.compat.v1.Session(config=gpu_config) as sess:
        # Initialize variables
        sess.run(tf.compat.v1.global_variables_initializer())
        
        # Load decoder-only model
        if not load_decoder_only_model(sess, args.model_dir):
            print("❌ Failed to load decoder-only model")
            return None
        
        print(f"\n🚀 Starting decoder-only testing with ground truth comparison...")
        print(f"   Test windows: {len(data_reader.sliding_windows):,}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Min probability: {args.min_prob}")
        
        # Create iterator
        test_iterator = tf.compat.v1.data.make_initializable_iterator(test_dataset)
        next_test_batch = test_iterator.get_next()
        sess.run(test_iterator.initializer)
        
        batch_count = 0
        total_windows = 0
        
        # Statistics tracking
        detection_stats = {'P': 0, 'S': 0, 'PS': 0}
        
        # Calculate total batches for progress bar
        total_batches = int(np.ceil(len(data_reader.sliding_windows) / args.batch_size))
        
        try:
            # Create progress bar dengan format yang lebih sederhana
            with tqdm(total=total_batches, desc="Testing Progress", unit="batch") as pbar:
                
                while True:
                    batch_count += 1
                    try:
                        # Get batch data (5 values: X, Y, fname, p_idx, s_idx)
                        X_batch, Y_batch, fname_batch, p_idx_batch, s_idx_batch = sess.run(next_test_batch)
                        
                        # Predict
                        pred_batch = sess.run(model.preds, feed_dict={
                            X_placeholder: X_batch,
                            Y_placeholder: Y_batch,
                            fname_placeholder: fname_batch,
                            model.drop_rate: 0.0,
                            model.is_training: False
                        })
                        
                        # Process each sample in batch
                        for i in range(len(X_batch)):
                            fname = fname_batch[i].decode('utf-8') if isinstance(fname_batch[i], bytes) else fname_batch[i]
                            pred = pred_batch[i]  # Shape: [3000, 1, 3]
                            
                            # Ground truth indices dari data reader
                            p_true = int(p_idx_batch[i][0]) if p_idx_batch[i][0] >= 0 else -1
                            s_true = int(s_idx_batch[i][0]) if s_idx_batch[i][0] >= 0 else -1
                            
                            # Handle prediction shape
                            if len(pred.shape) == 3 and pred.shape[1] == 1:
                                pred = np.squeeze(pred, axis=1)  # [3000, 3]
                            
                            # Extract predictions
                            prob_P = pred[:, 1]  # P-wave probability
                            prob_S = pred[:, 2]  # S-wave probability
                            
                            # Find peaks above threshold
                            P_peaks, _ = find_peaks(prob_P, height=args.min_prob, distance=10)
                            S_peaks, _ = find_peaks(prob_S, height=args.min_prob, distance=10)
                            
                            # Get best predictions
                            p_pred = P_peaks[np.argmax(prob_P[P_peaks])] if len(P_peaks) > 0 else -1
                            s_pred = S_peaks[np.argmax(prob_S[S_peaks])] if len(S_peaks) > 0 else -1
                            
                            # Get probabilities at predictions
                            p_prob = prob_P[p_pred] if p_pred >= 0 else 0.0
                            s_prob = prob_S[s_pred] if s_pred >= 0 else 0.0
                            
                            # Calculate errors (only if both true and pred exist)
                            p_error = abs(p_pred - p_true) if p_true >= 0 and p_pred >= 0 else -1
                            s_error = abs(s_pred - s_true) if s_true >= 0 and s_pred >= 0 else -1
                            
                            # Detection flags (using 50 sample tolerance)
                            p_detected = (p_error >= 0 and p_error <= 50) if p_error >= 0 else False
                            s_detected = (s_error >= 0 and s_error <= 50) if s_error >= 0 else False
                            
                            # Update statistics
                            if p_detected:
                                detection_stats['P'] += 1
                            if s_detected:
                                detection_stats['S'] += 1
                            if p_detected and s_detected:
                                detection_stats['PS'] += 1
                            
                            # Store comprehensive results (matching standard testing format)
                            results.append({
                                'filename': fname,
                                'p_true': p_true,
                                's_true': s_true,
                                'p_pred': p_pred,
                                's_pred': s_pred,
                                'p_prob': p_prob,
                                's_prob': s_prob,
                                'p_error': p_error,
                                's_error': s_error,
                                'p_detected': p_detected,
                                's_detected': s_detected,
                                'window_idx': total_windows,
                                # Additional decoder-only metrics
                                'max_P_prob': np.max(prob_P),
                                'max_S_prob': np.max(prob_S),
                                'avg_P_prob': np.mean(prob_P),
                                'avg_S_prob': np.mean(prob_S),
                                'n_P_picks': len(P_peaks),
                                'n_S_picks': len(S_peaks)
                            })
                            
                            total_windows += 1
                        
                        # Update progress bar dengan statistik terkini
                        p_rate = detection_stats['P'] / max(1, total_windows) * 100
                        s_rate = detection_stats['S'] / max(1, total_windows) * 100
                        
                        # Update description dengan statistik
                        pbar.set_description(f"Testing Progress - P:{p_rate:.1f}% S:{s_rate:.1f}%")
                        pbar.update(1)
                            
                    except tf.errors.OutOfRangeError:
                        break
                        
        except Exception as e:
            print(f"⚠️  Testing error: {e}")
        
        print(f"✅ Decoder-only testing completed!")
        print(f"   Total batches: {batch_count}")
        print(f"   Total windows: {total_windows}")
        
        # Final statistics
        if total_windows > 0:
            final_p_rate = detection_stats['P'] / total_windows * 100
            final_s_rate = detection_stats['S'] / total_windows * 100
            final_ps_rate = detection_stats['PS'] / total_windows * 100
            print(f"\n📊 DECODER-ONLY DETECTION RATES:")
            print(f"   P-wave: {final_p_rate:.1f}% ({detection_stats['P']}/{total_windows})")
            print(f"   S-wave: {final_s_rate:.1f}% ({detection_stats['S']}/{total_windows})")
            print(f"   Both P&S: {final_ps_rate:.1f}% ({detection_stats['PS']}/{total_windows})")
    
    return pd.DataFrame(results)

def plot_results(results_df, output_dir):
    """Create comprehensive plots untuk hasil decoder-only testing"""
    
    plt.figure(figsize=(20, 15))
    
    # Plot 1: Detection Success Rates
    plt.subplot(3, 4, 1)
    total = len(results_df)
    p_detected = results_df['p_detected'].sum()
    s_detected = results_df['s_detected'].sum()
    ps_detected = (results_df['p_detected'] & results_df['s_detected']).sum()
    
    categories = ['P-wave', 'S-wave', 'Both P&S']
    success_rates = [p_detected/total*100, s_detected/total*100, ps_detected/total*100]
    colors = ['blue', 'red', 'green']
    
    bars = plt.bar(categories, success_rates, color=colors, alpha=0.7)
    plt.title('Detection Success Rates\n(Decoder-Only)', fontweight='bold')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    
    for bar, rate in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Probability Distributions
    plt.subplot(3, 4, 2)
    plt.hist(results_df['max_P_prob'], bins=50, alpha=0.7, label='Max P-prob', color='blue')
    plt.hist(results_df['max_S_prob'], bins=50, alpha=0.7, label='Max S-prob', color='red')
    plt.xlabel('Maximum Probability')
    plt.ylabel('Frequency')
    plt.title('Max Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Average Probabilities
    plt.subplot(3, 4, 3)
    plt.hist(results_df['avg_P_prob'], bins=50, alpha=0.7, label='Avg P-prob', color='blue')
    plt.hist(results_df['avg_S_prob'], bins=50, alpha=0.7, label='Avg S-prob', color='red')
    plt.xlabel('Average Probability')
    plt.ylabel('Frequency')
    plt.title('Avg Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Error Distributions (for successful detections)
    plt.subplot(3, 4, 4)
    p_errors = results_df[results_df['p_error'] >= 0]['p_error']
    s_errors = results_df[results_df['s_error'] >= 0]['s_error']
    
    if len(p_errors) > 0:
        plt.hist(p_errors, bins=30, alpha=0.7, label=f'P-errors (n={len(p_errors)})', color='blue')
    if len(s_errors) > 0:
        plt.hist(s_errors, bins=30, alpha=0.7, label=f'S-errors (n={len(s_errors)})', color='red')
    
    plt.xlabel('Error (samples)')
    plt.ylabel('Frequency')
    plt.title('Detection Errors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: P vs S Probability Scatter
    plt.subplot(3, 4, 5)
    plt.scatter(results_df['max_P_prob'], results_df['max_S_prob'], alpha=0.6, s=10)
    plt.xlabel('Max P-wave Probability')
    plt.ylabel('Max S-wave Probability')
    plt.title('P vs S Probability Scatter')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Pick Counts
    plt.subplot(3, 4, 6)
    plt.hist(results_df['n_P_picks'], bins=30, alpha=0.7, label='P-picks', color='blue')
    plt.hist(results_df['n_S_picks'], bins=30, alpha=0.7, label='S-picks', color='red')
    plt.xlabel('Number of Picks per Window')
    plt.ylabel('Frequency')
    plt.title('Pick Count Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Predicted vs True P-wave indices
    plt.subplot(3, 4, 7)
    valid_p = results_df[(results_df['p_true'] >= 0) & (results_df['p_pred'] >= 0)]
    if len(valid_p) > 0:
        plt.scatter(valid_p['p_true'], valid_p['p_pred'], alpha=0.6, s=10, color='blue')
        min_val = min(valid_p['p_true'].min(), valid_p['p_pred'].min())
        max_val = max(valid_p['p_true'].max(), valid_p['p_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('True P-wave Index')
        plt.ylabel('Predicted P-wave Index')
        plt.title(f'P-wave Predictions (n={len(valid_p)})')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Valid P-wave\nPredictions', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('P-wave Predictions')
    
    # Plot 8: Predicted vs True S-wave indices  
    plt.subplot(3, 4, 8)
    valid_s = results_df[(results_df['s_true'] >= 0) & (results_df['s_pred'] >= 0)]
    if len(valid_s) > 0:
        plt.scatter(valid_s['s_true'], valid_s['s_pred'], alpha=0.6, s=10, color='red')
        min_val = min(valid_s['s_true'].min(), valid_s['s_pred'].min())
        max_val = max(valid_s['s_true'].max(), valid_s['s_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        plt.xlabel('True S-wave Index')
        plt.ylabel('Predicted S-wave Index')
        plt.title(f'S-wave Predictions (n={len(valid_s)})')
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No Valid S-wave\nPredictions', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('S-wave Predictions')
    
    # Plot 9: Summary Statistics Table
    plt.subplot(3, 4, 9)
    stats_data = [
        ['Total Windows', f"{len(results_df):,}"],
        ['P Detections', f"{results_df['p_detected'].sum():,}"],
        ['S Detections', f"{results_df['s_detected'].sum():,}"],
        ['P Detection Rate', f"{results_df['p_detected'].mean()*100:.1f}%"],
        ['S Detection Rate', f"{results_df['s_detected'].mean()*100:.1f}%"],
        ['Mean P-prob', f"{results_df['avg_P_prob'].mean():.4f}"],
        ['Mean S-prob', f"{results_df['avg_S_prob'].mean():.4f}"],
        ['Max P-prob', f"{results_df['max_P_prob'].max():.4f}"],
        ['Max S-prob', f"{results_df['max_S_prob'].max():.4f}"]
    ]
    
    plt.axis('off')
    table = plt.table(cellText=stats_data,
                     colLabels=['Metric', 'Value'],
                     cellLoc='left',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title('Decoder-Only Statistics', pad=20, fontweight='bold')
    
    # Plot 10: Probability vs Detection Success
    plt.subplot(3, 4, 10)
    detected_p_probs = results_df[results_df['p_detected']]['p_prob']
    missed_p_probs = results_df[~results_df['p_detected'] & (results_df['p_true'] >= 0)]['p_prob']
    
    if len(detected_p_probs) > 0:
        plt.hist(detected_p_probs, bins=20, alpha=0.7, label=f'Detected (n={len(detected_p_probs)})', color='green')
    if len(missed_p_probs) > 0:
        plt.hist(missed_p_probs, bins=20, alpha=0.7, label=f'Missed (n={len(missed_p_probs)})', color='red')
    
    plt.xlabel('P-wave Probability')
    plt.ylabel('Frequency')
    plt.title('P-wave Detection Success vs Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 11: Time Series Example (first valid window)
    plt.subplot(3, 4, 11)
    # This would require actual waveform data, so just show placeholder
    plt.text(0.5, 0.5, 'Waveform Example\n(Requires Raw Data)', ha='center', va='center', transform=plt.gca().transAxes)
    plt.title('Sample Waveform & Predictions')
    
    # Plot 12: Model Confidence Analysis
    plt.subplot(3, 4, 12)
    high_conf_p = (results_df['max_P_prob'] > 0.5).sum()
    med_conf_p = ((results_df['max_P_prob'] > 0.3) & (results_df['max_P_prob'] <= 0.5)).sum()
    low_conf_p = (results_df['max_P_prob'] <= 0.3).sum()
    
    confidence_levels = ['Low\n(≤0.3)', 'Med\n(0.3-0.5)', 'High\n(>0.5)']
    counts = [low_conf_p, med_conf_p, high_conf_p]
    colors = ['red', 'orange', 'green']
    
    bars = plt.bar(confidence_levels, counts, color=colors, alpha=0.7)
    plt.title('P-wave Confidence Levels')
    plt.ylabel('Number of Windows')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + len(results_df)*0.01,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'decoder_only_comprehensive_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Comprehensive results plot saved: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Test PhaseNet Indonesia Decoder-Only Model')
    
    # Required arguments
    parser.add_argument('--test_dir', type=str, required=True, help='Test data directory')
    parser.add_argument('--test_list', type=str, required=True, help='Test data list CSV')
    parser.add_argument('--model_dir', type=str, required=True, help='Decoder-only model directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    # Optional arguments
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for testing')
    parser.add_argument('--min_prob', type=float, default=0.1, help='Minimum probability threshold')
    parser.add_argument('--format', type=str, default='numpy', help='Data format')
    parser.add_argument('--plot_results', action='store_true', help='Generate result plots')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.test_dir):
        print(f"❌ Test directory tidak ditemukan: {args.test_dir}")
        return 1
    
    if not os.path.exists(args.test_list):   
        print(f"❌ Test list tidak ditemukan: {args.test_list}")
        return 1
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory tidak ditemukan: {args.model_dir}")
        return 1
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎯 PhaseNet Indonesia Decoder-Only Testing")
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
        config=data_config,
        format=args.format
    )
    
    print(f"📊 Test data loaded: {len(data_reader.sliding_windows):,} windows")
    
    # Run testing
    results_df = test_fn(args, data_reader)
    
    if results_df is not None and len(results_df) > 0:
        # Save results with comprehensive format
        results_file = os.path.join(args.output_dir, 'decoder_only_comprehensive_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"💾 Comprehensive results saved: {results_file}")
        
        # Print detailed summary
        print(f"\n📊 DECODER-ONLY COMPREHENSIVE SUMMARY:")
        print(f"   Total windows tested: {len(results_df):,}")
        
        # Detection rates
        p_detected = results_df['p_detected'].sum()
        s_detected = results_df['s_detected'].sum()
        ps_detected = (results_df['p_detected'] & results_df['s_detected']).sum()
        
        print(f"   P-wave detections: {p_detected:,} ({p_detected/len(results_df)*100:.1f}%)")
        print(f"   S-wave detections: {s_detected:,} ({s_detected/len(results_df)*100:.1f}%)")
        print(f"   Both P&S detections: {ps_detected:,} ({ps_detected/len(results_df)*100:.1f}%)")
        
        # Probability statistics
        print(f"\n📈 PROBABILITY STATISTICS:")
        print(f"   Mean P-wave probability: {results_df['avg_P_prob'].mean():.4f}")
        print(f"   Mean S-wave probability: {results_df['avg_S_prob'].mean():.4f}")
        print(f"   Max P-wave probability: {results_df['max_P_prob'].max():.4f}")
        print(f"   Max S-wave probability: {results_df['max_S_prob'].max():.4f}")
        
        # Error statistics (for successful detections)
        valid_p_errors = results_df[results_df['p_error'] >= 0]['p_error']
        valid_s_errors = results_df[results_df['s_error'] >= 0]['s_error']
        
        if len(valid_p_errors) > 0:
            print(f"\n🎯 ACCURACY STATISTICS:")
            print(f"   P-wave error - Mean: {valid_p_errors.mean():.1f} samples, Median: {valid_p_errors.median():.1f}")
        if len(valid_s_errors) > 0:
            print(f"   S-wave error - Mean: {valid_s_errors.mean():.1f} samples, Median: {valid_s_errors.median():.1f}")
        
        # Confidence analysis
        high_p = (results_df['max_P_prob'] > 0.5).sum()
        high_s = (results_df['max_S_prob'] > 0.5).sum()
        print(f"\n🔍 CONFIDENCE ANALYSIS:")
        print(f"   High confidence P-waves (>0.5): {high_p:,} ({high_p/len(results_df)*100:.1f}%)")
        print(f"   High confidence S-waves (>0.5): {high_s:,} ({high_s/len(results_df)*100:.1f}%)")
        
        # Generate comprehensive plots if requested
        if args.plot_results:
            plot_results(results_df, args.output_dir)
        
        print(f"\n✅ Decoder-only comprehensive testing completed successfully!")
        return 0
    else:
        print("❌ Testing failed atau tidak ada hasil")
        return 1

if __name__ == '__main__':
    exit(main()) 