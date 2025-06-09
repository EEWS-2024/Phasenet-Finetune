#!/usr/bin/env python3
"""
Script untuk membandingkan hasil testing antara model original dengan model finetuned
Membuat comparison charts dan analysis untuk mengevaluasi effectiveness dari fine-tuning
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import json

def load_test_results(results_dir):
    """Load test results from directory"""
    csv_path = os.path.join(results_dir, 'sliding_window_results.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Results file not found: {csv_path}")
        return None
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} test results from {results_dir}")
    return df

def get_model_info(results_dir):
    """Extract model information from results directory"""
    info = {'name': os.path.basename(results_dir), 'type': 'Unknown', 'details': ''}
    
    # Check for model info file
    info_file = os.path.join(results_dir, 'model_info.txt')
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            content = f.read()
            if 'Original PhaseNet' in content:
                info['type'] = 'Original (NCEDC)'
                info['details'] = 'Pre-trained on NCEDC, no fine-tuning'
            elif 'DECODER-ONLY' in content:
                info['type'] = 'Decoder-Only Fine-tuned'
                info['details'] = 'Only decoder trained, encoder frozen'
            elif 'from scratch' in content.lower():
                info['type'] = 'Trained from Scratch'
                info['details'] = 'Full training on Indonesia data'
    
    # Check for config file
    config_file = os.path.join(results_dir, 'config.json')
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                if 'epochs' in config:
                    info['details'] += f" ({config['epochs']} epochs)"
        except:
            pass
    
    return info

def calculate_metrics(df, min_prob=0.1):
    """Calculate performance metrics from results dataframe"""
    total_windows = len(df)
    
    # Detection rates
    p_detected = df['p_detected'].sum()
    s_detected = df['s_detected'].sum()
    p_detection_rate = (p_detected / total_windows) * 100
    s_detection_rate = (s_detected / total_windows) * 100
    
    # Average probabilities
    avg_p_prob = df['max_p_prob'].mean()
    avg_s_prob = df['max_s_prob'].mean()
    std_p_prob = df['max_p_prob'].std()
    std_s_prob = df['max_s_prob'].std()
    
    # Error metrics (only for windows with ground truth)
    p_errors = df['p_error_samples'].dropna()
    s_errors = df['s_error_samples'].dropna()
    
    p_mae = p_errors.mean() if len(p_errors) > 0 else None
    s_mae = s_errors.mean() if len(s_errors) > 0 else None
    p_mae_std = p_errors.std() if len(p_errors) > 0 else None
    s_mae_std = s_errors.std() if len(s_errors) > 0 else None
    
    return {
        'total_windows': total_windows,
        'p_detection_rate': p_detection_rate,
        's_detection_rate': s_detection_rate,
        'p_detected_count': p_detected,
        's_detected_count': s_detected,
        'avg_p_prob': avg_p_prob,
        'avg_s_prob': avg_s_prob,
        'std_p_prob': std_p_prob,
        'std_s_prob': std_s_prob,
        'p_mae': p_mae,
        's_mae': s_mae,
        'p_mae_std': p_mae_std,
        's_mae_std': s_mae_std,
        'p_error_samples': len(p_errors),
        's_error_samples': len(s_errors)
    }

def create_comparison_plots(models_data, output_dir, min_prob=0.1):
    """Create comprehensive comparison plots"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Model Performance Comparison (min_prob={min_prob})', fontsize=16, fontweight='bold')
    
    model_names = list(models_data.keys())
    colors = ['red', 'blue', 'green', 'orange', 'purple'][:len(model_names)]
    
    # 1. Detection Rates Comparison
    p_detection_rates = [models_data[model]['metrics']['p_detection_rate'] for model in model_names]
    s_detection_rates = [models_data[model]['metrics']['s_detection_rate'] for model in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, p_detection_rates, width, label='P-wave', color='blue', alpha=0.7)
    axes[0, 0].bar(x + width/2, s_detection_rates, width, label='S-wave', color='red', alpha=0.7)
    axes[0, 0].set_xlabel('Models')
    axes[0, 0].set_ylabel('Detection Rate (%)')
    axes[0, 0].set_title('Detection Rates Comparison')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels([models_data[model]['info']['type'] for model in model_names], rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add percentage labels
    for i, (p_rate, s_rate) in enumerate(zip(p_detection_rates, s_detection_rates)):
        axes[0, 0].text(i - width/2, p_rate + 1, f'{p_rate:.1f}%', ha='center', va='bottom')
        axes[0, 0].text(i + width/2, s_rate + 1, f'{s_rate:.1f}%', ha='center', va='bottom')
    
    # 2. Average Probability Comparison
    avg_p_probs = [models_data[model]['metrics']['avg_p_prob'] for model in model_names]
    avg_s_probs = [models_data[model]['metrics']['avg_s_prob'] for model in model_names]
    
    axes[0, 1].bar(x - width/2, avg_p_probs, width, label='P-wave', color='blue', alpha=0.7)
    axes[0, 1].bar(x + width/2, avg_s_probs, width, label='S-wave', color='red', alpha=0.7)
    axes[0, 1].set_xlabel('Models')
    axes[0, 1].set_ylabel('Average Probability')
    axes[0, 1].set_title('Average Detection Probabilities')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels([models_data[model]['info']['type'] for model in model_names], rotation=45)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Error Comparison (MAE)
    p_maes = [models_data[model]['metrics']['p_mae'] for model in model_names]
    s_maes = [models_data[model]['metrics']['s_mae'] for model in model_names]
    
    # Filter out None values
    p_maes_clean = [(i, mae) for i, mae in enumerate(p_maes) if mae is not None]
    s_maes_clean = [(i, mae) for i, mae in enumerate(s_maes) if mae is not None]
    
    if p_maes_clean:
        p_indices, p_values = zip(*p_maes_clean)
        axes[0, 2].bar([i - width/2 for i in p_indices], p_values, width, 
                      label='P-wave MAE', color='blue', alpha=0.7)
    
    if s_maes_clean:
        s_indices, s_values = zip(*s_maes_clean)
        axes[0, 2].bar([i + width/2 for i in s_indices], s_values, width, 
                      label='S-wave MAE', color='red', alpha=0.7)
    
    axes[0, 2].set_xlabel('Models')
    axes[0, 2].set_ylabel('Mean Absolute Error (samples)')
    axes[0, 2].set_title('Prediction Errors (Lower is better)')
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels([models_data[model]['info']['type'] for model in model_names], rotation=45)
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Probability Distribution Comparison (P-wave)
    for i, model in enumerate(model_names):
        df = models_data[model]['data']
        axes[1, 0].hist(df['max_p_prob'], bins=30, alpha=0.5, label=models_data[model]['info']['type'], 
                       color=colors[i], density=True)
    
    axes[1, 0].axvline(min_prob, color='black', linestyle='--', label=f'Threshold ({min_prob})')
    axes[1, 0].set_xlabel('P-wave Probability')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('P-wave Probability Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Probability Distribution Comparison (S-wave)
    for i, model in enumerate(model_names):
        df = models_data[model]['data']
        axes[1, 1].hist(df['max_s_prob'], bins=30, alpha=0.5, label=models_data[model]['info']['type'], 
                       color=colors[i], density=True)
    
    axes[1, 1].axvline(min_prob, color='black', linestyle='--', label=f'Threshold ({min_prob})')
    axes[1, 1].set_xlabel('S-wave Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('S-wave Probability Distributions')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Summary Statistics Table
    summary_text = "MODEL COMPARISON SUMMARY\n" + "="*40 + "\n\n"
    
    for model in model_names:
        info = models_data[model]['info']
        metrics = models_data[model]['metrics']
        
        summary_text += f"{info['type']}\n"
        summary_text += f"  Detection Rate: P={metrics['p_detection_rate']:.1f}%, S={metrics['s_detection_rate']:.1f}%\n"
        summary_text += f"  Avg Probability: P={metrics['avg_p_prob']:.3f}, S={metrics['avg_s_prob']:.3f}\n"
        
        if metrics['p_mae'] is not None:
            summary_text += f"  P-wave MAE: {metrics['p_mae']:.1f}±{metrics['p_mae_std']:.1f} samples\n"
        if metrics['s_mae'] is not None:
            summary_text += f"  S-wave MAE: {metrics['s_mae']:.1f}±{metrics['s_mae_std']:.1f} samples\n"
        
        summary_text += f"  Windows tested: {metrics['total_windows']:,}\n\n"
    
    # Find best performing model
    best_p_detection = max(model_names, key=lambda x: models_data[x]['metrics']['p_detection_rate'])
    best_s_detection = max(model_names, key=lambda x: models_data[x]['metrics']['s_detection_rate'])
    
    summary_text += "BEST PERFORMERS:\n"
    summary_text += f"  P-wave detection: {models_data[best_p_detection]['info']['type']}\n"
    summary_text += f"  S-wave detection: {models_data[best_s_detection]['info']['type']}\n"
    
    # Calculate improvement over original (if available)
    original_models = [m for m in model_names if 'Original' in models_data[m]['info']['type']]
    if original_models:
        original = original_models[0]
        summary_text += f"\nIMPROVEMENT OVER ORIGINAL:\n"
        
        for model in model_names:
            if model != original:
                p_improvement = models_data[model]['metrics']['p_detection_rate'] - models_data[original]['metrics']['p_detection_rate']
                s_improvement = models_data[model]['metrics']['s_detection_rate'] - models_data[original]['metrics']['s_detection_rate']
                summary_text += f"  {models_data[model]['info']['type']}: P={p_improvement:+.1f}%, S={s_improvement:+.1f}%\n"
    
    axes[1, 2].text(0.05, 0.95, summary_text, transform=axes[1, 2].transAxes, fontsize=9,
                   verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Comparison plot saved to: {plot_path}")

def save_comparison_csv(models_data, output_dir):
    """Save comparison metrics to CSV"""
    
    comparison_data = []
    
    for model_name, model_data in models_data.items():
        info = model_data['info']
        metrics = model_data['metrics']
        
        comparison_data.append({
            'model_name': model_name,
            'model_type': info['type'],
            'model_details': info['details'],
            'total_windows': metrics['total_windows'],
            'p_detection_rate': metrics['p_detection_rate'],
            's_detection_rate': metrics['s_detection_rate'],
            'p_detected_count': metrics['p_detected_count'],
            's_detected_count': metrics['s_detected_count'],
            'avg_p_prob': metrics['avg_p_prob'],
            'avg_s_prob': metrics['avg_s_prob'],
            'std_p_prob': metrics['std_p_prob'],
            'std_s_prob': metrics['std_s_prob'],
            'p_mae': metrics['p_mae'],
            's_mae': metrics['s_mae'],
            'p_mae_std': metrics['p_mae_std'],
            's_mae_std': metrics['s_mae_std'],
            'p_error_samples': metrics['p_error_samples'],
            's_error_samples': metrics['s_error_samples']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    csv_path = os.path.join(output_dir, 'model_comparison.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"📊 Comparison metrics saved to: {csv_path}")
    
    return comparison_df

def main():
    parser = argparse.ArgumentParser(description='Compare PhaseNet model testing results')
    
    parser.add_argument('--result_dirs', type=str, nargs='+', required=True, 
                       help='Directories containing test results (e.g., test_results_original model_dir/test_results)')
    parser.add_argument('--output_dir', type=str, default='comparison_results', help='Output directory for comparison')
    parser.add_argument('--min_prob', type=float, default=0.1, help='Minimum probability threshold used in testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🔍 Loading model test results...")
    print("=" * 50)
    
    # Load all model results
    models_data = {}
    
    for result_dir in args.result_dirs:
        if not os.path.exists(result_dir):
            print(f"⚠️  Result directory not found: {result_dir}")
            continue
        
        # Load test results
        df = load_test_results(result_dir)
        if df is None:
            continue
        
        # Get model info
        info = get_model_info(result_dir)
        
        # Calculate metrics
        metrics = calculate_metrics(df, args.min_prob)
        
        # Store data
        model_key = os.path.basename(result_dir)
        models_data[model_key] = {
            'data': df,
            'info': info,
            'metrics': metrics
        }
        
        print(f"✅ {info['type']}: {metrics['total_windows']} windows, "
              f"P={metrics['p_detection_rate']:.1f}%, S={metrics['s_detection_rate']:.1f}%")
    
    if len(models_data) == 0:
        print("❌ No valid test results found!")
        return
    
    print(f"\n📊 Comparing {len(models_data)} models...")
    
    # Create comparison plots
    create_comparison_plots(models_data, args.output_dir, args.min_prob)
    
    # Save comparison CSV
    comparison_df = save_comparison_csv(models_data, args.output_dir)
    
    print(f"\n✅ Comparison completed!")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Comparison plot: {args.output_dir}/model_comparison.png")
    print(f"   Comparison data: {args.output_dir}/model_comparison.csv")
    
    # Print summary
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"   Models compared: {len(models_data)}")
    
    best_p = max(models_data.keys(), key=lambda x: models_data[x]['metrics']['p_detection_rate'])
    best_s = max(models_data.keys(), key=lambda x: models_data[x]['metrics']['s_detection_rate'])
    
    print(f"   Best P-wave detection: {models_data[best_p]['info']['type']} ({models_data[best_p]['metrics']['p_detection_rate']:.1f}%)")
    print(f"   Best S-wave detection: {models_data[best_s]['info']['type']} ({models_data[best_s]['metrics']['s_detection_rate']:.1f}%)")
    
    # Show improvement over original if available
    original_models = [m for m in models_data.keys() if 'Original' in models_data[m]['info']['type']]
    if original_models:
        original = original_models[0]
        print(f"\n💡 Improvement over original model:")
        
        for model_key in models_data.keys():
            if model_key != original:
                p_improvement = models_data[model_key]['metrics']['p_detection_rate'] - models_data[original]['metrics']['p_detection_rate']
                s_improvement = models_data[model_key]['metrics']['s_detection_rate'] - models_data[original]['metrics']['s_detection_rate']
                print(f"     {models_data[model_key]['info']['type']}: P={p_improvement:+.1f}%, S={s_improvement:+.1f}%")

if __name__ == '__main__':
    main() 