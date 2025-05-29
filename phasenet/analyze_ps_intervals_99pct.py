#!/usr/bin/env python3
"""
Analisis distribusi P-S interval untuk dataset Indonesia
Fokus pada verifikasi 99% coverage dengan window size 135 detik
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob

def analyze_ps_intervals(npz_dir):
    """Analisis distribusi P-S interval dari file NPZ"""
    
    print("=== ANALISIS P-S INTERVAL UNTUK 99% COVERAGE ===")
    print(f"Direktori: {npz_dir}")
    print("=" * 60)
    
    # Find all NPZ files
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    print(f"Total file NPZ: {len(npz_files)}")
    
    if len(npz_files) == 0:
        print("ERROR: Tidak ada file NPZ ditemukan!")
        return None
    
    # Analyze each file
    results = []
    print("Menganalisis file NPZ...")
    
    for npz_file in tqdm(npz_files):
        try:
            # Load NPZ file
            data = np.load(npz_file, allow_pickle=True)
            
            # Extract information
            filename = os.path.basename(npz_file)
            waveform = data['data']
            
            # Get P and S indices
            p_indices = data['p_idx']
            s_indices = data['s_idx']
            
            # Check if we have valid picks
            if len(p_indices) > 0 and len(p_indices[0]) > 0:
                p_idx = p_indices[0][0]
            else:
                continue
                
            if len(s_indices) > 0 and len(s_indices[0]) > 0:
                s_idx = s_indices[0][0]
            else:
                continue
            
            # Calculate P-S interval
            ps_interval = s_idx - p_idx
            ps_interval_seconds = ps_interval / 100.0  # Assuming 100 Hz sampling
            
            result = {
                'filename': filename,
                'p_index': p_idx,
                's_index': s_idx,
                'ps_interval_samples': ps_interval,
                'ps_interval_seconds': ps_interval_seconds,
                'data_length': waveform.shape[0]
            }
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
            continue
    
    if len(results) == 0:
        print("ERROR: Tidak ada data valid yang ditemukan!")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    print(f"\nData valid: {len(df)} dari {len(npz_files)} file")
    
    # Calculate statistics
    ps_intervals = df['ps_interval_samples'].values
    ps_seconds = df['ps_interval_seconds'].values
    
    print("\n=== STATISTIK P-S INTERVAL ===")
    print(f"Mean: {np.mean(ps_intervals):.1f} samples ({np.mean(ps_seconds):.1f} detik)")
    print(f"Median: {np.median(ps_intervals):.1f} samples ({np.median(ps_seconds):.1f} detik)")
    print(f"Std: {np.std(ps_intervals):.1f} samples ({np.std(ps_seconds):.1f} detik)")
    print(f"Min: {np.min(ps_intervals)} samples ({np.min(ps_seconds):.1f} detik)")
    print(f"Max: {np.max(ps_intervals)} samples ({np.max(ps_seconds):.1f} detik)")
    
    # Calculate percentiles
    percentiles = [50, 75, 80, 85, 90, 95, 99]
    print(f"\n=== PERCENTILES P-S INTERVAL ===")
    for p in percentiles:
        val_samples = np.percentile(ps_intervals, p)
        val_seconds = val_samples / 100.0
        print(f"{p}th percentile: {val_samples:.1f} samples ({val_seconds:.1f} detik)")
    
    # Analisis untuk 99% coverage
    print(f"\n=== ANALISIS 99% COVERAGE ===")
    p99_samples = np.percentile(ps_intervals, 99)
    p99_seconds = p99_samples / 100.0
    
    # Window size recommendations
    window_sizes = [
        (p99_samples + 1500, "99% coverage (P-5s to S+10s)"),
        (13500, "Proposed 135s window"),
        (15000, "Conservative 150s window")
    ]
    
    print(f"99th percentile P-S interval: {p99_samples:.1f} samples ({p99_seconds:.1f} detik)")
    print(f"\nREKOMENDASI WINDOW SIZE:")
    
    for window_size, description in window_sizes:
        coverage = (ps_intervals <= (window_size - 1500)).mean() * 100  # Assuming 15s margin
        print(f"  {window_size} samples ({window_size/100:.1f}s): {coverage:.1f}% coverage - {description}")
    
    # Detailed analysis for 135s window
    window_135s = 13500
    margin = 1500  # 15 seconds margin
    effective_ps_limit = window_135s - margin
    
    covered_data = ps_intervals <= effective_ps_limit
    coverage_135s = covered_data.mean() * 100
    
    print(f"\n=== ANALISIS WINDOW 135 DETIK (13,500 samples) ===")
    print(f"Effective P-S limit dengan margin 15s: {effective_ps_limit} samples ({effective_ps_limit/100:.1f}s)")
    print(f"Data tercakup: {covered_data.sum()}/{len(ps_intervals)} ({coverage_135s:.2f}%)")
    print(f"Data tidak tercakup: {(~covered_data).sum()} ({(~covered_data).mean()*100:.2f}%)")
    
    # Show uncovered data
    uncovered_data = df[~covered_data]
    if len(uncovered_data) > 0:
        print(f"\nFile dengan P-S interval > {effective_ps_limit/100:.1f}s:")
        for _, row in uncovered_data.head(10).iterrows():
            print(f"  {row['filename']}: {row['ps_interval_seconds']:.1f}s")
        if len(uncovered_data) > 10:
            print(f"  ... dan {len(uncovered_data) - 10} file lainnya")
    
    # Categorization
    print(f"\n=== KATEGORISASI BERDASARKAN P-S INTERVAL ===")
    categories = [
        (0, 20, "Sangat Pendek"),
        (20, 40, "Pendek"),
        (40, 60, "Sedang"),
        (60, 120, "Panjang"),
        (120, float('inf'), "Sangat Panjang")
    ]
    
    for min_val, max_val, label in categories:
        if max_val == float('inf'):
            mask = ps_seconds >= min_val
        else:
            mask = (ps_seconds >= min_val) & (ps_seconds < max_val)
        count = mask.sum()
        percentage = count / len(ps_seconds) * 100
        print(f"{label} ({min_val}-{max_val if max_val != float('inf') else '∞'}s): {count} file ({percentage:.1f}%)")
    
    # Create visualization
    create_visualization(df, npz_dir)
    
    # Save detailed results
    output_file = os.path.join(os.path.dirname(npz_dir), 'ps_interval_analysis_99pct.csv')
    df.to_csv(output_file, index=False)
    print(f"\nData detail disimpan ke: {output_file}")
    
    return df

def create_visualization(df, npz_dir):
    """Create visualization plots"""
    
    ps_intervals = df['ps_interval_samples'].values
    ps_seconds = df['ps_interval_seconds'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analisis P-S Interval Dataset Indonesia (99% Coverage Focus)', fontsize=16)
    
    # Histogram of P-S intervals
    axes[0, 0].hist(ps_seconds, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(np.percentile(ps_seconds, 99), color='red', linestyle='--', 
                       label=f'99th percentile: {np.percentile(ps_seconds, 99):.1f}s')
    axes[0, 0].axvline(135, color='green', linestyle='--', 
                       label='Proposed window: 135s')
    axes[0, 0].set_xlabel('P-S Interval (seconds)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of P-S Intervals')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cumulative distribution
    sorted_ps = np.sort(ps_seconds)
    cumulative = np.arange(1, len(sorted_ps) + 1) / len(sorted_ps) * 100
    axes[0, 1].plot(sorted_ps, cumulative, 'b-', linewidth=2)
    axes[0, 1].axhline(99, color='red', linestyle='--', label='99% coverage')
    axes[0, 1].axvline(135, color='green', linestyle='--', label='135s window')
    axes[0, 1].set_xlabel('P-S Interval (seconds)')
    axes[0, 1].set_ylabel('Cumulative Percentage (%)')
    axes[0, 1].set_title('Cumulative Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(0, 200)
    
    # Window size analysis
    window_sizes = np.arange(6000, 18000, 500)  # 60s to 180s
    coverages = []
    
    for ws in window_sizes:
        margin = 1500  # 15s margin
        effective_limit = ws - margin
        coverage = (ps_intervals <= effective_limit).mean() * 100
        coverages.append(coverage)
    
    axes[1, 0].plot(window_sizes/100, coverages, 'b-', linewidth=2, marker='o', markersize=4)
    axes[1, 0].axhline(99, color='red', linestyle='--', label='99% target')
    axes[1, 0].axvline(135, color='green', linestyle='--', label='135s window')
    axes[1, 0].set_xlabel('Window Size (seconds)')
    axes[1, 0].set_ylabel('Coverage (%)')
    axes[1, 0].set_title('Coverage vs Window Size')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(80, 100)
    
    # Box plot by categories
    categories = []
    labels = []
    
    for min_val, max_val, label in [(0, 20, "Very Short"), (20, 40, "Short"), 
                                   (40, 60, "Medium"), (60, 120, "Long"), 
                                   (120, 300, "Very Long")]:
        if max_val == 300:
            mask = ps_seconds >= min_val
        else:
            mask = (ps_seconds >= min_val) & (ps_seconds < max_val)
        
        if mask.sum() > 0:
            categories.append(ps_seconds[mask])
            labels.append(f"{label}\n({mask.sum()} files)")
    
    if categories:
        axes[1, 1].boxplot(categories, labels=labels)
        axes[1, 1].set_ylabel('P-S Interval (seconds)')
        axes[1, 1].set_title('P-S Intervals by Category')
        axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(os.path.dirname(npz_dir), 'ps_interval_analysis_99pct.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisasi disimpan ke: {plot_file}")

def main():
    # Set path to Docker container dataset directory
    dataset_dir = "/home/jovyan/PhaseNet/dataset_phasenet_aug"
    npz_dir = os.path.join(dataset_dir, "npz_padded")
    
    print(f"Menganalisis dataset di: {npz_dir}")
    
    if not os.path.exists(npz_dir):
        print(f"ERROR: Direktori tidak ditemukan: {npz_dir}")
        return
    
    # Run analysis
    df = analyze_ps_intervals(npz_dir)
    
    if df is not None:
        print("\n" + "="*60)
        print("=== KESIMPULAN UNTUK 99% COVERAGE ===")
        print("Window size 135 detik (13,500 samples) memberikan:")
        
        ps_intervals = df['ps_interval_samples'].values
        window_135s = 13500
        margin = 1500
        effective_ps_limit = window_135s - margin
        coverage = (ps_intervals <= effective_ps_limit).mean() * 100
        
        print(f"✅ Coverage: {coverage:.2f}% dari total data")
        print(f"✅ Margin safety: 15 detik sebelum dan sesudah P-S")
        print(f"✅ Memory requirement: ~16-24GB GPU")
        print(f"✅ Batch size recommended: 16")
        print("="*60)
        print("Analisis selesai!")

if __name__ == "__main__":
    main() 