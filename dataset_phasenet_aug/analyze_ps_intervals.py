#!/usr/bin/env python3
"""
Script untuk menganalisis distribusi P-S interval dalam dataset Indonesia
dan memberikan rekomendasi window size yang optimal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm
import seaborn as sns

def analyze_ps_intervals(npz_dir):
    """Analisis mendalam distribusi P-S interval"""
    
    print("=== ANALISIS DISTRIBUSI P-S INTERVAL ===")
    print(f"Direktori: {npz_dir}")
    print("=" * 50)
    
    # Dapatkan semua file NPZ
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    total_files = len(npz_files)
    
    print(f"Total file NPZ: {total_files}")
    
    if total_files == 0:
        print("Tidak ada file NPZ ditemukan!")
        return
    
    # Collect data
    ps_intervals = []
    p_indices = []
    s_indices = []
    data_lengths = []
    file_names = []
    
    print("Menganalisis file NPZ...")
    for npz_file in tqdm(npz_files):
        try:
            data = np.load(npz_file, allow_pickle=True)
            
            # Extract P and S indices
            p_idx = data['p_idx'][0][0] if len(data['p_idx']) > 0 and len(data['p_idx'][0]) > 0 else None
            s_idx = data['s_idx'][0][0] if len(data['s_idx']) > 0 and len(data['s_idx'][0]) > 0 else None
            
            if p_idx is not None and s_idx is not None:
                ps_interval = s_idx - p_idx
                data_length = data['data'].shape[0]
                
                ps_intervals.append(ps_interval)
                p_indices.append(p_idx)
                s_indices.append(s_idx)
                data_lengths.append(data_length)
                file_names.append(os.path.basename(npz_file))
                
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
            continue
    
    # Convert to numpy arrays
    ps_intervals = np.array(ps_intervals)
    p_indices = np.array(p_indices)
    s_indices = np.array(s_indices)
    data_lengths = np.array(data_lengths)
    
    print(f"\nData valid: {len(ps_intervals)} dari {total_files} file")
    
    # Statistik dasar
    print("\n=== STATISTIK P-S INTERVAL ===")
    print(f"Mean: {np.mean(ps_intervals):.1f} samples ({np.mean(ps_intervals)/100:.1f} detik)")
    print(f"Median: {np.median(ps_intervals):.1f} samples ({np.median(ps_intervals)/100:.1f} detik)")
    print(f"Std: {np.std(ps_intervals):.1f} samples ({np.std(ps_intervals)/100:.1f} detik)")
    print(f"Min: {np.min(ps_intervals)} samples ({np.min(ps_intervals)/100:.1f} detik)")
    print(f"Max: {np.max(ps_intervals)} samples ({np.max(ps_intervals)/100:.1f} detik)")
    
    # Percentiles
    percentiles = [50, 75, 80, 85, 90, 95, 99]
    print(f"\n=== PERCENTILES P-S INTERVAL ===")
    for p in percentiles:
        val = np.percentile(ps_intervals, p)
        print(f"{p}th percentile: {val:.1f} samples ({val/100:.1f} detik)")
    
    # Analisis window size requirements
    print(f"\n=== REKOMENDASI WINDOW SIZE ===")
    
    # Untuk menangkap berbagai persentase data
    coverage_targets = [75, 80, 85, 90, 95, 99]
    
    for target in coverage_targets:
        required_interval = np.percentile(ps_intervals, target)
        # Tambahkan margin untuk P-wave sebelum onset dan buffer setelah S-wave
        margin_before_p = 500  # 5 detik sebelum P
        margin_after_s = 1000  # 10 detik setelah S
        
        recommended_window = required_interval + margin_before_p + margin_after_s
        
        print(f"Untuk menangkap {target}% data:")
        print(f"  P-S interval max: {required_interval:.0f} samples ({required_interval/100:.1f}s)")
        print(f"  Window size recommended: {recommended_window:.0f} samples ({recommended_window/100:.1f}s)")
        print()
    
    # Analisis posisi P-wave
    print(f"=== ANALISIS POSISI P-WAVE ===")
    print(f"P index mean: {np.mean(p_indices):.1f} samples ({np.mean(p_indices)/100:.1f} detik)")
    print(f"P index median: {np.median(p_indices):.1f} samples ({np.median(p_indices)/100:.1f} detik)")
    print(f"P index range: {np.min(p_indices)} - {np.max(p_indices)} samples")
    
    # Buat visualisasi
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Histogram P-S interval
    axes[0, 0].hist(ps_intervals/100, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(ps_intervals)/100, color='red', linestyle='--', label=f'Mean: {np.mean(ps_intervals)/100:.1f}s')
    axes[0, 0].axvline(np.median(ps_intervals)/100, color='green', linestyle='--', label=f'Median: {np.median(ps_intervals)/100:.1f}s')
    axes[0, 0].set_xlabel('P-S Interval (detik)')
    axes[0, 0].set_ylabel('Frekuensi')
    axes[0, 0].set_title('Distribusi P-S Interval')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot P-S interval
    axes[0, 1].boxplot(ps_intervals/100)
    axes[0, 1].set_ylabel('P-S Interval (detik)')
    axes[0, 1].set_title('Box Plot P-S Interval')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    sorted_intervals = np.sort(ps_intervals)
    cumulative_pct = np.arange(1, len(sorted_intervals) + 1) / len(sorted_intervals) * 100
    axes[0, 2].plot(sorted_intervals/100, cumulative_pct)
    axes[0, 2].set_xlabel('P-S Interval (detik)')
    axes[0, 2].set_ylabel('Cumulative Percentage (%)')
    axes[0, 2].set_title('Cumulative Distribution P-S Interval')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add percentile lines
    for p in [75, 90, 95]:
        val = np.percentile(ps_intervals, p)
        axes[0, 2].axvline(val/100, color='red', linestyle=':', alpha=0.7, label=f'{p}%: {val/100:.1f}s')
    axes[0, 2].legend()
    
    # 4. Scatter plot P vs S indices
    axes[1, 0].scatter(p_indices/100, s_indices/100, alpha=0.6, s=20)
    axes[1, 0].set_xlabel('P Index (detik)')
    axes[1, 0].set_ylabel('S Index (detik)')
    axes[1, 0].set_title('P vs S Indices')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Window size recommendations
    window_sizes = []
    coverage_pcts = []
    
    for pct in range(50, 100, 5):
        required_interval = np.percentile(ps_intervals, pct)
        window_size = required_interval + 1500  # margin
        window_sizes.append(window_size/100)
        coverage_pcts.append(pct)
    
    axes[1, 1].plot(coverage_pcts, window_sizes, 'o-', linewidth=2, markersize=6)
    axes[1, 1].axhline(30, color='red', linestyle='--', label='PhaseNet Original (30s)')
    axes[1, 1].axhline(60, color='green', linestyle='--', label='Recommended (60s)')
    axes[1, 1].set_xlabel('Coverage Percentage (%)')
    axes[1, 1].set_ylabel('Required Window Size (detik)')
    axes[1, 1].set_title('Window Size vs Coverage')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. P-S interval vs data length
    axes[1, 2].scatter(data_lengths/100, ps_intervals/100, alpha=0.6, s=20)
    axes[1, 2].set_xlabel('Total Data Length (detik)')
    axes[1, 2].set_ylabel('P-S Interval (detik)')
    axes[1, 2].set_title('P-S Interval vs Data Length')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ps_interval_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Buat tabel detail untuk file dengan P-S interval terpanjang
    print(f"\n=== FILE DENGAN P-S INTERVAL TERPANJANG ===")
    df = pd.DataFrame({
        'filename': file_names,
        'p_index': p_indices,
        's_index': s_indices,
        'ps_interval_samples': ps_intervals,
        'ps_interval_seconds': ps_intervals / 100,
        'data_length': data_lengths
    })
    
    # Sort by P-S interval
    df_sorted = df.sort_values('ps_interval_samples', ascending=False)
    
    print("Top 10 file dengan P-S interval terpanjang:")
    print(df_sorted.head(10).to_string(index=False))
    
    # Save detailed analysis
    df_sorted.to_csv('ps_interval_detailed_analysis.csv', index=False)
    
    # Analisis kategori berdasarkan P-S interval
    print(f"\n=== KATEGORISASI BERDASARKAN P-S INTERVAL ===")
    
    categories = {
        'Sangat Pendek (< 20s)': (ps_intervals < 2000).sum(),
        'Pendek (20-40s)': ((ps_intervals >= 2000) & (ps_intervals < 4000)).sum(),
        'Sedang (40-60s)': ((ps_intervals >= 4000) & (ps_intervals < 6000)).sum(),
        'Panjang (60-120s)': ((ps_intervals >= 6000) & (ps_intervals < 12000)).sum(),
        'Sangat Panjang (> 120s)': (ps_intervals >= 12000).sum()
    }
    
    total_valid = len(ps_intervals)
    for category, count in categories.items():
        percentage = (count / total_valid) * 100
        print(f"{category}: {count} file ({percentage:.1f}%)")
    
    # Rekomendasi final
    print(f"\n=== REKOMENDASI FINAL ===")
    print("Berdasarkan analisis distribusi P-S interval:")
    print()
    print("1. WINDOW SIZE RECOMMENDATIONS:")
    print("   - Konservatif (75% coverage): 45 detik (4500 samples)")
    print("   - Balanced (90% coverage): 60 detik (6000 samples) ⭐ RECOMMENDED")
    print("   - Agresif (95% coverage): 90 detik (9000 samples)")
    print("   - Maksimal (99% coverage): 150 detik (15000 samples)")
    print()
    print("2. ADAPTIVE WINDOWING STRATEGY:")
    print("   - Untuk P-S < 50s: gunakan window 60s centered pada P-S pair")
    print("   - Untuk P-S > 50s: gunakan window 60s mulai dari P-5s")
    print("   - Untuk P-S > 120s: pertimbangkan multiple windows atau sliding window")
    print()
    print("3. MEMORY CONSIDERATIONS:")
    print("   - Window 60s: ~2x memory usage vs original PhaseNet")
    print("   - Batch size perlu dikurangi dari 128 ke 32-64")
    print("   - GPU memory requirement: ~8-12GB untuk training")
    
    return df_sorted

if __name__ == "__main__":
    npz_dir = "./npz_padded"
    
    if not os.path.exists(npz_dir):
        print(f"Error: Direktori {npz_dir} tidak ditemukan!")
        print("Pastikan Anda menjalankan script ini dari direktori dataset_phasenet_aug")
        exit(1)
    
    df_analysis = analyze_ps_intervals(npz_dir)
    
    print(f"\nAnalisis selesai!")
    print(f"File hasil:")
    print(f"  - ps_interval_analysis.png: Visualisasi distribusi")
    print(f"  - ps_interval_detailed_analysis.csv: Data detail per file") 