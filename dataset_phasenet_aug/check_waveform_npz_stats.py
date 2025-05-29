#!/usr/bin/env python3
"""
Script untuk menganalisis statistik file NPZ yang sudah di-padding
untuk dataset PhaseNet Indonesia
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import json

def analyze_npz_files(npz_dir):
    """Analisis lengkap file NPZ yang sudah di-padding"""
    
    print("=== ANALISIS FILE NPZ PADDED UNTUK PHASENET ===")
    print(f"Direktori: {npz_dir}")
    print("=" * 60)
    
    # Dapatkan semua file NPZ
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    total_files = len(npz_files)
    
    print(f"Total file NPZ: {total_files}")
    
    if total_files == 0:
        print("Tidak ada file NPZ ditemukan!")
        return
    
    # Inisialisasi lists untuk statistik
    data_stats = []
    p_indices = []
    s_indices = []
    stations = []
    channels = []
    channel_types = []
    data_shapes = []
    is_augmented_list = []
    original_channels = []
    
    print("\nMemproses file NPZ...")
    
    for npz_file in tqdm(npz_files, desc="Analyzing NPZ files"):
        try:
            # Load NPZ file
            data = np.load(npz_file, allow_pickle=True)
            
            # Extract informasi dasar
            waveform_data = data['data']
            p_idx = data['p_idx']
            s_idx = data['s_idx']
            station_id = str(data['station_id'])
            
            # Extract informasi tambahan jika ada
            channel = str(data.get('channel', 'unknown'))
            channel_type = str(data.get('channel_type', 'unknown'))
            is_augmented = bool(data.get('is_augmented', False))
            original_channel = str(data.get('original_channel', ''))
            
            # Statistik data waveform
            data_shape = waveform_data.shape
            data_length = data_shape[0]
            num_components = data_shape[1] if len(data_shape) > 1 else 1
            
            # Statistik amplitudo
            data_min = np.min(waveform_data)
            data_max = np.max(waveform_data)
            data_mean = np.mean(waveform_data)
            data_std = np.std(waveform_data)
            
            # P dan S indices
            p_sample = p_idx[0][0] if len(p_idx) > 0 and len(p_idx[0]) > 0 else -1
            s_sample = s_idx[0][0] if len(s_idx) > 0 and len(s_idx[0]) > 0 else -1
            
            # Simpan statistik
            data_stats.append({
                'filename': os.path.basename(npz_file),
                'station': station_id,
                'channel': channel,
                'channel_type': channel_type,
                'is_augmented': is_augmented,
                'original_channel': original_channel,
                'data_length': data_length,
                'num_components': num_components,
                'data_shape': str(data_shape),
                'p_index': p_sample,
                's_index': s_sample,
                'p_s_interval': s_sample - p_sample if s_sample > p_sample else -1,
                'data_min': data_min,
                'data_max': data_max,
                'data_mean': data_mean,
                'data_std': data_std,
                'data_range': data_max - data_min
            })
            
            # Collect untuk analisis agregat
            p_indices.append(p_sample)
            s_indices.append(s_sample)
            stations.append(station_id)
            channels.append(channel)
            channel_types.append(channel_type)
            data_shapes.append(data_shape)
            is_augmented_list.append(is_augmented)
            original_channels.append(original_channel)
            
        except Exception as e:
            print(f"Error processing {npz_file}: {str(e)}")
    
    # Convert ke DataFrame
    df_stats = pd.DataFrame(data_stats)
    
    # === STATISTIK DESKRIPTIF ===
    print("\n=== STATISTIK DESKRIPTIF DATA ===")
    numeric_cols = ['data_length', 'num_components', 'p_index', 's_index', 'p_s_interval', 
                   'data_min', 'data_max', 'data_mean', 'data_std', 'data_range']
    print(df_stats[numeric_cols].describe())
    
    # === DISTRIBUSI PANJANG DATA ===
    print("\n=== DISTRIBUSI PANJANG DATA ===")
    length_counts = df_stats['data_length'].value_counts().sort_index()
    print(length_counts)
    
    # === DISTRIBUSI STASIUN ===
    print("\n=== DISTRIBUSI PER STASIUN ===")
    station_counts = df_stats['station'].value_counts()
    print(station_counts)
    
    # === DISTRIBUSI CHANNEL TYPE ===
    print("\n=== DISTRIBUSI PER CHANNEL TYPE ===")
    channel_type_counts = df_stats['channel_type'].value_counts()
    print(channel_type_counts)
    
    # === DISTRIBUSI DATA ORIGINAL VS AUGMENTED ===
    print("\n=== DISTRIBUSI ORIGINAL VS AUGMENTED ===")
    augmented_counts = df_stats['is_augmented'].value_counts()
    print(f"Original data: {augmented_counts.get(False, 0)}")
    print(f"Augmented data: {augmented_counts.get(True, 0)}")
    
    # === ANALISIS P DAN S INDICES ===
    print("\n=== ANALISIS P DAN S INDICES ===")
    valid_p = df_stats[df_stats['p_index'] >= 0]
    valid_s = df_stats[df_stats['s_index'] >= 0]
    valid_ps = df_stats[(df_stats['p_index'] >= 0) & (df_stats['s_index'] >= 0)]
    
    print(f"File dengan P index valid: {len(valid_p)}")
    print(f"File dengan S index valid: {len(valid_s)}")
    print(f"File dengan P dan S index valid: {len(valid_ps)}")
    
    if len(valid_ps) > 0:
        print(f"P index - Min: {valid_ps['p_index'].min()}, Max: {valid_ps['p_index'].max()}, Mean: {valid_ps['p_index'].mean():.1f}")
        print(f"S index - Min: {valid_ps['s_index'].min()}, Max: {valid_ps['s_index'].max()}, Mean: {valid_ps['s_index'].mean():.1f}")
        print(f"P-S interval - Min: {valid_ps['p_s_interval'].min()}, Max: {valid_ps['p_s_interval'].max()}, Mean: {valid_ps['p_s_interval'].mean():.1f}")
    
    # === VALIDASI PADDING ===
    print("\n=== VALIDASI PADDING (P INDEX >= 3000) ===")
    p_below_3000 = df_stats[df_stats['p_index'] < 3000]
    p_above_3000 = df_stats[df_stats['p_index'] >= 3000]
    
    print(f"File dengan P index < 3000: {len(p_below_3000)}")
    print(f"File dengan P index >= 3000: {len(p_above_3000)}")
    
    if len(p_below_3000) > 0:
        print("⚠️  WARNING: Ada file dengan P index < 3000!")
        print("File yang bermasalah:")
        for _, row in p_below_3000.iterrows():
            print(f"  - {row['filename']}: P index = {row['p_index']}")
    else:
        print("✅ Semua file memiliki P index >= 3000 (padding berhasil)")
    
    # === KONSISTENSI SHAPE ===
    print("\n=== KONSISTENSI SHAPE DATA ===")
    unique_shapes = df_stats['data_shape'].value_counts()
    print("Distribusi shape data:")
    for shape, count in unique_shapes.items():
        print(f"  {shape}: {count} files")
    
    if len(unique_shapes) == 1:
        print("✅ Semua file memiliki shape yang konsisten")
    else:
        print("⚠️  WARNING: Ada variasi dalam shape data!")
    
    # === KUALITAS DATA ===
    print("\n=== KUALITAS DATA ===")
    
    # Check untuk data yang semua nol
    zero_data = df_stats[df_stats['data_range'] == 0]
    print(f"File dengan data semua nol: {len(zero_data)}")
    
    # Check untuk data dengan range sangat kecil (mungkin masalah normalisasi)
    small_range = df_stats[df_stats['data_range'] < 1e-10]
    print(f"File dengan range data sangat kecil: {len(small_range)}")
    
    # Check untuk outliers dalam amplitudo
    q1 = df_stats['data_range'].quantile(0.25)
    q3 = df_stats['data_range'].quantile(0.75)
    iqr = q3 - q1
    outliers = df_stats[(df_stats['data_range'] < q1 - 1.5*iqr) | (df_stats['data_range'] > q3 + 1.5*iqr)]
    print(f"File dengan range data outlier: {len(outliers)}")
    
    # === SIMPAN HASIL ANALISIS ===
    print("\n=== MENYIMPAN HASIL ANALISIS ===")
    
    # Simpan statistik detail ke CSV
    output_csv = os.path.join(os.path.dirname(npz_dir), "npz_padded_stats.csv")
    df_stats.to_csv(output_csv, index=False)
    print(f"Statistik detail disimpan ke: {output_csv}")
    
    # Simpan ringkasan ke JSON
    summary = {
        "total_files": total_files,
        "data_length_stats": {
            "min": int(df_stats['data_length'].min()),
            "max": int(df_stats['data_length'].max()),
            "mean": float(df_stats['data_length'].mean()),
            "unique_lengths": df_stats['data_length'].unique().tolist()
        },
        "p_index_stats": {
            "min": int(valid_ps['p_index'].min()) if len(valid_ps) > 0 else None,
            "max": int(valid_ps['p_index'].max()) if len(valid_ps) > 0 else None,
            "mean": float(valid_ps['p_index'].mean()) if len(valid_ps) > 0 else None,
            "files_below_3000": len(p_below_3000)
        },
        "s_index_stats": {
            "min": int(valid_ps['s_index'].min()) if len(valid_ps) > 0 else None,
            "max": int(valid_ps['s_index'].max()) if len(valid_ps) > 0 else None,
            "mean": float(valid_ps['s_index'].mean()) if len(valid_ps) > 0 else None
        },
        "station_distribution": station_counts.to_dict(),
        "channel_type_distribution": channel_type_counts.to_dict(),
        "augmentation_stats": {
            "original": int(augmented_counts.get(False, 0)),
            "augmented": int(augmented_counts.get(True, 0))
        },
        "data_quality": {
            "zero_data_files": len(zero_data),
            "small_range_files": len(small_range),
            "outlier_files": len(outliers)
        }
    }
    
    summary_json = os.path.join(os.path.dirname(npz_dir), "npz_padded_summary.json")
    with open(summary_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Ringkasan disimpan ke: {summary_json}")
    
    # === BUAT VISUALISASI ===
    print("\n=== MEMBUAT VISUALISASI ===")
    create_visualizations(df_stats, os.path.dirname(npz_dir))
    
    print("\n=== ANALISIS SELESAI ===")
    return df_stats

def create_visualizations(df_stats, output_dir):
    """Buat visualisasi untuk analisis NPZ"""
    
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    # 1. Distribusi P dan S indices
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    valid_ps = df_stats[(df_stats['p_index'] >= 0) & (df_stats['s_index'] >= 0)]
    plt.hist(valid_ps['p_index'], bins=50, alpha=0.7, color='blue')
    plt.axvline(3000, color='red', linestyle='--', label='Minimum P index (3000)')
    plt.xlabel('P Index')
    plt.ylabel('Frequency')
    plt.title('Distribusi P Index')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.hist(valid_ps['s_index'], bins=50, alpha=0.7, color='red')
    plt.xlabel('S Index')
    plt.ylabel('Frequency')
    plt.title('Distribusi S Index')
    
    plt.subplot(1, 3, 3)
    plt.hist(valid_ps['p_s_interval'], bins=50, alpha=0.7, color='green')
    plt.xlabel('P-S Interval (samples)')
    plt.ylabel('Frequency')
    plt.title('Distribusi P-S Interval')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'npz_indices_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Distribusi per stasiun
    plt.figure(figsize=(12, 6))
    station_counts = df_stats['station'].value_counts()
    station_counts.plot(kind='bar')
    plt.title('Distribusi File NPZ per Stasiun')
    plt.xlabel('Stasiun')
    plt.ylabel('Jumlah File')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'npz_station_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Distribusi channel type
    plt.figure(figsize=(10, 6))
    channel_counts = df_stats['channel_type'].value_counts()
    plt.pie(channel_counts.values, labels=channel_counts.index, autopct='%1.1f%%')
    plt.title('Distribusi Channel Type')
    plt.savefig(os.path.join(fig_dir, 'npz_channel_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Original vs Augmented
    plt.figure(figsize=(8, 6))
    aug_counts = df_stats['is_augmented'].value_counts()
    labels = ['Original', 'Augmented']
    plt.pie(aug_counts.values, labels=labels, autopct='%1.1f%%')
    plt.title('Distribusi Data Original vs Augmented')
    plt.savefig(os.path.join(fig_dir, 'npz_augmentation_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Data quality metrics
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(df_stats['data_range'], bins=50, alpha=0.7)
    plt.xlabel('Data Range')
    plt.ylabel('Frequency')
    plt.title('Distribusi Range Data')
    plt.yscale('log')
    
    plt.subplot(1, 3, 2)
    plt.hist(df_stats['data_std'], bins=50, alpha=0.7, color='orange')
    plt.xlabel('Standard Deviation')
    plt.ylabel('Frequency')
    plt.title('Distribusi Std Dev Data')
    
    plt.subplot(1, 3, 3)
    plt.scatter(df_stats['data_mean'], df_stats['data_std'], alpha=0.5)
    plt.xlabel('Mean')
    plt.ylabel('Standard Deviation')
    plt.title('Mean vs Std Dev')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'npz_data_quality.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisasi disimpan di: {fig_dir}")

def main():
    """Fungsi utama"""
    # Direktori NPZ padded
    npz_dir = "./npz_padded"
    
    if not os.path.exists(npz_dir):
        print(f"Direktori {npz_dir} tidak ditemukan!")
        print("Pastikan Anda menjalankan script ini dari direktori dataset_phasenet_aug")
        return
    
    # Jalankan analisis
    df_stats = analyze_npz_files(npz_dir)
    
    if df_stats is not None:
        print(f"\nAnalisis selesai! Total {len(df_stats)} file NPZ dianalisis.")
        print("File output:")
        print("- npz_padded_stats.csv: Statistik detail per file")
        print("- npz_padded_summary.json: Ringkasan statistik")
        print("- figures/: Visualisasi hasil analisis")

if __name__ == "__main__":
    main() 