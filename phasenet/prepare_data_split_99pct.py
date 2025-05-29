#!/usr/bin/env python3
"""
Script untuk mempersiapkan data split untuk training PhaseNet Indonesia 99% coverage
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

def prepare_data_split():
    """Prepare data split untuk training dan validation"""
    
    # Set paths - Docker container environment
    dataset_dir = "/home/jovyan/PhaseNet/dataset_phasenet_aug"
    
    # Check if padded data list exists
    padded_data_list = os.path.join(dataset_dir, "padded_data_list.csv")
    
    if not os.path.exists(padded_data_list):
        print(f"ERROR: File tidak ditemukan: {padded_data_list}")
        print("Silakan jalankan script padding data terlebih dahulu")
        return False
    
    print("=== PERSIAPAN DATA SPLIT UNTUK 99% COVERAGE ===")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Data list: {padded_data_list}")
    
    # Load data list
    df = pd.read_csv(padded_data_list, header=None, names=['filename'])
    print(f"Total data: {len(df)} file")
    
    # Split data: 80% training, 20% validation
    train_files, valid_files = train_test_split(
        df['filename'].values, 
        test_size=0.2, 
        random_state=42,
        shuffle=True
    )
    
    print(f"Training data: {len(train_files)} file ({len(train_files)/len(df)*100:.1f}%)")
    print(f"Validation data: {len(valid_files)} file ({len(valid_files)/len(df)*100:.1f}%)")
    
    # Save training list
    train_list_file = os.path.join(dataset_dir, "padded_train_list_99pct.csv")
    train_df = pd.DataFrame(train_files, columns=['fname'])
    train_df.to_csv(train_list_file, index=False, header=True)
    print(f"Training list saved: {train_list_file}")
    
    # Save validation list
    valid_list_file = os.path.join(dataset_dir, "padded_valid_list_99pct.csv")
    valid_df = pd.DataFrame(valid_files, columns=['fname'])
    valid_df.to_csv(valid_list_file, index=False, header=True)
    print(f"Validation list saved: {valid_list_file}")
    
    # Verify files exist
    npz_dir = os.path.join(dataset_dir, "npz_padded")
    missing_files = []
    
    print("\nVerifying file existence...")
    for filename in list(train_files) + list(valid_files):
        file_path = os.path.join(npz_dir, filename)
        if not os.path.exists(file_path):
            missing_files.append(filename)
    
    if missing_files:
        print(f"WARNING: {len(missing_files)} file tidak ditemukan:")
        for f in missing_files[:10]:  # Show first 10
            print(f"  {f}")
        if len(missing_files) > 10:
            print(f"  ... dan {len(missing_files) - 10} file lainnya")
    else:
        print("✅ Semua file ditemukan!")
    
    print("\n=== DATA SPLIT SUMMARY ===")
    print(f"Total files: {len(df)}")
    print(f"Training: {len(train_files)} files")
    print(f"Validation: {len(valid_files)} files")
    print(f"Missing: {len(missing_files)} files")
    print("=" * 40)
    
    return True

def main():
    success = prepare_data_split()
    
    if success:
        print("\n✅ Data split preparation completed successfully!")
        print("\nNext steps:")
        print("1. Run analysis: python3 analyze_ps_intervals_99pct.py")
        print("2. Start training: bash ../run_training_indonesia.sh")
    else:
        print("\n❌ Data split preparation failed!")

if __name__ == "__main__":
    main() 