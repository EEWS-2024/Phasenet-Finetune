#!/bin/bash

# Script untuk menampilkan daftar model yang tersedia
# Berguna untuk menentukan model mana yang akan digunakan untuk resume training

echo "=================================================="
echo "📋 DAFTAR MODEL YANG TERSEDIA"
echo "=================================================="

# Function to display models in a directory
display_models() {
    local dir=$1
    local type=$2
    
    echo ""
    echo "🔹 $type Models:"
    echo "   Directory: $dir"
    echo ""
    
    if [ ! -d "$dir" ]; then
        echo "   ❌ Directory tidak ditemukan"
        return
    fi
    
    # Check if directory is empty
    if [ -z "$(ls -A $dir 2>/dev/null)" ]; then
        echo "   📭 Tidak ada model ditemukan"
        return
    fi
    
    # List models with details
    echo "   Available models (sorted by modification time):"
    echo "   ================================================"
    
    # Get model directories sorted by modification time (newest first)
    for model_dir in $(ls -t $dir); do
        if [ -d "$dir/$model_dir" ]; then
            echo "   📂 $model_dir"
            
            # Show creation time
            creation_time=$(stat -c %y "$dir/$model_dir" 2>/dev/null | cut -d' ' -f1,2 | cut -d'.' -f1)
            echo "      🕒 Created: $creation_time"
            
            # Show directory size
            size=$(du -sh "$dir/$model_dir" 2>/dev/null | cut -f1)
            echo "      💾 Size: $size"
            
            # Check for key files
            key_files=()
            if [ -f "$dir/$model_dir/checkpoint" ]; then
                key_files+=("checkpoint")
            fi
            if [ -f "$dir/$model_dir/config.json" ]; then
                key_files+=("config.json")
            fi
            if [ -f "$dir/$model_dir/model_summary.txt" ]; then
                key_files+=("model_summary.txt")
            fi
            
            if [ ${#key_files[@]} -gt 0 ]; then
                echo "      📄 Files: ${key_files[*]}"
            fi
            
            # Check for checkpoint files
            checkpoint_count=$(find "$dir/$model_dir" -name "*.ckpt*" 2>/dev/null | wc -l)
            if [ $checkpoint_count -gt 0 ]; then
                echo "      🎯 Checkpoints: $checkpoint_count files"
            fi
            
            echo ""
        fi
    done
}

# Display scratch models
display_models "model_indonesia/scratch" "Scratch Training"

# Display resume models
display_models "model_indonesia/resume" "Resume Training"

# Display any other model directories
for model_type in "model_indonesia"/*; do
    if [ -d "$model_type" ]; then
        type_name=$(basename "$model_type")
        if [ "$type_name" != "scratch" ] && [ "$type_name" != "resume" ]; then
            display_models "$model_type" "$(echo $type_name | tr '[:lower:]' '[:upper:]')"
        fi
    fi
done

echo "=================================================="
echo "📖 CARA PENGGUNAAN:"
echo ""
echo "1. Untuk menggunakan model terbaru secara otomatis:"
echo "   bash resume_training_indonesia.sh"
echo ""
echo "2. Untuk menggunakan model tertentu:"
echo "   bash resume_training_indonesia.sh <folder_model>"
echo ""
echo "   Contoh:"
echo "   bash resume_training_indonesia.sh 241215-123456"
echo ""
echo "3. Untuk melihat detail model tertentu:"
echo "   ls -la model_indonesia/scratch/<folder_model>/"
echo ""
echo "==================================================" 