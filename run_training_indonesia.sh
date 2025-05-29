#!/bin/bash

# Script untuk menjalankan training PhaseNet Indonesia dengan 99% coverage
# Window size: 135 detik (13,500 samples) untuk menangkap 99% data Indonesia

echo "=================================================="
echo "🚀 TRAINING PHASENET INDONESIA 99% COVERAGE"
echo "=================================================="

# Configuration
DATASET_DIR="dataset_phasenet_aug"
OUTPUT_MODEL_DIR="model_indonesia"

# Training parameters - OPTIMAL REALISTIC SETTINGS
EPOCHS=1                   # Full training epochs
BATCH_SIZE=8                 # Increased batch size for better training
LEARNING_RATE=0.0003         # Higher learning rate (3e-4) for faster convergence
DROP_RATE=0.15               # Standard dropout rate
WEIGHT_DECAY=0.0001          # Standard weight decay
SAVE_INTERVAL=5              # Save every 5 epochs

# Testing parameters
MIN_PROB=0.05                # Minimum probability threshold for detection (lower = more sensitive)

echo "🎯 Configuration:"
echo "   Dataset: $DATASET_DIR"
echo "   Output model: $OUTPUT_MODEL_DIR"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LEARNING_RATE (optimized)"
echo "   Dropout rate: $DROP_RATE"
echo "   Weight decay: $WEIGHT_DECAY"
echo "   Testing min prob: $MIN_PROB"
echo "=================================================="

# Check if dataset exists
if [ ! -d "$DATASET_DIR/npz_padded" ]; then
    echo "❌ Dataset directory not found: $DATASET_DIR/npz_padded"
    exit 1
fi

# Check if CSV files exist
if [ ! -f "$DATASET_DIR/train_list_99pct.csv" ] || [ ! -f "$DATASET_DIR/valid_list_99pct.csv" ]; then
    echo "❌ CSV files not found. Please run prepare_data_split_99pct.py first"
    exit 1
fi

echo "✅ All prerequisites found"
echo ""

# Change to phasenet directory
cd phasenet

echo "🚀 Starting training from scratch with optimized parameters..."

# Run training
python train_indonesia.py \
    --train_dir="../$DATASET_DIR/npz_padded" \
    --train_list="../$DATASET_DIR/train_list_99pct.csv" \
    --valid_dir="../$DATASET_DIR/npz_padded" \
    --valid_list="../$DATASET_DIR/valid_list_99pct.csv" \
    --model_dir="../$OUTPUT_MODEL_DIR" \
    --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE \
    --learning_rate=$LEARNING_RATE \
    --drop_rate=$DROP_RATE \
    --weight_decay=$WEIGHT_DECAY \
    --save_interval=$SAVE_INTERVAL \
    --summary

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✅ TRAINING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Model tersimpan di: ../$OUTPUT_MODEL_DIR"
    echo ""
    echo "Running quick test on validation data..."
    
    # Find the latest model directory
    LATEST_MODEL=$(ls -t ../$OUTPUT_MODEL_DIR/ | head -n 1)
    
    if [ -n "$LATEST_MODEL" ]; then
        echo "Testing with latest model: $LATEST_MODEL"
        
        python test_indonesia.py \
            --test_dir="../$DATASET_DIR/npz_padded" \
            --test_list="../$DATASET_DIR/valid_list_99pct.csv" \
            --model_dir="../$OUTPUT_MODEL_DIR/$LATEST_MODEL" \
            --output_dir="../$OUTPUT_MODEL_DIR/$LATEST_MODEL/test_results" \
            --batch_size=2 \
            --plot_results \
            --min_prob=$MIN_PROB
        
        if [ $? -eq 0 ]; then
            echo "✅ Testing completed successfully!"
        else
            echo "⚠️  Testing failed, but training was successful"
        fi
    else
        echo "⚠️  Could not find trained model for testing"
    fi
    
    echo ""
    echo "=== TRAINING SUMMARY ==="
    echo "Training type: From scratch"
    echo "Data: Indonesia"
    echo "Window size: 13500 samples (135 seconds)"
    echo "Batch size: $BATCH_SIZE"
    echo "Learning rate: $LEARNING_RATE (optimized)"
    echo "Total epochs: $EPOCHS"
    echo "Output model: ../$OUTPUT_MODEL_DIR/$LATEST_MODEL"
    echo "Test results: ../$OUTPUT_MODEL_DIR/$LATEST_MODEL/test_results"
    echo "========================="
else
    echo ""
    echo "❌ TRAINING FAILED!"
    echo "Check the error messages above for details."
    exit 1
fi 