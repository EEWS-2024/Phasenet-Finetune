#!/bin/bash

# Script untuk melanjutkan training PhaseNet Indonesia dengan 99% coverage
# Resume dari model pre-trained yang sudah bagus: 190703-214543

echo "=================================================="
echo "🔄 RESUME TRAINING PHASENET INDONESIA 99% COVERAGE"
echo "=================================================="

# Configuration
DATASET_DIR="dataset_phasenet_aug"
PRETRAINED_MODEL_DIR="model/190703-214543"
OUTPUT_MODEL_DIR="model_indonesia"

# Training parameters - OPTIMAL REALISTIC SETTINGS
EPOCHS=1                   # Full training epochs
BATCH_SIZE=8                 # Increased batch size for better training
LEARNING_RATE=0.0003         # Higher learning rate (3e-4) for faster convergence
DROP_RATE=0.15               # Standard dropout rate
WEIGHT_DECAY=0.0001          # Standard weight decay
SAVE_INTERVAL=5              # Save every 5 epochs

echo "🎯 Configuration:"
echo "   Dataset: $DATASET_DIR"
echo "   Pre-trained model: $PRETRAINED_MODEL_DIR"
echo "   Output model: $OUTPUT_MODEL_DIR"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LEARNING_RATE"
echo "   Dropout rate: $DROP_RATE"
echo "   Weight decay: $WEIGHT_DECAY"
echo "=================================================="

# Check if pre-trained model exists
if [ ! -d "$PRETRAINED_MODEL_DIR" ]; then
    echo "❌ Pre-trained model directory not found: $PRETRAINED_MODEL_DIR"
    exit 1
fi

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

echo "🚀 Starting resume training with realistic parameters..."

# Run training with realistic parameters
python train_indonesia.py \
    --train_dir="../$DATASET_DIR/npz_padded" \
    --train_list="../$DATASET_DIR/train_list_99pct.csv" \
    --valid_dir="../$DATASET_DIR/npz_padded" \
    --valid_list="../$DATASET_DIR/valid_list_99pct.csv" \
    --model_dir="../$OUTPUT_MODEL_DIR" \
    --load_model \
    --load_model_dir="../$PRETRAINED_MODEL_DIR" \
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
    echo "✅ RESUME TRAINING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Model fine-tuned tersimpan di: ../$OUTPUT_MODEL_DIR"
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
            --output_dir="../test_results_indonesia" \
            --batch_size=2 \
            --plot_results
        
        if [ $? -eq 0 ]; then
            echo "✅ Testing completed successfully!"
        else
            echo "⚠️  Testing failed, but training was successful"
        fi
    else
        echo "⚠️  Could not find trained model for testing"
    fi
    
    echo ""
    echo "=== FINE-TUNING SUMMARY ==="
    echo "Pre-trained model: 190703-214543 (pretrained model)"
    echo "Fine-tuned untuk: Data Indonesia"
    echo "Window size: 13500 samples (135 seconds)"
    echo "Batch size: $BATCH_SIZE"
    echo "Learning rate: $LEARNING_RATE"
    echo "Total epochs: $EPOCHS"
    echo "Output model: ../$OUTPUT_MODEL_DIR/$LATEST_MODEL"
    echo "========================="
else
    echo ""
    echo "❌ TRAINING FAILED!"
    echo "Check the error messages above for details."
    exit 1
fi 