#!/bin/bash

# Script untuk Transfer Learning PhaseNet Indonesia
# Menggunakan pretrained model 190703-214543 (30s) untuk model 135s

echo "=================================================="
echo "🔄 TRANSFER LEARNING PHASENET INDONESIA"
echo "=================================================="

# Configuration
DATASET_DIR="dataset_phasenet_aug"
PRETRAINED_MODEL_DIR="model/190703-214543"  # Model bagus yang harus digunakan
OUTPUT_MODEL_DIR="model_indonesia/transfer_learning"

# Training parameters - EXTREMELY DEFENSIVE untuk guaranteed success
EPOCHS=100                   # More epochs karena learning rate sangat rendah
BATCH_SIZE=1                 # SINGLE sample per batch (most stable possible)
LEARNING_RATE=0.0000005      # EXTREMELY LOW learning rate (5e-7) for absolute stability
DROP_RATE=0.02               # Almost no dropout untuk maximum stability
WEIGHT_DECAY=0.000001        # Minimal weight decay
SAVE_INTERVAL=10             # Save every 10 epochs

# Testing parameters
MIN_PROB=0.05                # Minimum probability threshold for detection

echo "🎯 Configuration:"
echo "   Dataset: $DATASET_DIR"
echo "   Pretrained model: $PRETRAINED_MODEL_DIR (30s model -> 135s)"
echo "   Output model: $OUTPUT_MODEL_DIR"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE (SINGLE sample per batch)"
echo "   Learning rate: $LEARNING_RATE (EXTREMELY LOW for transfer)"
echo "   Dropout rate: $DROP_RATE"
echo "   Weight decay: $WEIGHT_DECAY"
echo "   Testing min prob: $MIN_PROB"
echo "=================================================="

# Check if pretrained model exists
if [ ! -d "$PRETRAINED_MODEL_DIR" ]; then
    echo "❌ Pretrained model directory not found: $PRETRAINED_MODEL_DIR"
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

# Create output directories
mkdir -p "$OUTPUT_MODEL_DIR"

# Change to phasenet directory
cd phasenet

echo "🚀 Starting TRANSFER LEARNING dengan architecture adaptation..."
echo "   Loading weights dari model 30s ke architecture 135s"
echo "   Ini memerlukan partial loading dengan skip incompatible layers"

# Run transfer learning dengan VERY SAFE parameters
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
    echo "✅ TRANSFER LEARNING COMPLETED SUCCESSFULLY!"
    echo ""
    echo "Model transfer learned tersimpan di: ../$OUTPUT_MODEL_DIR"
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
            --batch_size=1 \
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
    echo "=== TRANSFER LEARNING SUMMARY ==="
    echo "Training type: Transfer Learning (30s -> 135s)"
    echo "Base model: 190703-214543 (pretrained 30s model)"
    echo "Target: Indonesia 135s windows"
    echo "Window size: 13500 samples (135 seconds)"
    echo "Batch size: $BATCH_SIZE (SINGLE sample per batch)"
    echo "Learning rate: $LEARNING_RATE (EXTREMELY LOW)"
    echo "Total epochs: $EPOCHS"
    echo "Output model: ../$OUTPUT_MODEL_DIR/$LATEST_MODEL"
    echo "Test results: ../$OUTPUT_MODEL_DIR/$LATEST_MODEL/test_results"
    echo "========================="
else
    echo ""
    echo "❌ TRANSFER LEARNING FAILED!"
    echo "Check the error messages above for details."
    exit 1
fi 