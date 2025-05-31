#!/bin/bash

# Script untuk melanjutkan training PhaseNet Indonesia
# Resume dari model scratch yang sudah berhasil
# 
# Usage:
#   bash resume_training_indonesia.sh                    # Use latest model automatically
#   bash resume_training_indonesia.sh model_folder_name  # Use specific model folder

echo "=================================================="
echo "🔄 RESUME TRAINING PHASENET INDONESIA"
echo "=================================================="

# Configuration
DATASET_DIR="dataset_phasenet_aug"

# Check if specific model folder is provided as argument
if [ $# -eq 1 ]; then
    # Use user-specified model folder
    SPECIFIED_MODEL_FOLDER="$1"
    PRETRAINED_MODEL_DIR="model_indonesia/scratch/$SPECIFIED_MODEL_FOLDER"
    
    # Verify that the specified model exists
    if [ ! -d "$PRETRAINED_MODEL_DIR" ]; then
        echo "❌ Specified model directory not found: $PRETRAINED_MODEL_DIR"
        echo ""
        echo "Available models in model_indonesia/scratch/:"
        if [ -d "model_indonesia/scratch" ]; then
            ls -la model_indonesia/scratch/
        else
            echo "   (no models found)"
        fi
        echo ""
        echo "Usage:"
        echo "   bash resume_training_indonesia.sh                    # Use latest model automatically"
        echo "   bash resume_training_indonesia.sh model_folder_name  # Use specific model folder"
        exit 1
    fi
    
    echo "🎯 Using SPECIFIED model: $SPECIFIED_MODEL_FOLDER"
else
    # Use latest model automatically (original behavior)
    if [ ! -d "model_indonesia/scratch" ]; then
        echo "❌ No scratch models directory found: model_indonesia/scratch"
        echo "Please run training from scratch first using: bash run_training_scratch_indonesia.sh"
        exit 1
    fi
    
    LATEST_SCRATCH_MODEL=$(ls -t model_indonesia/scratch/ | head -n 1)
    
    if [ -z "$LATEST_SCRATCH_MODEL" ]; then
        echo "❌ No models found in model_indonesia/scratch/"
        echo "Please run training from scratch first using: bash run_training_scratch_indonesia.sh"
        exit 1
    fi
    
    PRETRAINED_MODEL_DIR="model_indonesia/scratch/$LATEST_SCRATCH_MODEL"
    echo "🎯 Using LATEST model automatically: $LATEST_SCRATCH_MODEL"
fi

OUTPUT_MODEL_DIR="model_indonesia/resume"

# Training parameters
EPOCHS=2                    
BATCH_SIZE=4               
LEARNING_RATE=0.00001      
DROP_RATE=0.15               # Standard dropout rate
WEIGHT_DECAY=0.0001          # Standard weight decay
SAVE_INTERVAL=5              # Save every 5 epochs

# Testing parameters
MIN_PROB=0.05                # Minimum probability threshold for detection

echo ""
echo "🎯 Configuration:"
echo "   Dataset: $DATASET_DIR"
echo "   Base model directory: $PRETRAINED_MODEL_DIR"
echo "   Output model: $OUTPUT_MODEL_DIR"
echo "   Epochs: $EPOCHS"
echo "   Batch size: $BATCH_SIZE"
echo "   Learning rate: $LEARNING_RATE"
echo "   Dropout rate: $DROP_RATE"
echo "   Weight decay: $WEIGHT_DECAY"
echo "   Testing min prob: $MIN_PROB"
echo "=================================================="

# Check if base model exists
if [ ! -d "$PRETRAINED_MODEL_DIR" ]; then
    echo "❌ Base model directory not found: $PRETRAINED_MODEL_DIR"
    echo "Available models:"
    if [ -d "model_indonesia/scratch" ]; then
        ls -la model_indonesia/scratch/
    else
        echo "   (no models found)"
    fi
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

echo "🚀 Starting resume training with SAFE parameters..."

# Run training with SAFE parameters
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
    echo "Model resumed tersimpan di: ../$OUTPUT_MODEL_DIR"
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
    echo "=== RESUME TRAINING SUMMARY ==="
    if [ $# -eq 1 ]; then
        echo "Training type: Resume (from specified model: $SPECIFIED_MODEL_FOLDER)"
    else
        echo "Training type: Resume (from latest model: $LATEST_SCRATCH_MODEL)"
    fi
    echo "Base model directory: $PRETRAINED_MODEL_DIR"
    echo "Resume untuk: Extended training Indonesia"
    echo "Batch size: $BATCH_SIZE"
    echo "Learning rate: $LEARNING_RATE"
    echo "Total epochs: $EPOCHS"
    echo "Output model: ../$OUTPUT_MODEL_DIR/$LATEST_MODEL"
    echo "Test results: ../$OUTPUT_MODEL_DIR/$LATEST_MODEL/test_results"
    echo "========================="
else
    echo ""
    echo "ERROR: TRAINING FAILED!"
    echo "Check the error messages above for details."
    exit 1
fi 