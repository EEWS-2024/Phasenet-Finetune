#!/bin/bash

# Training script untuk PhaseNet Indonesia dengan Sliding Window 3000 samples
# Kompatibel dengan model pretrained 190703-214543 (NCEDC dataset)

echo "🚀 PhaseNet Indonesia Training dengan Sliding Window 3000 samples"
echo "================================================================="

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <train_dir> <train_list> [valid_dir] [valid_list]"
    echo ""
    echo "Example:"
    echo "  $0 data_train_indonesia train_list_indonesia.csv"
    echo "  $0 data_train_indonesia train_list_indonesia.csv data_valid_indonesia valid_list_indonesia.csv"
    echo ""
    exit 1
fi

TRAIN_DIR="$1"
TRAIN_LIST="$2"
VALID_DIR="$3"
VALID_LIST="$4"

# Check if training data exists
if [ ! -d "$TRAIN_DIR" ]; then
    echo "❌ Training directory tidak ditemukan: $TRAIN_DIR"
    exit 1
fi

if [ ! -f "$TRAIN_LIST" ]; then
    echo "❌ Training list tidak ditemukan: $TRAIN_LIST"
    exit 1
fi

# Check pretrained model
PRETRAINED_MODEL="model/190703-214543"
if [ ! -d "$PRETRAINED_MODEL" ]; then
    echo "❌ Pretrained model tidak ditemukan: $PRETRAINED_MODEL"
    echo "   Pastikan model pretrained ada di direktori model/"
    exit 1
fi

echo "📊 KONFIGURASI TRAINING:"
echo "  Training dir: $TRAIN_DIR"
echo "  Training list: $TRAIN_LIST"
echo "  Pretrained model: $PRETRAINED_MODEL"

if [ -n "$VALID_DIR" ] && [ -n "$VALID_LIST" ]; then
    if [ -d "$VALID_DIR" ] && [ -f "$VALID_LIST" ]; then
        echo "  Validation dir: $VALID_DIR"
        echo "  Validation list: $VALID_LIST"
        VALIDATION_ARGS="--valid_dir $VALID_DIR --valid_list $VALID_LIST"
    else
        echo "⚠️  Validation data tidak valid, training tanpa validation"
        VALIDATION_ARGS=""
    fi
else
    echo "  Validation: Tidak ada"
    VALIDATION_ARGS=""
fi

echo ""

# Training parameters
EPOCHS=4                  
BATCH_SIZE=128              
LEARNING_RATE=0.00001     
DROP_RATE=0.05             
DECAY_STEP=10              
DECAY_RATE=0.98            
SAVE_INTERVAL=5            

echo "🎛️  TRAINING PARAMETERS:"
echo "  Window: 3000 samples (30 detik)"
echo "  Strategy: Sliding window (50% overlap)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Dropout rate: $DROP_RATE"
echo "  Transfer learning: NCEDC pretrained model"
echo ""

# Create output directories
mkdir -p model_indonesia_3000
mkdir -p logs_indonesia_3000

echo "🔄 Starting training..."
echo "   (Training akan menggunakan sliding window strategy)"
echo "   (1 file NPZ ~30000 samples → multiple 3000-sample windows)"
echo ""

# Run training
python phasenet/train_indonesia_3000.py \
    --train_dir "$TRAIN_DIR" \
    --train_list "$TRAIN_LIST" \
    $VALIDATION_ARGS \
    --model_dir model_indonesia_3000 \
    --pretrained_model_path "$PRETRAINED_MODEL" \
    --log_dir logs_indonesia_3000 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --drop_rate $DROP_RATE \
    --decay_step $DECAY_STEP \
    --decay_rate $DECAY_RATE \
    --save_interval $SAVE_INTERVAL \
    --summary

TRAINING_EXIT_CODE=$?

echo ""
echo "================================================================="

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    echo ""
    echo "📁 Output files:"
    echo "  Models: model_indonesia_3000/sliding3000_YYMMDD-HHMMSS/"
    echo "  Logs: logs_indonesia_3000/"
    echo ""
    echo "📈 Generated files dalam model directory:"
    echo "  - model checkpoints (.ckpt files)"
    echo "  - training_history.csv (loss data)"
    echo "  - loss_curves.png (training plots)"
    echo "  - config.json (model configuration)"
    echo ""
    echo "🎯 Model siap untuk testing dan inference!"
    echo "   Gunakan model terbaru untuk mendapatkan hasil optimal."
    
    # Find the latest model
    LATEST_MODEL=$(find model_indonesia_3000 -name "sliding3000_*" -type d | sort | tail -1)
    if [ -n "$LATEST_MODEL" ]; then
        echo ""
        echo "📍 Latest model: $LATEST_MODEL"
        
        # Show training history if available
        if [ -f "$LATEST_MODEL/training_history.csv" ]; then
            echo ""
            echo "📊 Training History Summary:"
            tail -5 "$LATEST_MODEL/training_history.csv" | column -t -s,
        fi
    fi
    
else
    echo "❌ Training failed dengan exit code: $TRAINING_EXIT_CODE"
    echo ""
    echo "🔍 Check log files untuk debugging:"
    echo "  - Training output di atas"
    echo "  - Log directory: logs_indonesia_3000/"
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "  1. Check data directory dan file list"
    echo "  2. Check GPU memory availability"
    echo "  3. Check pretrained model path"
    echo "  4. Reduce batch size jika out of memory"
fi

echo "=================================================================" 