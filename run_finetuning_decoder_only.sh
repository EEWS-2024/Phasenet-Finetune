#!/bin/bash

# Training script untuk PhaseNet Indonesia dengan Decoder-Only Fine-tuning
# Hanya decoder yang dilatih, encoder dibekukan (frozen)

echo "PhaseNet Indonesia Decoder-Only Fine-tuning dengan Sliding Window 3000 samples"
echo "=============================================================================="

# Activate conda environment phasenet
echo "Activating conda environment phasenet..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phasenet

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "phasenet" ]]; then
    echo "❌ Failed to activate phasenet environment"
    exit 1
fi

echo "✅ Conda environment phasenet activated"

# Disable GPU - use CPU only
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
echo "🖥️  GPU disabled - using CPU only for training"

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
    echo "Training directory tidak ditemukan: $TRAIN_DIR"
    exit 1
fi

if [ ! -f "$TRAIN_LIST" ]; then
    echo "Training list tidak ditemukan: $TRAIN_LIST"
    exit 1
fi

# Check pretrained model
PRETRAINED_MODEL="model/190703-214543"
if [ ! -d "$PRETRAINED_MODEL" ]; then
    echo "Pretrained model tidak ditemukan: $PRETRAINED_MODEL"
    echo "   Pastikan model pretrained ada di direktori model/"
    exit 1
fi

echo "KONFIGURASI DECODER-ONLY FINE-TUNING:"
echo "  Training dir: $TRAIN_DIR"
echo "  Training list: $TRAIN_LIST"
echo "  Pretrained model: $PRETRAINED_MODEL"

if [ -n "$VALID_DIR" ] && [ -n "$VALID_LIST" ]; then
    if [ -d "$VALID_DIR" ] && [ -f "$VALID_LIST" ]; then
        echo "  Validation dir: $VALID_DIR"
        echo "  Validation list: $VALID_LIST"
        VALIDATION_ARGS="--valid_dir $VALID_DIR --valid_list $VALID_LIST"
        HAS_VALIDATION=true
    else
        echo "Validation data tidak valid, training tanpa validation"
        VALIDATION_ARGS=""
        HAS_VALIDATION=false
    fi
else
    echo "  Validation: Tidak ada"
    VALIDATION_ARGS=""
    HAS_VALIDATION=false
fi

echo ""

# Training parameters untuk decoder-only fine-tuning 
EPOCHS=200                   
BATCH_SIZE=256               
LEARNING_RATE=0.0001        
DROP_RATE=0.05             
DECAY_STEP=8               
DECAY_RATE=0.95            
SAVE_INTERVAL=5             

# Testing parameters
MIN_PROB=0.1                

echo "DECODER-ONLY FINE-TUNING PARAMETERS (CPU):"
echo "  Strategy: Encoder FROZEN, Decoder TRAINABLE"
echo "  Window: 3000 samples (30 detik)"
echo "  Sliding window: 50% overlap"
echo "  Epochs: $EPOCHS (dikurangi untuk CPU training)"
echo "  Batch size: $BATCH_SIZE (dikurangi untuk CPU training)"
echo "  Learning rate: $LEARNING_RATE (lebih tinggi untuk decoder)"
echo "  Dropout rate: $DROP_RATE"
echo "  Hardware: 🖥️  CPU ONLY (GPU disabled)"
echo "  Encoder status: ❄️  FROZEN (tidak akan diupdate)"
echo "  Decoder status: 🔥 TRAINABLE (akan diupdate)"
echo "  Testing min prob: $MIN_PROB"
echo "  Validation: $HAS_VALIDATION"
echo ""

# Create output directories
mkdir -p model_indonesia/decoder_only
mkdir -p logs_indonesia/decoder_only

echo "Starting decoder-only fine-tuning (CPU)..."
echo "   (Encoder akan dibekukan, hanya decoder yang dilatih)"
echo "   (Training menggunakan CPU - akan lebih lambat tapi stabil)"
echo "   (Mencegah catastrophic forgetting pada feature extraction)"
echo ""

# Run decoder-only training
python phasenet/train_indonesia_3000_decoder_only.py \
    --train_dir "$TRAIN_DIR" \
    --train_list "$TRAIN_LIST" \
    $VALIDATION_ARGS \
    --model_dir model_indonesia/decoder_only \
    --pretrained_model_path "$PRETRAINED_MODEL" \
    --log_dir logs_indonesia/decoder_only \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --drop_rate $DROP_RATE \
    --decay_step $DECAY_STEP \
    --decay_rate $DECAY_RATE \
    --save_interval $SAVE_INTERVAL \
    --freeze_encoder \
    --summary 2>&1 | tee logs_indonesia/decoder_only/training_output.log

TRAINING_EXIT_CODE=$?

echo ""
echo "=============================================================================="

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Decoder-only fine-tuning completed successfully!"
    echo ""
    echo "Output files:"
    echo "  Models: model_indonesia/decoder_only/decoder3000_YYMMDD-HHMMSS/"
    echo "  Logs: logs_indonesia/decoder_only/"
    echo ""
    
    # Find the latest model
    LATEST_MODEL=$(find model_indonesia/decoder_only -name "decoder3000_*" -type d | sort | tail -1)
    if [ -n "$LATEST_MODEL" ]; then
        echo "Latest model: $LATEST_MODEL"
        
        # Verify that training artifacts exist
        echo ""
        echo "Checking training artifacts..."
        if [ -f "$LATEST_MODEL/training_history.csv" ]; then
            echo "✅ Training history found: $LATEST_MODEL/training_history.csv"
            echo ""
            echo "Training History Summary (Last 5 epochs):"
            tail -5 "$LATEST_MODEL/training_history.csv" | column -t -s,
        else
            echo "⚠️  Training history not found"
        fi
        
        if [ -f "$LATEST_MODEL/loss_curves.png" ]; then
            echo "✅ Loss curves plot found: $LATEST_MODEL/loss_curves.png"
        else
            echo "⚠️  Loss curves plot not found"
        fi
        
        if [ -f "$LATEST_MODEL/frozen_layers.txt" ]; then
            echo "✅ Frozen layers info found: $LATEST_MODEL/frozen_layers.txt"
            echo ""
            echo "Frozen layers summary:"
            head -10 "$LATEST_MODEL/frozen_layers.txt"
        else
            echo "⚠️  Frozen layers info not found"
        fi
        
        echo ""
        echo "Running comprehensive test on validation/training data..."
        
        # Use validation data if provided, otherwise use training data for testing
        if [ "$HAS_VALIDATION" = true ]; then
            TEST_DIR="$VALID_DIR"
            TEST_LIST="$VALID_LIST"
            echo "Testing with validation dataset: $TEST_LIST"
        else
            TEST_DIR="$TRAIN_DIR"
            TEST_LIST="$TRAIN_LIST"
            echo "Testing with training dataset (no validation provided): $TEST_LIST"
        fi
        
        # Create test_results directory first
        mkdir -p "$LATEST_MODEL/test_results"
        
        # Run testing
        echo "Starting testing phase..."
        python phasenet/test_indonesia_3000_decoder_only.py \
            --test_dir "$TEST_DIR" \
            --test_list "$TEST_LIST" \
            --model_dir "$LATEST_MODEL" \
            --output_dir "$LATEST_MODEL/test_results" \
            --batch_size=2 \
            --plot_results \
            --min_prob=$MIN_PROB 2>&1 | tee "$LATEST_MODEL/test_results/testing_output.log"
        
        TESTING_EXIT_CODE=$?
        
        if [ $TESTING_EXIT_CODE -eq 0 ]; then
            echo "✅ Testing completed successfully!"
            echo "Test results directory: $LATEST_MODEL/test_results"
            
            # Check what was generated in test_results
            echo ""
            echo "Generated test files:"
            if [ -f "$LATEST_MODEL/test_results/sliding_window_results.csv" ]; then
                echo "✅ Results CSV: sliding_window_results.csv"
                TOTAL_WINDOWS=$(tail -n +2 "$LATEST_MODEL/test_results/sliding_window_results.csv" | wc -l)
                echo "   Total windows tested: $TOTAL_WINDOWS"
            fi
            
            if [ -f "$LATEST_MODEL/test_results/sliding_window_performance.png" ]; then
                echo "✅ Performance plots: sliding_window_performance.png"
            fi
            
            # Check for any other generated files
            OTHER_FILES=$(find "$LATEST_MODEL/test_results" -type f | wc -l)
            echo "   Total files in test_results: $OTHER_FILES"
            
        else
            echo "❌ Testing failed dengan exit code: $TESTING_EXIT_CODE"
            echo "Check testing log: $LATEST_MODEL/test_results/testing_output.log"
        fi
    else
        echo "❌ Could not find trained model for testing"
        echo "Check training logs: logs_indonesia/decoder_only/training_output.log"
    fi
    
    echo ""
    echo "=== DECODER-ONLY FINE-TUNING SUMMARY ==="
    echo "Training type: Decoder-only fine-tuning (selective transfer learning)"
    echo "Strategy: Encoder frozen ❄️ , Decoder trainable 🔥"
    echo "Base model: $PRETRAINED_MODEL"
    echo "Data: Indonesia sliding window 3000 samples"
    echo "Window size: 3000 samples (30 detik) dengan 50% overlap"
    echo "Batch size: $BATCH_SIZE (lebih besar karena encoder frozen)"
    echo "Learning rate: $LEARNING_RATE (optimized untuk decoder)"
    echo "Total epochs: $EPOCHS"
    echo "Benefits:"
    echo "  ⚡ Training lebih cepat (fewer parameters)"
    echo "  💾 Memory lebih hemat"
    echo "  🛡️  Tidak ada catastrophic forgetting"
    echo "  🎯 Fokus adaptasi output untuk data Indonesia"
    echo ""
    echo "Output model: $LATEST_MODEL"
    echo "Test results: $LATEST_MODEL/test_results"
    echo "Training log: logs_indonesia/decoder_only/training_output.log"
    echo ""
    echo "Expected output files:"
    echo "  - $LATEST_MODEL/training_history.csv (loss data)"
    echo "  - $LATEST_MODEL/loss_curves.png (training plots)"  
    echo "  - $LATEST_MODEL/config.json (model configuration)"
    echo "  - $LATEST_MODEL/frozen_layers.txt (frozen layers info)"
    echo "  - $LATEST_MODEL/test_results/ (testing results)"
    echo ""
    echo "Model siap untuk testing dan inference!"
    echo "========================================"
    
else
    echo "❌ Decoder-only fine-tuning failed dengan exit code: $TRAINING_EXIT_CODE"
    echo ""
    echo "Check log files untuk debugging:"
    echo "  - Training output: logs_indonesia/decoder_only/training_output.log"
    echo "  - Log directory: logs_indonesia/decoder_only/"
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Check data directory dan file list"
    echo "  2. Check GPU memory availability"
    echo "  3. Check pretrained model path"
    echo "  4. Pastikan script train_indonesia_3000_decoder_only.py ada"
    echo "  5. Check if validation data is properly formatted"
fi

echo "==============================================================================" 