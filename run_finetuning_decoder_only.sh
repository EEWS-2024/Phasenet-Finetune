#!/bin/bash

# Training script untuk PhaseNet Indonesia dengan Decoder-Only Fine-tuning
# Hanya decoder yang dilatih, encoder dibekukan (frozen)

echo "PhaseNet Indonesia Decoder-Only Fine-tuning dengan Sliding Window 3000 samples"
echo "=============================================================================="

# Activate conda environment
echo "Activating conda environment phasenet..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phasenet

if [ $? -eq 0 ]; then
    echo "✅ Conda environment phasenet activated"
else
    echo "❌ Failed to activate conda environment phasenet"
    exit 1
fi

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "🚀 GPU detected - using GPU for training"
    GPU_AVAILABLE=true
else
    echo "🖥️  No GPU detected - using CPU only"
    GPU_AVAILABLE=false
fi

# Set parameters
TRAINING_DIR="$1"
TRAINING_LIST="$2"
PRETRAINED_MODEL="model/190703-214543"
VALIDATION_DIR="$3"
VALIDATION_LIST="$4"

EPOCHS=200                 # More epochs untuk GPU
BATCH_SIZE=512            # Larger batch size untuk GPU
LEARNING_RATE=0.0001      # Higher learning rate untuk decoder
DROP_RATE=0.05            

# Learning rate decay settings
# Set DECAY_STEP=0 atau DECAY_RATE=1.0 untuk disable decay
DECAY_STEP=0              # 0 = disable decay, or number of epochs between decay
DECAY_RATE=1           # Learning rate multiplier (0.98 = reduce by 2% each step)
TESTING_MIN_PROB=0.1

echo "KONFIGURASI DECODER-ONLY FINE-TUNING:"
echo "  Training dir: $TRAINING_DIR"
echo "  Training list: $TRAINING_LIST"
echo "  Pretrained model: $PRETRAINED_MODEL"

if [ -n "$VALIDATION_DIR" ] && [ -n "$VALIDATION_LIST" ]; then
    echo "  Validation dir: $VALIDATION_DIR"
    echo "  Validation list: $VALIDATION_LIST"
    VALIDATION_PARAMS="--valid_dir $VALIDATION_DIR --valid_list $VALIDATION_LIST"
    VALIDATION_ENABLED=true
else
    VALIDATION_PARAMS=""
    VALIDATION_ENABLED=false
fi

echo ""
echo "DECODER-ONLY FINE-TUNING PARAMETERS (GPU-optimized):"
echo "  Strategy: Encoder FROZEN, Decoder TRAINABLE"
echo "  Window: 3000 samples (30 detik)"
echo "  Sliding window: 50% overlap"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
if [ "$DECAY_STEP" -gt 0 ] && [ "$(echo "$DECAY_RATE < 1.0" | bc -l)" -eq 1 ]; then
    echo "  Learning rate decay: Every $DECAY_STEP epochs, multiply by $DECAY_RATE"
else
    echo "  Learning rate decay: DISABLED (constant rate)"
fi
echo "  Dropout rate: $DROP_RATE"
echo "  Testing min prob: $TESTING_MIN_PROB"
echo "  Validation: $VALIDATION_ENABLED"

echo ""
echo "Starting decoder-only fine-tuning..."
echo "   (Encoder akan dibekukan, hanya decoder yang dilatih)"
if [ "$GPU_AVAILABLE" = true ]; then
    echo "   (Training menggunakan GPU - akan lebih cepat)"
else
    echo "   (Training menggunakan CPU - akan lebih lambat tapi stabil)"
fi
echo "   (Mencegah catastrophic forgetting pada feature extraction)"
echo ""

# Create logs directory
mkdir -p logs_indonesia/decoder_only

# Run decoder-only fine-tuning
python phasenet/train_indonesia_3000_decoder_only.py \
    --train_dir "$TRAINING_DIR" \
    --train_list "$TRAINING_LIST" \
    --pretrained_model_path "$PRETRAINED_MODEL" \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --drop_rate $DROP_RATE \
    --decay_step $DECAY_STEP \
    --decay_rate $DECAY_RATE \
    $VALIDATION_PARAMS \
    2>&1 | tee logs_indonesia/decoder_only/training_output.log

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
        if [ "$VALIDATION_ENABLED" = true ]; then
            TEST_DIR="$VALIDATION_DIR"
            TEST_LIST="$VALIDATION_LIST"
            echo "Testing with validation dataset: $TEST_LIST"
        else
            TEST_DIR="$TRAINING_DIR"
            TEST_LIST="$TRAINING_LIST"
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
            --min_prob=$TESTING_MIN_PROB 2>&1 | tee "$LATEST_MODEL/test_results/testing_output.log"
        
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