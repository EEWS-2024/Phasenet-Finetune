#!/bin/bash

# Training script untuk PhaseNet Indonesia dengan Sliding Window 3000 samples
# Kompatibel dengan model pretrained 190703-214543 (NCEDC dataset)

echo "PhaseNet Indonesia Training dengan Sliding Window 3000 samples"
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

echo "KONFIGURASI TRAINING:"
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

# Training parameters
EPOCHS=3                  
BATCH_SIZE=256              
LEARNING_RATE=0.00005       
DROP_RATE=0.05             
DECAY_STEP=10              
DECAY_RATE=0.98            
SAVE_INTERVAL=10             

# Testing parameters
MIN_PROB=0.1                # Minimum probability threshold for detection

echo "TRAINING PARAMETERS:"
echo "  Window: 3000 samples (30 detik)"
echo "  Strategy: Sliding window (50% overlap)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Dropout rate: $DROP_RATE"
echo "  Transfer learning: NCEDC pretrained model"
echo "  Testing min prob: $MIN_PROB"
echo "  Validation: $HAS_VALIDATION"
echo ""

# Create output directories
mkdir -p model_indonesia/finetuned
mkdir -p logs_indonesia/finetuned

echo "Starting training..."
echo "   (Training akan menggunakan sliding window strategy)"
echo "   (1 file NPZ ~30000 samples → multiple 3000-sample windows)"
echo ""

# Run training
python phasenet/train_indonesia_3000.py \
    --train_dir "$TRAIN_DIR" \
    --train_list "$TRAIN_LIST" \
    $VALIDATION_ARGS \
    --model_dir model_indonesia/finetuned \
    --pretrained_model_path "$PRETRAINED_MODEL" \
    --log_dir logs_indonesia/finetuned \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --drop_rate $DROP_RATE \
    --decay_step $DECAY_STEP \
    --decay_rate $DECAY_RATE \
    --save_interval $SAVE_INTERVAL \
    --summary 2>&1 | tee logs_indonesia/finetuned/training_output.log

TRAINING_EXIT_CODE=$?

echo ""
echo "================================================================="

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    echo ""
    echo "Output files:"
    echo "  Models: model_indonesia/finetuned/sliding3000_YYMMDD-HHMMSS/"
    echo "  Logs: logs_indonesia/finetuned/"
    echo ""
    
    # Find the latest model
    LATEST_MODEL=$(find model_indonesia/finetuned -name "sliding3000_*" -type d | sort | tail -1)
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
        
        if [ -f "$LATEST_MODEL/config.json" ]; then
            echo "✅ Model config found: $LATEST_MODEL/config.json"
        else
            echo "⚠️  Model config not found"
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
        
        # Run testing with improved error handling and logging
        echo "Starting testing phase..."
        python phasenet/test_indonesia_3000.py \
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
            echo "Possible solutions:"
            echo "  1. Check if model files are complete"
            echo "  2. Reduce batch size for testing"
            echo "  3. Check data format compatibility"
        fi
    else
        echo "❌ Could not find trained model for testing"
        echo "Check training logs: logs_indonesia/finetuned/training_output.log"
    fi
    
    echo ""
    echo "=== FINE-TUNING SUMMARY ==="
    echo "Training type: Fine-tuning (transfer learning)"
    echo "Base model: $PRETRAINED_MODEL"
    echo "Data: Indonesia sliding window 3000 samples"
    echo "Window size: 3000 samples (30 detik) dengan 50% overlap"
    echo "Batch size: $BATCH_SIZE"
    echo "Learning rate: $LEARNING_RATE"
    echo "Total epochs: $EPOCHS"
    echo "Output model: $LATEST_MODEL"
    echo "Test results: $LATEST_MODEL/test_results"
    echo "Training log: logs_indonesia/finetuned/training_output.log"
    echo ""
    echo "Expected output files:"
    echo "  - $LATEST_MODEL/training_history.csv (loss data)"
    echo "  - $LATEST_MODEL/loss_curves.png (training plots)"  
    echo "  - $LATEST_MODEL/config.json (model configuration)"
    echo "  - $LATEST_MODEL/test_results/ (testing results)"
    echo ""
    echo "Model siap untuk testing dan inference!"
    echo "========================="
    
else
    echo "❌ Training failed dengan exit code: $TRAINING_EXIT_CODE"
    echo ""
    echo "Check log files untuk debugging:"
    echo "  - Training output: logs_indonesia/finetuned/training_output.log"
    echo "  - Log directory: logs_indonesia/finetuned/"
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Check data directory dan file list"
    echo "  2. Check GPU memory availability"
    echo "  3. Check pretrained model path"
    echo "  4. Reduce batch size jika out of memory (current: $BATCH_SIZE)"
    echo "  5. Check if validation data is properly formatted"
fi

echo "=================================================================" 