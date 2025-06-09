#!/bin/bash

# Script untuk testing model original (pretrained) dengan data validation Indonesia
# Model original: 190703-214543 (NCEDC dataset)
# Testing dengan data Indonesia untuk comparison baseline

echo "Testing Model Original PhaseNet dengan Data Indonesia"
echo "===================================================="

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

# Set parameters
VALIDATION_DIR="$1"
VALIDATION_LIST="$2"
ORIGINAL_MODEL="model/190703-214543"
OUTPUT_DIR="test_results_original"
BATCH_SIZE=2
TESTING_MIN_PROB=0.3

# Validate input parameters
if [ -z "$VALIDATION_DIR" ] || [ -z "$VALIDATION_LIST" ]; then
    echo "❌ Usage: $0 <VALIDATION_DIR> <VALIDATION_LIST>"
    echo ""
    echo "Example:"
    echo "  $0 data_list_indonesia/npz_files data_list_indonesia/validation_data.csv"
    echo ""
    echo "Purpose: Test original pretrained model dengan data Indonesia"
    echo "         untuk mendapatkan baseline performance comparison."
    exit 1
fi

# Check if files exist
if [ ! -d "$VALIDATION_DIR" ]; then
    echo "❌ Validation directory not found: $VALIDATION_DIR"
    exit 1
fi

if [ ! -f "$VALIDATION_LIST" ]; then
    echo "❌ Validation list not found: $VALIDATION_LIST"
    exit 1
fi

if [ ! -d "$ORIGINAL_MODEL" ]; then
    echo "❌ Original model not found: $ORIGINAL_MODEL"
    echo "   Expected path: $ORIGINAL_MODEL"
    echo "   This should be the pretrained model downloaded from original PhaseNet"
    exit 1
fi

echo ""
echo "KONFIGURASI TESTING MODEL ORIGINAL:"
echo "  Model: $ORIGINAL_MODEL (Original PhaseNet - NCEDC dataset)"
echo "  Validation dir: $VALIDATION_DIR"
echo "  Validation list: $VALIDATION_LIST"
echo "  Output dir: $OUTPUT_DIR"
echo "  Batch size: $BATCH_SIZE"
echo "  Min prob threshold: $TESTING_MIN_PROB (higher for original model)"
echo "  Window size: 3000 samples (30 detik)"
echo ""

# Count validation samples
if [ -f "$VALIDATION_LIST" ]; then
    TOTAL_FILES=$(tail -n +2 "$VALIDATION_LIST" | wc -l)
    echo "Total validation files: $TOTAL_FILES"
fi

echo ""
echo "Creating output directory..."
mkdir -p "$OUTPUT_DIR"

echo "Starting testing original model..."
echo "  (Model ini belum pernah melihat data Indonesia)"
echo "  (Hasil akan menjadi baseline untuk comparison)"
echo ""

# Create logs directory
mkdir -p logs_indonesia/original_test

# Run testing on original model
python phasenet/test_indonesia_3000_original.py \
    --test_dir "$VALIDATION_DIR" \
    --test_list "$VALIDATION_LIST" \
    --model_dir "$ORIGINAL_MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size=$BATCH_SIZE \
    --plot_results \
    --min_prob=$TESTING_MIN_PROB \
    2>&1 | tee logs_indonesia/original_test/testing_output.log

TESTING_EXIT_CODE=$?

echo ""
echo "=============================================================================="

if [ $TESTING_EXIT_CODE -eq 0 ]; then
    echo "✅ Testing original model completed successfully!"
    echo ""
    echo "Output files in: $OUTPUT_DIR/"
    
    # Check generated files
    echo ""
    echo "Generated test files:"
    if [ -f "$OUTPUT_DIR/sliding_window_results.csv" ]; then
        echo "✅ Results CSV: $OUTPUT_DIR/sliding_window_results.csv"
        TOTAL_WINDOWS=$(tail -n +2 "$OUTPUT_DIR/sliding_window_results.csv" | wc -l)
        echo "   Total windows tested: $TOTAL_WINDOWS"
        
        # Show basic statistics
        echo ""
        echo "Basic Statistics (first few rows):"
        head -10 "$OUTPUT_DIR/sliding_window_results.csv" | column -t -s,
    else
        echo "⚠️  Results CSV not found"
    fi
    
    if [ -f "$OUTPUT_DIR/sliding_window_performance.png" ]; then
        echo "✅ Performance plots: $OUTPUT_DIR/sliding_window_performance.png"
    else
        echo "⚠️  Performance plots not found"
    fi
    
    if [ -f "$OUTPUT_DIR/model_info.txt" ]; then
        echo "✅ Model info: $OUTPUT_DIR/model_info.txt"
        echo ""
        echo "Model Information:"
        head -10 "$OUTPUT_DIR/model_info.txt"
    else
        echo "⚠️  Model info not found"
    fi
    
    # Count total files
    OTHER_FILES=$(find "$OUTPUT_DIR" -type f | wc -l)
    echo ""
    echo "Total files in output directory: $OTHER_FILES"
    
    echo ""
    echo "=== ORIGINAL MODEL TESTING SUMMARY ==="
    echo "Model tested: $ORIGINAL_MODEL"
    echo "Model type: Original PhaseNet (NCEDC dataset - belum ada Indonesia)"
    echo "Test data: Indonesia validation dataset"
    echo "Window: 3000 samples (30 detik) dengan sliding window 50% overlap"
    echo "Min prob: $TESTING_MIN_PROB"
    echo "Batch size: $BATCH_SIZE"
    echo ""
    echo "Purpose: Baseline comparison untuk finetuned models"
    echo "Expected: Performance mungkin rendah karena domain mismatch"
    echo "         (Model dilatih dengan NCEDC, ditest dengan Indonesia)"
    echo ""
    echo "Output location: $OUTPUT_DIR/"
    echo "Test log: logs_indonesia/original_test/testing_output.log"
    echo ""
    echo "Files untuk comparison:"
    echo "  - $OUTPUT_DIR/sliding_window_results.csv"
    echo "  - $OUTPUT_DIR/sliding_window_performance.png"
    echo ""
    echo "Next steps:"
    echo "  1. Compare dengan results dari finetuned models"
    echo "  2. Analyze domain adaptation effectiveness"
    echo "  3. Check improvement dari transfer learning"
    echo "======================================"
    
else
    echo "❌ Testing original model failed dengan exit code: $TESTING_EXIT_CODE"
    echo ""
    echo "Check log files untuk debugging:"
    echo "  - Testing output: logs_indonesia/original_test/testing_output.log"
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Check validation data directory dan file list"
    echo "  2. Check original model path: $ORIGINAL_MODEL"
    echo "  3. Pastikan script test_indonesia_3000_original.py ada"
    echo "  4. Check if validation data is properly formatted"
    echo "  5. Check GPU/CPU memory availability"
fi

echo "==============================================================================" 