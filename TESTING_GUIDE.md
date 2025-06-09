# PhaseNet Indonesia - Testing Guide

Panduan lengkap untuk testing model original dan membandingkan dengan model yang sudah difinetuning.

## 📋 Overview

Script ini memungkinkan Anda untuk:
1. **Testing model original** (pretrained NCEDC) dengan data Indonesia
2. **Membandingkan hasil** antara model original dengan model finetuned  
3. **Menganalisis efektivitas** dari fine-tuning process

## 🗂️ File Structure

```
PhaseNet/
├── test_original_model.sh              # Script testing model original
├── phasenet/test_indonesia_3000_original.py  # Python script untuk testing
├── compare_model_results.py             # Script comparison antar model
├── TESTING_GUIDE.md                     # Dokumentasi ini
└── test_results_original/               # Output testing model original
```

## 🚀 Quick Start

### 1. Testing Model Original

```bash
# Format: ./test_original_model.sh <VALIDATION_DIR> <VALIDATION_LIST>
./test_original_model.sh data_list_indonesia/npz_files data_list_indonesia/validation_data.csv
```

**Output:**
- `test_results_original/sliding_window_results.csv` - Hasil testing detail
- `test_results_original/sliding_window_performance.png` - Grafik performance  
- `test_results_original/model_info.txt` - Info model yang ditest
- `logs_indonesia/original_test/testing_output.log` - Log testing

### 2. Membandingkan Model

```bash
# Compare original vs finetuned models
python compare_model_results.py \
  --result_dirs test_results_original model_indonesia/decoder_only/decoder3000_YYMMDD-HHMMSS/test_results \
  --output_dir comparison_results \
  --min_prob 0.1
```

**Output:**
- `comparison_results/model_comparison.png` - Grafik comparison lengkap
- `comparison_results/model_comparison.csv` - Data comparison dalam CSV

## 📖 Detailed Usage

### Testing Model Original

#### Purpose
Model original (190703-214543) adalah model PhaseNet yang dilatih dengan data NCEDC (Northern California). Testing ini memberikan baseline performance untuk comparison dengan model yang sudah diadaptasi untuk data Indonesia.

#### Command Breakdown
```bash
./test_original_model.sh <VALIDATION_DIR> <VALIDATION_LIST>
```

**Parameters:**
- `VALIDATION_DIR`: Directory berisi file NPZ data Indonesia
- `VALIDATION_LIST`: File CSV berisi list data validation

**Expected Performance:**
- Performance mungkin **rendah** karena domain mismatch
- Model tidak pernah melihat karakteristik seismik Indonesia
- Gunakan sebagai **baseline** untuk mengukur improvement

#### Example Usage
```bash
# Test dengan validation data
./test_original_model.sh data_list_indonesia/npz_files data_list_indonesia/validation_data.csv

# Jika tidak ada validation data terpisah, gunakan subset training data  
./test_original_model.sh data_list_indonesia/npz_files data_list_indonesia/training_data.csv
```

### Model Comparison

#### Purpose
Membandingkan performance antara model original dengan model yang sudah difinetuning untuk mengukur efektivitas dari domain adaptation.

#### Command Breakdown
```bash
python compare_model_results.py \
  --result_dirs <RESULT_DIR_1> <RESULT_DIR_2> ... \
  --output_dir <OUTPUT_DIR> \
  --min_prob <THRESHOLD>
```

**Parameters:**
- `--result_dirs`: List directory berisi hasil testing (bisa multiple)
- `--output_dir`: Directory output untuk hasil comparison (default: comparison_results)
- `--min_prob`: Threshold probability untuk detection (default: 0.1)

#### Example Scenarios

**Scenario 1: Original vs Decoder-Only Fine-tuned**
```bash
python compare_model_results.py \
  --result_dirs test_results_original model_indonesia/decoder_only/decoder3000_241220-143502/test_results \
  --output_dir comparison_original_vs_decoder
```

**Scenario 2: Multiple Model Comparison**
```bash
python compare_model_results.py \
  --result_dirs \
    test_results_original \
    model_indonesia/decoder_only/decoder3000_241220-143502/test_results \
    model_indonesia/scratch/scratch3000_241220-150302/test_results \
  --output_dir comparison_all_models
```

**Scenario 3: Different Thresholds**
```bash
# Compare dengan threshold lebih ketat
python compare_model_results.py \
  --result_dirs test_results_original model_indonesia/decoder_only/decoder3000_*/test_results \
  --min_prob 0.3 \
  --output_dir comparison_strict_threshold
```

## 📊 Output Analysis

### Testing Results (test_results_original/)

#### sliding_window_results.csv
Berisi hasil testing detail untuk setiap sliding window:

| Column | Description |
|--------|-------------|
| `filename` | Nama file NPZ yang ditest |
| `window_start` | Index mulai window |
| `window_end` | Index akhir window |
| `p_arrival_true` | Ground truth P-wave arrival (jika ada) |
| `s_arrival_true` | Ground truth S-wave arrival (jika ada) |
| `max_p_prob` | Probabilitas maksimum P-wave detection |
| `max_s_prob` | Probabilitas maksimum S-wave detection |
| `max_p_idx` | Index dengan probabilitas P-wave tertinggi |
| `max_s_idx` | Index dengan probabilitas S-wave tertinggi |
| `p_error_samples` | Error prediksi P-wave dalam samples |
| `s_error_samples` | Error prediksi S-wave dalam samples |
| `p_detected` | Apakah P-wave terdeteksi (prob > threshold) |
| `s_detected` | Apakah S-wave terdeteksi (prob > threshold) |

#### sliding_window_performance.png
Grafik comprehensive berisi:
1. **Probability Distributions** - Distribusi probabilitas P dan S wave
2. **Detection Counts** - Jumlah dan persentase detection
3. **Prediction Errors** - Histogram error prediction
4. **Confidence vs Accuracy** - Scatter plot probabilitas vs error
5. **Detections per File** - Breakdown detection per file NPZ
6. **Overall Statistics** - Summary statistik performance

### Comparison Results (comparison_results/)

#### model_comparison.png
Grafik comparison comprehensive berisi:
1. **Detection Rates Comparison** - Bar chart detection rate antar model
2. **Average Probabilities** - Rata-rata confidence antar model
3. **Error Comparison (MAE)** - Mean Absolute Error comparison
4. **P-wave Probability Distributions** - Histogram distribusi P-wave
5. **S-wave Probability Distributions** - Histogram distribusi S-wave  
6. **Summary Statistics** - Tabel summary dan best performers

#### model_comparison.csv
Data comparison dalam format CSV untuk analysis lebih lanjut.

## 🔍 Interpretation Guidelines

### Expected Results

#### Original Model (Baseline)
```
🔴 Expected Low Performance:
- P-wave detection: 20-40%
- S-wave detection: 15-35%  
- Higher prediction errors
- Lower confidence scores
- Reason: Domain mismatch (NCEDC vs Indonesia)
```

#### Decoder-Only Fine-tuned
```
🟡 Expected Moderate Improvement:
- P-wave detection: +10-25% improvement
- S-wave detection: +5-20% improvement
- Better adaptation to Indonesian data
- Faster training, less overfitting risk
```

#### Full Fine-tuned/From Scratch
```
🟢 Expected High Performance:
- P-wave detection: +20-40% improvement
- S-wave detection: +15-35% improvement
- Best adaptation but higher training cost
- Risk of overfitting dengan data terbatas
```

### Performance Metrics

#### Detection Rate
- **Good**: >70% untuk P-wave, >60% untuk S-wave
- **Fair**: 50-70% untuk P-wave, 40-60% untuk S-wave  
- **Poor**: <50% untuk P-wave, <40% untuk S-wave

#### Mean Absolute Error (MAE)
- **Excellent**: <50 samples (0.5 detik)
- **Good**: 50-100 samples (0.5-1.0 detik)
- **Fair**: 100-200 samples (1.0-2.0 detik)
- **Poor**: >200 samples (>2.0 detik)

#### Average Probability
- **High Confidence**: >0.6
- **Medium Confidence**: 0.3-0.6
- **Low Confidence**: <0.3

## 🛠️ Troubleshooting

### Common Issues

#### 1. "No checkpoint found in model directory"
```bash
# Check if original model exists
ls -la model/190703-214543/

# Download if missing (original PhaseNet model required)
```

#### 2. "Validation directory not found"
```bash
# Check data paths
ls -la data_list_indonesia/npz_files/
ls -la data_list_indonesia/validation_data.csv

# Gunakan training data jika validation tidak ada
./test_original_model.sh data_list_indonesia/npz_files data_list_indonesia/training_data.csv
```

#### 3. "Results file not found" saat comparison
```bash
# Pastikan testing sudah selesai dan generate results
ls -la test_results_original/sliding_window_results.csv
ls -la model_indonesia/*/test_results/sliding_window_results.csv
```

#### 4. Memory/GPU Issues
```bash
# Reduce batch size jika out of memory
# Edit test_original_model.sh: BATCH_SIZE=1

# Atau gunakan CPU only
export CUDA_VISIBLE_DEVICES=""
```

### Debug Tips

#### Check Log Files
```bash
# Testing original model
tail -f logs_indonesia/original_test/testing_output.log

# Training logs untuk finetuned models
tail -f logs_indonesia/decoder_only/training_output.log
```

#### Verify Data Format
```bash
# Check NPZ file format
python -c "
import numpy as np
data = np.load('data_list_indonesia/npz_files/FILENAME.npz')
print('Keys:', list(data.keys()))
print('Shape:', data['data'].shape if 'data' in data else 'No data key')
"
```

#### Manual Comparison
```bash
# Quick comparison tanpa plots
python -c "
import pandas as pd
original = pd.read_csv('test_results_original/sliding_window_results.csv')
finetuned = pd.read_csv('MODEL_DIR/test_results/sliding_window_results.csv')

print(f'Original: P={original.p_detected.mean()*100:.1f}%, S={original.s_detected.mean()*100:.1f}%')
print(f'Finetuned: P={finetuned.p_detected.mean()*100:.1f}%, S={finetuned.s_detected.mean()*100:.1f}%')
"
```

## 📈 Next Steps

### 1. Setelah Testing Original
1. Run testing original model dulu
2. Analyze baseline performance  
3. Compare dengan model finetuned
4. Identify improvement areas

### 2. Multiple Model Comparison
1. Test semua model variants (decoder-only, full fine-tuning, scratch)
2. Compare dengan different thresholds
3. Analyze trade-offs (performance vs training time)
4. Choose best model untuk production

### 3. Advanced Analysis
1. Per-station performance analysis
2. Error distribution per magnitude/distance
3. Threshold optimization
4. Ensemble model combination

## 🎯 Best Practices

### Testing Strategy
1. **Consistent Data**: Gunakan validation set yang sama untuk semua model
2. **Multiple Thresholds**: Test dengan berbagai min_prob values
3. **Statistical Significance**: Pastikan sample size cukup besar
4. **Domain Analysis**: Perhatikan karakteristik data Indonesia vs NCEDC

### Performance Evaluation
1. **Not Just Detection Rate**: Perhatikan juga accuracy dan confidence
2. **Error Analysis**: Analyze distribution errors, bukan hanya rata-rata
3. **Practical Metrics**: Consider latency dan computational cost
4. **Real-world Validation**: Test dengan data baru yang belum pernah dilihat

### Documentation
1. **Record Settings**: Save semua parameter dan configuration
2. **Version Control**: Track model versions dan data versions
3. **Comparative Analysis**: Document improvement dan trade-offs
4. **Reproducibility**: Ensure experiments bisa direproduce

---

**💡 Tips:** 
- Testing original model memberikan insight seberapa jauh domain adaptation membantu
- Model original biasanya perform poorly pada data Indonesia - ini normal!
- Focus pada relative improvement, bukan absolute performance
- Consider computational cost vs performance improvement trade-off 