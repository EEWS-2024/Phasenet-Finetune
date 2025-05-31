# PhaseNet Indonesia

Repository ini adalah adaptasi dari [PhaseNet](https://github.com/AI4EPS/PhaseNet) yang telah dioptimalkan untuk dataset gempa Indonesia menggunakan **sliding window strategy** dan **transfer learning** dari model pretrained NCEDC dataset.

## **Overview**

PhaseNet Indonesia menggunakan **dua strategi windowing** yang telah dioptimalkan untuk karakteristik seismik Indonesia:

1. **Sliding Window 3000 samples (30s)**
   - Compatible dengan model pretrained PhaseNet
   - Transfer learning dari NCEDC dataset

2. **Fixed Window 170s** - Legacy approach
   - Custom architecture untuk data Indonesia

---

## **Dataset Indonesia**

### **Format Data:**
- **NPZ Files**: `dataset_phasenet_aug/npz_padded/`
- **Training List**: `dataset_phasenet_aug/padded_train_list.csv`
- **Validation List**: `dataset_phasenet_aug/padded_valid_list.csv`

### **Struktur NPZ File:**
| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `data` | `float64` | (30085, 3) | Waveform data (E, N, Z channels) |
| `p_idx` | `int64` | (1, 1) | P-wave arrival index |
| `s_idx` | `int64` | (1, 1) | S-wave arrival index |
| `station_id` | `str` | () | Station ID (e.g., 'BBJI') |
| `t0` | `str` | () | Start timestamp |
| `channel` | `str array` | (3,) | Channel names |

### **Statistik P-S Interval:**
- **Mean**: 36.0 detik
- **Median**: 29.8 detik  
- **P99**: 117.1 detik
- **Maximum**: 240.8 detik
- **Total Files**: 2,053 NPZ files

---

## **Sliding Window Strategy**

### **1. Training Baru dengan Pretrained Model:**
```bash
bash run_finetuning_indonesia.sh \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/npz_padded \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/padded_train_list.csv \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/npz_padded \
    ../augmentasi-data-phasenet-full/dataset_phasenet_aug/padded_valid_list.csv

# Output: model_indonesia/finetuned/sliding3000_YYMMDD-HHMMSS/
```

### **2. Training From Scratch:**
```bash
bash run_training_scratch_indonesia.sh

# Output: model_indonesia/scratch/YYMMDD-HHMMSS/
```

### **3. Training Parameters:**
```bash
EPOCHS=50                 
BATCH_SIZE=16             
LEARNING_RATE=0.00001      
DROP_RATE=0.05             
DECAY_STEP=10              
DECAY_RATE=0.98           
```

### **4. Monitor Training:**
```bash
# Check training progress
tail -f model_indonesia/finetuned/sliding3000_*/training_history.csv

# View loss curves
eog model_indonesia/finetuned/sliding3000_*/loss_curves.png
```

---

## **Sliding Window Strategy - Technical Details**

### **Strategy Overview:**
```
Original Data Indonesia: [--------- 30,000+ samples (300+ detik) ---------]

Sliding Windows (3000 samples each):
Window 1: [--- 3000 samples ---]
Window 2:      [--- 3000 samples ---]  (50% overlap)
Window 3:           [--- 3000 samples ---]
...
Window N:                          [--- 3000 samples ---]
```

### **Benefits:**
- **20x More Training Data**: 2,053 files → 37,050+ training windows
- **Compatible Architecture**: Langsung compatible dengan model pretrained


### **Configuration:**
```python
# DataConfig_Indonesia_3000
window_length = 3000    # 30 seconds (same as pretrained)
window_step = 1500      # 15 seconds (50% overlap)
sampling_rate = 100     # 100 Hz
X_shape = [3000, 1, 3]  # Compatible dengan model pretrained NCEDC
Y_shape = [3000, 1, 3]
```

### **Transfer Learning Process:**

#### **1. Pretrained Model Loading:**
- **Source**: Model `190703-214543` (NCEDC dataset)
- **Loaded**: 57/327 variables (17.4% - core neural network weights)
- **Skipped**: 270 optimizer variables (normal untuk transfer learning)
- **Scaling**: Weights di-scale untuk numerical stability

#### **2. Weight Scaling Strategy:**
```python
# Convolution kernels: 0.1x (numerical stability)
scaled_value = checkpoint_value * 0.1

# Bias terms: 0.01x (conservative scaling)  
scaled_value = checkpoint_value * 0.01

# Batch norm gamma: 0.5x (moderate scaling)
scaled_value = checkpoint_value * 0.5
```

#### **3. Warmup Learning Rate:**
```
Epoch 1: LR = base_lr × 0.1   (10% of base) - Gentle start
Epoch 2: LR = base_lr × 0.55  (55% of base) - Gradual increase
Epoch 3: LR = base_lr × 1.0   (100% of base) - Full learning rate
Epoch 4+: Normal decay schedule
```

---

## **Files & Architecture**

### **Output Directory Structure:**
```
model_indonesia/
├── finetuned/                        # Fine-tuning results (transfer learning)
│   └── sliding3000_YYMMDD-HHMMSS/    # Timestamped model directories
├── scratch/                          # Training from scratch results  
│   └── YYMMDD-HHMMSS/               # Timestamped model directories
└── resume/                           # Resumed training results (legacy)

logs_indonesia/
├── finetuned/                        # Fine-tuning logs
└── scratch/                          # Training from scratch logs
```

### **Core Training Files:**
```
phasenet/
├── train_indonesia_3000.py           # Main training script (sliding window)
├── data_reader_indonesia_sliding.py  # Data reader dengan sliding windows
├── model.py                          # PhaseNet architecture
└── util.py                          # Utility functions

run_finetuning_indonesia.sh        # Training launcher script for finetuning
run_training_scratch_indonesia.sh  # Training launcher script for training from scratch
resume_training_indonesia.sh       # Resume training script
```

### **Configuration Files:**
```
model_indonesia/
├── finetuned/
│   └── sliding3000_YYMMDD-HHMMSS/
├── scratch/
│   └── YYMMDD-HHMMSS/
└── resume/
   └── YYMMDD-HHMMSS/
```

---

## **Model Performance & Validation**

### **Training Data Generation:**
```
Original: 1,950 files × ~19 windows each = 37,050 training windows
Validation: 103 files × ~19 windows each = 1,957 validation windows

Data multiplication: 20x increase dari original dataset
Coverage: 99.7% dari original data dengan sliding windows
```

---

## **Model Comparison**

### **Sliding Window (3000s) vs Fixed Window (170s):**

| Metric | Sliding 3000s | Fixed 170s |
|--------|---------------|------------|
| **Training Data** | 37,050 windows | 2,053 files |
| **Transfer Learning** | NCEDC pretrained | From scratch |
| **Architecture** | Proven & stable | Custom adaptation |
| **Training Time** | Longer (more data) | Shorter |
| **Coverage** | Variable per window | 99%+ guaranteed |
| **P-S Compatibility** | Best for <30s | Best for >30s |

---
